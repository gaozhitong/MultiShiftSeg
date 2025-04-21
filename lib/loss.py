import torch
import torch.nn as nn
import torch.nn.functional as F


class RelContrastiveLoss(nn.Module):
    """Relative Contrastive Loss for anomaly detection.
    
    Args:
        param_dict (dict): Dictionary containing configuration parameters.
    """
    
    def __init__(self, param_dict):
        super().__init__()
        # Initialize base loss function
        self.nll_loss = nn.NLLLoss(reduction='none', ignore_index=255)
        
        # Contrastive learning parameters
        self.inoutaug_contras_margins_tri = param_dict.get('inoutaug_contras_margins_tri', None)
        self.sample_ratio = param_dict.get('sample_ratio', 1)
        
        # Sample selection parameters
        self.conduct_pixel_selection = param_dict.get('conduct_pixel_selection', False)
        self.selection_ratio = param_dict.get('selection_ratio', 1.0)
        
        # Loss weights
        self.ce_weights = param_dict.get('ce_weights', [1, 1])
        self.contras_weight = param_dict.get('contras_weight', 1.0)

        # Class IDs for in-distribution and void classes.
        self.in_id = 99
        self.void_id = 255

    def forward(self, logits, anomaly_score, targets):
        """Compute the relative contrastive loss.
        
        Args:
            logits (torch.Tensor): Model predictions (logits).
            anomaly_score (torch.Tensor): Anomaly scores for each pixel.
            targets (torch.Tensor): Ground truth labels.
            
        Returns:
            torch.Tensor: Computed loss value.
        """
        # Create masks for in-distribution and out-of-distribution pixels
        ood_mask = (targets > self.in_id) & (targets != self.void_id)
        in_mask = (targets < self.in_id)
        
        # Initialize loss
        loss = 0.0
        batch_size = logits.shape[0]
        
        # Prepare targets for in-distribution pixels
        in_targets = targets.clone()
        in_targets[~in_mask] = 255
        in_mask_selected = in_mask.clone()
        
        # Compute cross-entropy loss for original samples
        ce_original = self.nll_loss(F.log_softmax(logits[:batch_size//2]), 
                                   in_targets[:batch_size//2]).mean()
        
        # Compute cross-entropy loss for augmented samples with optional pixel selection
        if self.conduct_pixel_selection and 0.0 < self.selection_ratio < 1.0:
            ce_aug = self._compute_augmented_ce_with_selection(
                logits, in_targets, in_mask_selected, targets, batch_size)
        else:
            ce_aug = self.nll_loss(F.log_softmax(logits[batch_size//2:]), 
                                 in_targets[batch_size//2:]).mean()
            ce_aug = torch.tensor(0.0, device=logits.device) if torch.isnan(ce_aug) else ce_aug
        
        
        # Combine cross-entropy losses
        ce_loss = (self.ce_weights[0] * ce_original + 
                   self.ce_weights[1] * ce_aug)
        loss += ce_loss
        
        # Prepare masks for contrastive loss
        in_mask_original, in_mask_aug = in_mask.clone(), in_mask.clone()
        in_mask_original[batch_size//2:] = False
        in_mask_aug[:batch_size//2] = False
        
        # Compute contrastive loss components
        contrastive_loss = self._compute_contrastive_loss(
            anomaly_score, in_mask_original, in_mask_aug, ood_mask, batch_size)
        
        loss += self.contras_weight * contrastive_loss
        
        return loss
    
    def _compute_augmented_ce_with_selection(self, logits, in_targets, in_mask_selected, targets, batch_size):
        """Compute CE loss for augmented samples with pixel selection."""
        ce_aug = self.nll_loss(F.log_softmax(logits[batch_size//2:]), 
                              in_targets[batch_size//2:]).flatten()
        
        ce_aug_detach = ce_aug.detach()
        ce_aug_detach[in_targets[batch_size//2:].flatten() == 255] = float('inf')
        
        total_num = in_mask_selected[batch_size//2:].sum()
        select_num = int(self.selection_ratio * total_num)
        
        if select_num > 0:
            select_index = torch.topk(ce_aug_detach, select_num, largest=False)[1]
            ce_aug = ce_aug[select_index].mean()
            
            _, height, width = in_mask_selected.shape
            select_index_mask = torch.zeros_like(in_mask_selected[batch_size//2:]).flatten().bool()
            select_index_mask[select_index] = True
            select_index_mask = select_index_mask.reshape(batch_size//2, height, width)
            
            in_mask_selected[batch_size//2:][~select_index_mask] = False
            targets[batch_size//2:][~select_index_mask] = 255
        else:
            ce_aug = torch.tensor(0.0, device=logits.device)
            in_mask_selected[batch_size//2:] = False
            targets[batch_size//2:] = 255
        
        return ce_aug
    
    def _compute_contrastive_loss(self, anomaly_score, in_mask_original, in_mask_aug, ood_mask, batch_size):
        """Compute the three components of contrastive loss."""
        # Sample features for contrastive learning
        original_feature = anomaly_score[in_mask_original]
        aug_feature = anomaly_score[in_mask_aug]
        ood_feature = anomaly_score[ood_mask]
        
        # Apply sampling ratio
        num_samples = self._get_num_samples(in_mask_original, in_mask_aug, ood_mask)
        
        original_feature = original_feature[torch.randperm(original_feature.shape[0])[:num_samples]]
        aug_feature = aug_feature[torch.randperm(aug_feature.shape[0])[:num_samples]]
        ood_feature = ood_feature[torch.randperm(ood_feature.shape[0])[:num_samples]]
        
        # Compute contrastive loss components
        contras_original = F.relu(
            original_feature + self.inoutaug_contras_margins_tri[0] - ood_feature).mean()
        contras_aug = F.relu(
            aug_feature + self.inoutaug_contras_margins_tri[1] - ood_feature).mean()
        
        # Compute in-distribution consistency loss
        # Create same_in_mask by comparing corresponding pairs in original and augmented batches
        same_in_mask = in_mask_original[:batch_size//2] & in_mask_aug[batch_size//2:]
        
        contras_in = F.relu(
            anomaly_score[batch_size//2:] - anomaly_score[:batch_size//2] - 
            self.inoutaug_contras_margins_tri[2])[same_in_mask].mean()
        
        return ( contras_original + contras_aug + contras_in)
    
    def _get_num_samples(self, in_mask_original, in_mask_aug, ood_mask):
        """Calculate number of samples based on sampling ratio."""
        total_pixels = in_mask_original.shape[0] * in_mask_original.shape[1] * in_mask_original.shape[2]
        num_samples = int(total_pixels * self.sample_ratio)
        num_samples = min(num_samples, ood_mask.sum())
        num_samples = min(num_samples, in_mask_original.sum())
        num_samples = min(num_samples, in_mask_aug.sum())
        return num_samples