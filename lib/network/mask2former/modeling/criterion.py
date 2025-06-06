# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/models/detr.py
"""
MaskFormer criterion.
"""
import logging

import torch
import torch.nn.functional as F
from torch import nn

from detectron2.utils.comm import get_world_size
from detectron2.projects.point_rend.point_features import (
    get_uncertain_point_coords_with_randomness,
    point_sample,
)

from ..utils.misc import is_dist_avail_and_initialized, nested_tensor_from_tensor_list
import numpy as np
from lib.configs.config import config as cfg

def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


dice_loss_jit = torch.jit.script(
    dice_loss
)  # type: torch.jit.ScriptModule


def sigmoid_ce_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    return loss.mean(1).sum() / num_masks


sigmoid_ce_loss_jit = torch.jit.script(
    sigmoid_ce_loss
)  # type: torch.jit.ScriptModule


def calculate_uncertainty(logits):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))


class SetCriterion(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses,
                 num_points, oversample_ratio, importance_sample_ratio, ood_loss, margin, deep_supervision):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.extra_loss = None
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

        # pointwise mask loss parameters
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio
        self.ood_loss = ood_loss
        self.margin = margin

        self.deep_supervision = deep_supervision

    def loss_ood(self, outputs, targets, indices, num_masks):

        ood_masks_ = torch.cat([x["ood_mask"].unsqueeze(0) for x in targets], dim=0)
        ood_mask = (ood_masks_ == 1)
        id_mask = (ood_masks_ == 0)
        mask_logits = outputs["pred_masks"]
        class_logits = outputs["pred_logits"]
        class_logits = F.softmax(class_logits, dim=-1)[..., :-1]
        mask_logits = mask_logits.sigmoid()

        logits = torch.einsum("bqc,bqhw->bchw", class_logits, mask_logits)

        if self.ood_loss == "margin":
            score = -torch.max(logits, dim=1)[0]
            score = F.interpolate(score.unsqueeze(1), size=ood_masks_.shape[-2:], mode="bilinear",
                                  align_corners=True).squeeze(1)
            ood_score = score[ood_mask.squeeze(1)]
            id_score = score[id_mask.squeeze(1)]
            loss = torch.pow(id_score, 2).mean()
            if ood_mask.sum() > 0:
                loss = loss + torch.pow(torch.clamp(self.margin - ood_score, min=0.0), 2).mean()
            loss = 0.5 * loss

        elif self.ood_loss == "bce":
            score = -torch.max(logits, dim=1)[0]
            score = F.interpolate(score.unsqueeze(1), size=ood_masks_.shape[-2:], mode="bilinear",
                                  align_corners=True).squeeze(1)
            ood_score = score[ood_mask.squeeze(1)]
            id_score = score[id_mask.squeeze(1)]
            loss = F.binary_cross_entropy_with_logits(id_score, torch.zeros(id_score.size()).cuda(), reduction="mean")
            if ood_mask.sum() > 0:
                loss = loss + F.binary_cross_entropy_with_logits(ood_score, torch.ones(ood_score.size()).cuda(),
                                                                 reduction="mean")
            loss = 0.5 * loss
        elif self.ood_loss == 'RCL':  # RCL: Relative Contrastive Loss
            assert self.extra_loss is not None
            target = np.stack([target['sem_seg'] for target in targets], axis=0)

            logits = logits[:, :19]
            logits = F.interpolate(logits, size=ood_masks_.shape[-2:], mode="bilinear", align_corners=False) #704,704
            logits = logits[:, :, :target.shape[-2], : target.shape[-1]] #700, 700
  
            class_logits = outputs["pred_logits_ood"]
            mask_logits = outputs["pred_masks_ood"]
            class_logits = F.softmax(class_logits, dim=-1)[..., :-1]
            mask_logits = mask_logits.sigmoid()

            logit_balanced = torch.einsum("bqc,bqhw->bchw", class_logits, mask_logits)

            logit_balanced = F.interpolate(logit_balanced, size=ood_masks_.shape[-2:], mode="bilinear",
                                            align_corners=False)
            logit_balanced = logit_balanced[:, :, :target.shape[-2], : target.shape[-1]]

            score = -torch.max(logit_balanced, dim=1)[0] 

            loss = self.extra_loss(logits, score, torch.tensor(target, device=logits.device))
        else:
            raise ValueError("define_ood_loss")

        return {"loss_ood": loss}

    def loss_labels(self, outputs, targets, indices, num_masks):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"].float()

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {"loss_ce": loss_ce}
        return losses

    def get_sampled_logits_targets(self, src_masks, target_masks, sample_way='uncertain'):
        with torch.no_grad():
            # sample point_coords
            if sample_way == 'uncertain':
                point_coords = get_uncertain_point_coords_with_randomness(
                    src_masks,
                    lambda logits: calculate_uncertainty(logits),
                    self.num_points,
                    self.oversample_ratio,
                    self.importance_sample_ratio,
                )
            elif sample_way == 'random':
                point_coords = self.sample_points_randomly(src_masks, self.num_points)
            elif sample_way == 'clean':
                point_coords = self.get_clean_point_coords_with_randomness(
                    src_masks,
                    target_masks,
                    self.num_points,
                    self.oversample_ratio,
                    self.importance_sample_ratio,
                )

            # get gt labels
            point_labels = point_sample(
                target_masks,
                point_coords,
                align_corners=False,
            ).squeeze(1)

        point_logits = point_sample(
            src_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)

        return point_logits, point_labels

    def loss_masks_aug(self, outputs, targets, indices, num_masks, sample_way='uncertain'):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """

        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]

        Half_B = src_masks.shape[0] // 2

        selected_idx = src_idx[0] < Half_B
        batch_idx = src_idx[0][selected_idx]
        mask_idx = src_idx[1][selected_idx]
        batch_idx2 = src_idx[0][~selected_idx] - Half_B
        mask_idx2 = src_idx[1][~selected_idx]

        num_masks_original = (selected_idx).sum()
        num_masks_aug = (~selected_idx).sum()
        assert ((num_masks_original + num_masks_aug) == num_masks)

        src_masks_original = src_masks[:Half_B][(batch_idx, mask_idx)]
        src_masks_aug = src_masks[Half_B:][(batch_idx2, mask_idx2)]

        # src_masks = src_masks[src_idx]

        masks = [t["masks"] for t in targets]
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)

        selected_idx = tgt_idx[0] < Half_B
        batch_idx = tgt_idx[0][selected_idx]
        mask_idx = tgt_idx[1][selected_idx]
        batch_idx2 = tgt_idx[0][~selected_idx] - Half_B
        mask_idx2 = tgt_idx[1][~selected_idx]

        target_masks_original = target_masks[:Half_B][(batch_idx, mask_idx)]
        target_masks_aug = target_masks[Half_B:][(batch_idx2, mask_idx2)]

        # target_masks = target_masks[tgt_idx]

        # No need to upsample predictions as we are using normalized coordinates :)
        # N x 1 x H x W
        src_masks_original = src_masks_original[:, None]
        target_masks_original = target_masks_original[:, None]

        src_masks_aug = src_masks_aug[:, None]
        target_masks_aug = target_masks_aug[:, None]

        point_logits_original, point_labels_original = self.get_sampled_logits_targets(src_masks_original,
                                                                                       target_masks_original,
                                                                                       sample_way='random')  # mv + img
        point_logits_aug, point_labels_aug = self.get_sampled_logits_targets(src_masks_aug, target_masks_aug,
                                                                             sample_way='clean')

        losses = {
            "loss_original_mask": 2*sigmoid_ce_loss_jit(point_logits_original, point_labels_original, num_masks_original),
            "loss_original_dice": 2*dice_loss_jit(point_logits_original, point_labels_original, num_masks_original),
            "loss_aug_mask": sigmoid_ce_loss_jit(point_logits_aug, point_labels_aug, num_masks_aug),
            "loss_aug_dice": dice_loss_jit(point_logits_aug, point_labels_aug, num_masks_aug),
        }

        del src_masks
        del target_masks
        return losses

    def loss_masks(self, outputs, targets, indices, num_masks):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]

        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # No need to upsample predictions as we are using normalized coordinates :)
        # N x 1 x H x W
        src_masks = src_masks[:, None]
        target_masks = target_masks[:, None]

        with torch.no_grad():
            # sample point_coords
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks,
                lambda logits: calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )

            # get gt labels
            point_labels = point_sample(
                target_masks,
                point_coords,
                align_corners=False,
            ).squeeze(1)

        point_logits = point_sample(
            src_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)

        losses = {
            "loss_mask": sigmoid_ce_loss_jit(point_logits, point_labels, num_masks),
            "loss_dice": dice_loss_jit(point_logits, point_labels, num_masks),
        }

        del src_masks
        del target_masks
        return losses

    def sample_points_randomly(self, coarse_logits, num_points):
        num_boxes = coarse_logits.shape[0]  # Number of batches
        # Generate random coordinates for the specified number of points
        point_coords = torch.rand(num_boxes, num_points, 2, device=coarse_logits.device)
        return point_coords

    def get_clean_point_coords_with_randomness(
            self, coarse_logits, targets, num_points, oversample_ratio, importance_sample_ratio
    ):
        importance_sample_ratio = 0.95
        # num_points = 100 * 100
        oversample_ratio = 1 / 0.8

        assert oversample_ratio >= 1
        assert 1 >= importance_sample_ratio >= 0

        num_boxes = coarse_logits.shape[0]
        num_sampled = int(num_points * oversample_ratio) #min(int(num_points * oversample_ratio), coarse_logits.shape[-1] * coarse_logits.shape[-2])
        point_coords = torch.rand(num_boxes, num_sampled, 2, device=coarse_logits.device)
        point_logits = point_sample(coarse_logits, point_coords, align_corners=False)
        point_targets = point_sample(targets, point_coords, align_corners=False)
        point_uncertainties = -F.binary_cross_entropy_with_logits(point_logits, point_targets, reduction="none")
        num_uncertain_points = int(importance_sample_ratio * num_points)
        num_random_points = num_points - num_uncertain_points

        idx = torch.topk(point_uncertainties[:, 0, :], k=num_uncertain_points, dim=1)[1]

        shift = num_sampled * torch.arange(num_boxes, dtype=torch.long, device=coarse_logits.device)
        idx += shift[:, None]
        point_coords = point_coords.view(-1, 2)[idx.view(-1), :].view(
            num_boxes, num_uncertain_points, 2
        )  # [C, N, 2]

        if num_random_points > 0:
            point_coords = torch.cat(
                [
                    point_coords,
                    torch.rand(num_boxes, num_random_points, 2, device=coarse_logits.device),
                ],
                dim=1,
            )

        return point_coords

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_masks):
        loss_map = {
            'labels': self.loss_labels,
            'masks': self.loss_masks_aug \
            if cfg.model.mask2anomaly.mask_loss_with_pixel_selection and len(cfg.data.generated_subdir_names) \
            else self.loss_masks,
            'ood': self.loss_ood,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_masks)

    def forward(self, outputs, targets):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """

        # return {'loss_mask':outputs['pred_masks'].sum()}
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_masks = sum(len(t["labels"]) for t in targets)
        num_masks = torch.as_tensor(
            [num_masks], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_masks))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs and self.deep_supervision:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_masks)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses

    def set_extra_loss(self, extra_loss):
        self.extra_loss = extra_loss

    def __repr__(self):
        head = "Criterion " + self.__class__.__name__
        body = [
            "matcher: {}".format(self.matcher.__repr__(_repr_indent=8)),
            "losses: {}".format(self.losses),
            "weight_dict: {}".format(self.weight_dict),
            "num_classes: {}".format(self.num_classes),
            "eos_coef: {}".format(self.eos_coef),
            "num_points: {}".format(self.num_points),
            "oversample_ratio: {}".format(self.oversample_ratio),
            "importance_sample_ratio: {}".format(self.importance_sample_ratio),
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
