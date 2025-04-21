import logging
import os
import copy
import itertools
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import torch
from torch import nn
from torch.nn import DataParallel
import torch.nn.functional as F
from torch.utils.data import DataLoader

from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg, CfgNode
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.structures import Instances, BitMasks
import detectron2.data.transforms as T
from detectron2.solver.build import maybe_add_gradient_clipping

from lib.configs.parse_arg import opt, args
from lib.dataset import *
from lib.utils.img_utils import *
from lib.network.deepv3 import *
from lib.utils.metric import *
from lib.network.mask2former import add_maskformer2_config
from lib.utils import random_init
import lib.loss as loss_module
from train_deeplab import TrainDeepLabOOD

torch.backends.cudnn.benchmark = True


class TrainM2FOOD(TrainDeepLabOOD):
    """Trainer class for Mask2Former-based OOD detection, inheriting from TrainDeepLabOOD."""
    
    def __init__(self) -> None:
        """Initialize the trainer with logging, datasets, model, and loss."""
        self.log_init()
        self.best: Dict[str, float] = {}
        self.build_dataset()
        self.model = self.build_model(weight_path=args.weight_path)
        self.criterion = self.build_loss()

    def build_dataset(self) -> None:
        """Initialize datasets and transformations for training and validation."""
        train_tf = Compose([
            [ToTensor(), 1.0],
            [ColorJitter(), 0.5],
            [GaussianBlur(), 0.5],
            [RandSharpness(), 0.5],
            [AutoContrast(), 0.5],
            [Equalize(), 0.5],
            [RandResize(scale=[0.7, 0.8, 0.9, 1.0]), 0.5],
            [RandRotate(), 0.5],
            [RandHorizontalFlip(), 0.5],
            [RandVerticalFlip(), 0.5],
            [RandCrop(size=(opt.data.crop_size[0], opt.data.crop_size[1])), 1.0],
            [Normalize(mean=opt.data.mean, std=opt.data.std), 1.0],
        ])
        
        test_tf = Compose([
            ToTensor(),
            Normalize(mean=opt.data.mean, std=opt.data.std),
        ])

        train_ds = DiverseCityscapes(
            split="train",
            transform=train_tf,
            anomaly_mix=opt.data.anomaly_mix,
            mixup=opt.data.mixup,
        )
        val_ds = RoadAnomaly21(transform=test_tf)

        self.data_loaders = {
            'train': DataLoader(
                train_ds,
                batch_size=opt.train.train_batch,
                drop_last=True,
                num_workers=opt.data.num_workers,
                shuffle=True,
                pin_memory=True
            ),
            'val': DataLoader(
                val_ds,
                batch_size=opt.train.valid_batch,
                drop_last=True,
                shuffle=False
            )
        }

    def build_model(self, weight_path: str = '',
                   config_file: str = 'lib/network/mask2former/configs/mask2former-cityscapes/semantic-segmentation/anomaly_ft.yaml') -> DataParallel:
        """
        Build and configure the model.
        
        Args:
            weight_path: Path to model weights
            config_file: Path to model config file
            
        Returns:
            Configured model wrapped in DataParallel
            
        Raises:
            AssertionError: If weight path is not provided
            FileNotFoundError: If weight path doesn't exist
        """
        self.logger = logging.getLogger()
        assert weight_path != '', 'Please provide the path to the trained/pretrained model'
        if not os.path.isfile(weight_path):
            raise FileNotFoundError('The provided weight path does not exist')
        
        self.weight_path = weight_path
        self.cfg = self.setup(config_file)
        self.aug = T.ResizeShortestEdge(
            [self.cfg.INPUT.MIN_SIZE_TEST, self.cfg.INPUT.MIN_SIZE_TEST],
            self.cfg.INPUT.MAX_SIZE_TEST
        )
        
        model = build_model(self.cfg)
        model = DataParallel(model)
        DetectionCheckpointer(model).load(self.weight_path)

        self.logger.warning("Initializing OOD head from Original classifier weights")

        # Initialize OOD head from original classifier weights
        weight = model.module.sem_seg_head.predictor.class_embed.weight.data.clone()
        model.module.sem_seg_head.predictor.class_embed2.weight.data = weight

        bias = model.module.sem_seg_head.predictor.class_embed.bias.data.clone()
        model.module.sem_seg_head.predictor.class_embed2.bias.data = bias
    
        return model

    def setup(self, config_file: str) -> CfgNode:
        """
        Set up model configuration.
        
        Args:
            config_file: Path to config file
            
        Returns:
            Configured CfgNode object
        """
        cfg = get_cfg()
        cfg.set_new_allowed(True)
        add_deeplab_config(cfg)
        add_maskformer2_config(cfg)
        cfg.merge_from_file(config_file)
        
        # Configure Mask2Anomaly settings
        cfg.MODEL.MASK2ANOMALY = CfgNode(new_allowed=True)
        cfg.MODEL.MASK2ANOMALY.USE_DEFAULT_LOSS = opt.model.mask2anomaly.use_official_loss
        cfg.MODEL.MASK2ANOMALY.DEEP_SUPERVISION = opt.model.mask2anomaly.deep_supervision
        
        if opt.model.mask2anomaly.replace_official_odd_loss_with_RCL:
            cfg.MODEL.MASK_FORMER.OOD_LOSS = 'RCL'
            
        if hasattr(opt.loss.params, 'mask2anomaly_loss_weight'):
            cfg.MODEL.MASK_FORMER.CLASS_WEIGHT = opt.loss.params.mask2anomaly_loss_weight.get(
                'class_weight', cfg.MODEL.MASK_FORMER.CLASS_WEIGHT)
            cfg.MODEL.MASK_FORMER.MASK_WEIGHT = opt.loss.params.mask2anomaly_loss_weight.get(
                'mask_weight', cfg.MODEL.MASK_FORMER.MASK_WEIGHT)
            cfg.MODEL.MASK_FORMER.DICE_WEIGHT = opt.loss.params.mask2anomaly_loss_weight.get(
                'dice_weight', cfg.MODEL.MASK_FORMER.DICE_WEIGHT)
            cfg.MODEL.MASK_FORMER.OOD_WEIGHT = opt.loss.params.mask2anomaly_loss_weight.get(
                'ood_weight', cfg.MODEL.MASK_FORMER.OOD_WEIGHT)
            
        cfg.freeze()
        return cfg

    def configure_trainable_params(self) -> Tuple[List[torch.Tensor], List[str]]:
        """
        Configure which model parameters should be trained.
        
        Returns:
            Tuple containing:
                - List of trainable parameters
                - List of parameter names
        """
        params = []
        names = []
        for name, param in self.model.named_parameters():
            if any(i in name for i in opt.model.trainable_params_name):
                param.requires_grad = True
                params.append(param)
                names.append(name)
            else:
                param.requires_grad = False

        return params, names
    
    def build_optimizer(self, params: List[torch.Tensor], lr: float) -> torch.optim.Optimizer:
        """
        Build standard Adam optimizer.
        
        Args:
            params: Parameters to optimize
            lr: Learning rate
            
        Returns:
            Configured optimizer
        """
        return torch.optim.Adam(
            params,
            lr=lr,
            weight_decay=opt.train.weight_decay
        )

    def build_m2f_optimizer(self, params: List[Dict[str, Any]]) -> torch.optim.Optimizer:
        """
        Build Mask2Former-specific optimizer with custom parameter groups.
        
        Args:
            params: List of parameter groups
            
        Returns:
            Configured optimizer
            
        Raises:
            NotImplementedError: If unsupported optimizer type is specified
        """
        cfg = self.cfg
        weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
        weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED

        defaults = {
            "lr": cfg.SOLVER.BASE_LR,
            "weight_decay": cfg.SOLVER.WEIGHT_DECAY
        }

        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )

        params = []
        memo = set()
        for module_name, module in self.model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad or value in memo:
                    continue
                memo.add(value)

                hyperparams = copy.copy(defaults)
                if "backbone" in module_name:
                    hyperparams["lr"] *= cfg.SOLVER.BACKBONE_MULTIPLIER
                if ("relative_position_bias_table" in module_param_name or 
                    "absolute_pos_embed" in module_param_name):
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed
                    
                params.append({"params": [value], **hyperparams})

        def maybe_add_full_model_gradient_clipping(optim):
            """Add gradient clipping if configured."""
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED and
                cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model" and
                clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
            
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
            
        return optimizer

    def update_trainable_params(self) -> None:
        """Update trainable parameters and optimizer for second stage training."""
        self.logger.warning(
            f"Change trainable_params_name from {opt.model.trainable_params_name} "
            f"to {opt.model.trainable_params_name_update}"
        )
        self.logger.warning(
            f"Change lr from {opt.train.lr} to {opt.train.lr_update}"
        )

        opt.model.trainable_params_name = opt.model.trainable_params_name_update
        opt.train.lr = opt.train.lr_update
        
        params, names = self.configure_trainable_params()
        self.logger.warning('Trainable Params: ' + str(names))
        self.optimizer = self.build_m2f_optimizer(params)
        self.model.module.use_official_loss = True

    def build_loss(self) -> nn.Module:
        """Build and configure the loss function."""
        Criterion = getattr(loss_module, opt.loss.name)
        criterion = Criterion(opt.loss.params)
        self.model.module.criterion.set_extra_loss(criterion)
        return criterion

    def prepare_input(self, image: torch.Tensor, target: torch.Tensor,
                     ignore_label: int = 255, ood_label: int = 254,
                     label_threshold: int = 100) -> List[Dict[str, Any]]:
        """
        Prepare input data for Mask2Former model.
        
        Args:
            image: Input image tensor
            target: Target segmentation mask
            ignore_label: Label for ignored pixels
            ood_label: Label for OOD pixels
            label_threshold: Threshold for valid class labels
            
        Returns:
            List of dataset dictionaries for each batch item
        """
        B, C, H, W = image.shape
        input_data = []

        for b in range(B):
            dataset_dict = {'image': image[b]}
            
            if target is not None:
                sem_seg_gt = target[b].cpu().numpy()
                dataset_dict['sem_seg'] = sem_seg_gt
                
                # Prepare instances
                instances = Instances((H, W))
                classes = np.unique(sem_seg_gt)
                classes = classes[classes < label_threshold]
                instances.gt_classes = torch.tensor(classes, dtype=torch.int64)

                # Prepare masks
                masks = [sem_seg_gt == class_id for class_id in classes]
                ood_masks = [(sem_seg_gt > label_threshold) & (sem_seg_gt != ignore_label)]

                if len(masks) == 0:
                    instances.gt_masks = torch.zeros((0, H, W))
                else:
                    masks = BitMasks(torch.stack([
                        torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks
                    ]))
                    instances.gt_masks = masks.tensor

                dataset_dict["instances"] = instances

                # Prepare OOD masks
                if len(ood_masks) == 0:
                    ood_masks = torch.zeros((0, H, W))
                else:
                    ood_masks = BitMasks(torch.stack([
                        torch.from_numpy(np.ascontiguousarray(x.copy())) for x in ood_masks
                    ]))
                    ood_masks = ood_masks.tensor

                dataset_dict["ood_mask"] = ood_masks

            input_data.append(dataset_dict)

        return input_data

    def get_anomaly_score(self, other_outputs: Dict[str, torch.Tensor],
                         size: Tuple[int, int]) -> torch.Tensor:
        """
        Calculate anomaly score from model outputs.
        
        Args:
            other_outputs: Dictionary containing model outputs
            size: Target output size (H, W)
            
        Returns:
            Anomaly score tensor
        """
        class_logits = other_outputs["pred_logits_ood"]
        mask_logits = other_outputs["pred_masks_ood"]

        class_probs = torch.nn.functional.softmax(class_logits, dim=-1)[..., :-1]
        mask_probs = mask_logits.sigmoid()

        uncertainty_logit = torch.einsum("bqc,bqhw->bchw", class_probs, mask_probs)
        uncertainty_logit = uncertainty_logit[:, :, :size[0], :size[1]]
        return 1 - torch.max(uncertainty_logit, dim=1)[0]

    def set_train_mode(self) -> None:
        """Set model to appropriate training mode."""
        self.model.train(True)
        self.model.module.backbone.train(False)

    def train(self) -> None:
        """Main training loop."""
        params, names = self.configure_trainable_params()
        self.logger.warning('Trainable Params: ' + str(names))
        self.optimizer = self.build_optimizer(params, opt.train.lr)
        self.best['AUPRC'] = -1

        loss_meter = MultiRunningMeter()
        self.set_train_mode()

        for epoch in range(args.start_epoch, opt.train.n_epochs):
            # Two stage training
            if epoch == opt.train.warmup_epoch:
                self.update_trainable_params()

            for id_data, data in enumerate(self.data_loaders['train']):
                img, target = data[0].cuda(), data[1].cuda().long()
                div_img, div_target = data[2].cuda(), data[3].cuda().long()
                img = torch.cat([img, div_img], dim=0)
                target = torch.cat([target, div_target], dim=0)

                input = self.prepare_input(img, target)

                if epoch < opt.train.warmup_epoch: 
                    batched_outputs, anomaly_outputs = self.model(input)
                    logit = torch.concat([
                        x["sem_seg"][:19].unsqueeze(0).cuda() 
                        for x in batched_outputs
                    ])
                    anomaly_score = self.get_anomaly_score(
                        anomaly_outputs,
                        (logit.shape[-2], logit.shape[-1])
                    )
                    loss = self.criterion(logit, anomaly_score, target).mean()
                else:
                    losses = self.model(input)
                    loss = sum(losses.values())

                self.log_epoch({'loss': loss.item()}, f"{epoch}_{id_data}")

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            avg_terms = loss_meter.get_metric()
            loss_meter.reset()

            if epoch % 1 == 0:
                metrics = self.valid_batch(dl=self.data_loaders['val'])
                self.set_train_mode()
                self.log_epoch(metrics, epoch)
                if metrics['AUPRC'] > self.best['AUPRC']:
                    self.logger.warning('update best model for AUPRC')
                    self.update_best(metrics['AUPRC'], save_name='AUPRC')

    def valid_batch(self, dl: Optional[DataLoader] = None) -> Dict[str, float]:
        """
        Run validation on a batch of data.
        
        Args:
            dl: DataLoader for validation data
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.model.module.train(False)
        anomaly_score_list = []
        ood_gts_list = []
        
        with torch.no_grad():
            for data in dl:
                img, target = data[0].cuda(), data[1].cuda().long()
                input = self.prepare_input(img, target)

                batched_outputs, anomaly_outputs = self.model(input)
                logit = torch.concat([
                    x["sem_seg"][:19].unsqueeze(0).cuda() 
                    for x in batched_outputs
                ])
                anomaly_score = self.get_anomaly_score(
                    anomaly_outputs,
                    (logit.shape[-2], logit.shape[-1])
                )

                ood_gts_list.extend(target.cpu().numpy())
                anomaly_score_list.extend(anomaly_score.cpu().numpy())

        ood_gts = np.array(ood_gts_list)
        anomaly_scores = np.array(anomaly_score_list)
        roc_auc, prc_auc, fpr95 = eval_ood_measure(anomaly_scores, ood_gts)

        return {
            'AUROC': roc_auc,
            'AUPRC': prc_auc,
            'FPR_TPR95': fpr95
        }


if __name__ == '__main__':
    # Initialize directories and random seeds
    random_init(args.seed)
    os.makedirs(opt.model_dir, exist_ok=True)
    os.makedirs(opt.log_dir, exist_ok=True)

    # Set default weight path if not provided
    if not args.weight_path:
        args.weight_path = "../pretrained/bt-f-xl.pth"
        print(f"Load {args.weight_path}")

    # Run training
    ood = TrainM2FOOD()
    run_fn = getattr(ood, args.run)
    run_fn()