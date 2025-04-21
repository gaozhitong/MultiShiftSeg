import datetime
import logging
import os
import time
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from lib.configs.parse_arg import opt, args
from lib.dataset import *
from lib.utils.img_utils import *
from lib.network.deepv3 import DeepWV3Plus
from lib.utils.metric import *
import lib.loss as loss_module
from lib.utils import random_init

torch.backends.cudnn.benchmark = True


class TrainDeepLabOOD:
    """Trainer class for DeepLabV3+ with Out-of-Distribution detection."""
    
    def __init__(self) -> None:
        """Initialize the trainer with logging, datasets, model, and loss."""
        super().__init__()
        self.log_init()
        self.best: Dict[str, float] = {}
        self.criterion = self.build_loss()
        self.build_dataset()
        self.model = self.build_model(weight_path=args.weight_path)
        self.since = time.time()

    def build_dataset(self) -> None:
        """Initialize datasets and transformations for training and validation."""
        train_tf = Compose([
            ToTensor(),
            RandCrop(size=(opt.data.crop_size[0], opt.data.crop_size[1])),
            Normalize(mean=opt.data.mean, std=opt.data.std)
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

    def build_model(self, class_num: int = 19, parallel: bool = True, 
                   weight_path: str = '') -> nn.Module:
        """
        Build and configure the DeepWV3Plus model.
        
        Args:
            class_num: Number of classes for segmentation
            parallel: Whether to use DataParallel
            weight_path: Path to pretrained weights
            
        Returns:
            Configured model
        """
        model = DeepWV3Plus(class_num)

        if parallel:
            model = nn.DataParallel(model)

        if not weight_path:
            self.logger.warning(
                "Using pretrained model trained in closed world without OOD. "
                "Please download the model and set weight_path in config file."
            )
            return model.cuda()

        state_dict = torch.load(weight_path)
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        
        # Load state dict with strict=False to handle mismatches
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        self.logger.info(f"Missing keys: {missing_keys}")
        self.logger.info(f"Unexpected keys: {unexpected_keys}")

        # Initialize uncertainty function
        model.module.uncertainty_func_init()
        return model.cuda()

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
        Build the optimizer for training.
        
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

    def update_trainable_params(self) -> None:
        """Update trainable parameters for second stage training."""
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
        self.logger.warning(f'Trainable Params: {names}')
        self.optimizer = self.build_optimizer(params, opt.train.lr)

    def build_loss(self) -> nn.Module:
        """Build the loss function from configuration."""
        Criterion = getattr(loss_module, opt.loss.name)
        return Criterion(opt.loss.params)

    def train(self) -> None:
        """Main training loop."""
        self.writer = SummaryWriter(opt.tb_dir)
        params, names = self.configure_trainable_params()
        self.logger.warning(f'Trainable Params: {names}')
        self.optimizer = self.build_optimizer(params, opt.train.lr)
        self.best['AUPRC'] = -1

        loss_meter = MultiRunningMeter()
        self.model.train()

        for epoch in range(args.start_epoch, opt.train.n_epochs):
            # Two stage training
            if epoch == opt.train.warmup_epoch:
                self.update_trainable_params()

            for id_data, data in enumerate(self.data_loaders['train']):
                img, target = data[0].cuda(), data[1].cuda().long()
                div_img, div_target = data[2].cuda(), data[3].cuda().long()
                
                # Combine main and diverse images
                img = torch.cat([img, div_img], dim=0)
                target = torch.cat([target, div_target], dim=0)

                anomaly_score, logit = self.model(img)
                loss = self.criterion(logit, anomaly_score, target).mean()

                self.log_epoch({'loss': loss.item()}, f"{epoch}_{id_data}")

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            avg_terms = loss_meter.get_metric()
            loss_meter.reset()

            if epoch % 1 == 0:  # Evaluate every epoch
                metrics = self.valid_batch(dl=self.data_loaders['val'])
                self.model.module.train(True)
                self.log_epoch(metrics, epoch)
                
                if metrics['AUPRC'] > self.best['AUPRC']:
                    self.logger.warning('Update best model for AUPRC')
                    self.update_best(metrics['AUPRC'], save_name='AUPRC')

    def valid_batch(self, dl: Optional[DataLoader] = None) -> Dict[str, float]:
        """
        Validate the model on a batch of data.
        
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
                anomaly_score, logit = self.model(img)
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

    def update_best(self, avg_term: float, save_name: str = '') -> None:
        """
        Update the best model checkpoint.
        
        Args:
            avg_term: Metric value that determines if this is the best model
            save_name: Name to use when saving the model
        """
        self.best[save_name] = avg_term
        torch.save(
            self.model.state_dict(),
            f'{opt.model_dir}/{save_name}_best_model.pth'
        )
        self.logger.warning(f'{args.id} saved best model for {save_name}')

    def plot_curves_multi(self, data: Dict[str, Any], epoch: int, phase: str = 'train') -> None:
        """
        Plot multiple curves to TensorBoard.
        
        Args:
            data: Dictionary of metrics to plot
            epoch: Current epoch number
            phase: Phase of training (train/val)
        """
        group_name = 'epoch_verbose'
        for key, value in data.items():
            self.writer.add_scalar(
                f'{group_name}/{phase}_{key}',
                value,
                epoch
            )

    def log_init(self) -> None:
        """Initialize logging configuration."""
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)

        os.makedirs(opt.log_dir, exist_ok=True)
        logfile = os.path.join(opt.log_dir, "log.txt")
        
        # File handler
        fh = logging.FileHandler(logfile)
        fh.setLevel(logging.DEBUG)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.WARNING)

        formatter = logging.Formatter(
            '[%(asctime)s.%(msecs)03d] %(message)s',
            datefmt='%H:%M:%S'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        self.logger.handlers = [fh, ch]

        self.logger.info(str(opt))
        self.logger.info(f'Time: {datetime.datetime.now()}')

    def log_epoch(self, data: Dict[str, float], epoch: int, phase: str = 'train') -> None:
        """
        Log epoch information.
        
        Args:
            data: Dictionary of metrics to log
            epoch: Current epoch number
            phase: Phase of training (train/val)
        """
        log_str = f'{phase} Epoch: {epoch} '
        log_str += ' '.join([f'{k}: {v:.4f}' for k, v in data.items()])
        logging.warning(log_str)

    def log_final(self) -> None:
        """Log final training results and timing."""
        log_str = 'Best ' + ' '.join([f'{k}: {v:.6f}' for k, v in self.best.items()])
        logging.warning(log_str)

        time_elapsed = time.time() - self.since
        logging.warning(
            'Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
                time_elapsed // 60 // 60,
                time_elapsed // 60 % 60,
                time_elapsed % 60
            )
        )


if __name__ == '__main__':
    # Initialize directories and random seeds
    random_init(args.seed)
    os.makedirs(opt.model_dir, exist_ok=True)
    os.makedirs(opt.log_dir, exist_ok=True)

    # Set default weight path if not provided
    if not args.weight_path:
        args.weight_path = '../pretrained_model/DeepLabV3+_WideResNet38_baseline.pth'
        print(f"Load {args.weight_path}")

    # Run training
    ood = TrainDeepLabOOD()
    run_fn = getattr(ood, args.run)
    run_fn()