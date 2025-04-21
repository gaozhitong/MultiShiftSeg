import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn import DataParallel
import torch.nn.functional as F
import logging
import os
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Any

from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
import detectron2.data.transforms as T

from lib.configs.parse_arg import opt, args
from lib.dataset import *
from lib.utils.img_utils import *
from lib.network.deepv3 import *
from lib.utils.metric import *
from lib.utils import random_init
from train_m2f import TrainM2FOOD

torch.backends.cudnn.benchmark = True


class TestM2FOOD(TrainM2FOOD):
    """Test class for Mask2Former-based OOD detection, inheriting from TrainM2FOOD."""
    
    def __init__(self) -> None:
        """Initialize the tester with parent class configuration."""
        super().__init__()

    def build_dataset(self) -> None:
        """Initialize test datasets and transformations."""
        self.test_datasets = {
            "RoadAnomaly": RoadAnomaly,
            "RoadAnomaly21": RoadAnomaly21,
            "RoadObstacle21": RoadObstacle21,
            "MUAD": MUAD,
            # "ACDC_POC": ACDC_POC,
        }

        self.test_tf = Compose([
            ToTensor(),
            Normalize(mean=opt.data.mean, std=opt.data.std),
        ])

    def build_dataloader(self, dataset_name: str) -> None:
        """
        Build dataloader for specified test dataset.
        
        Args:
            dataset_name: Name of the dataset to load
            
        Raises:
            ValueError: If dataset name is not recognized
        """
        if dataset_name not in self.test_datasets:
            raise ValueError(f"Unknown test dataset: {dataset_name}")
            
        ds_class = self.test_datasets[dataset_name]
        ds = ds_class(transform=self.test_tf)
        
        self.logger.warning(f"Loaded test dataset: {dataset_name} with {len(ds)} samples")
        
        self.data_loader = DataLoader(
            ds,
            batch_size=opt.train.valid_batch,
            shuffle=False,
            num_workers=opt.data.num_workers,
            pin_memory=True
        )

    def build_model(self, weight_path: str = '', 
                   config_file: str = 'lib/network/mask2former/configs/mask2former-cityscapes/semantic-segmentation/anomaly_ft.yaml') -> DataParallel:
        """
        Build and configure the model for testing.
        
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
        return model

    def test(self, dataset_name: str = 'RoadAnomaly21') -> Dict[str, Any]:
        """
        Run testing on specified dataset.
        
        Args:
            dataset_name: Name of dataset to test on
            
        Returns:
            Dictionary containing test metrics
        """
        self.build_dataloader(dataset_name)
        
        # Set model to eval mode
        model = self.model.module if hasattr(self.model, 'module') else self.model
        model.eval()
        
        anomaly_scores = []
        ood_gts = []
        
        with torch.no_grad():
            for data in tqdm(self.data_loader, desc=f"Testing {dataset_name}"):
                img, target = data[0].cuda(), data[1].cuda().long()
                input = self.prepare_input(img, target)

                batched_outputs, anomaly_outputs = self.model(input)
                logit = torch.concat([x["sem_seg"][:19].unsqueeze(0).cuda() for x in batched_outputs])
                anomaly_score = self.get_anomaly_score(anomaly_outputs, (logit.shape[-2], logit.shape[-1]))
                
                anomaly_scores.append(anomaly_score.cpu().numpy())
                ood_gts.append(target.cpu().numpy())
        
        # Calculate and log metrics
        roc_auc, prc_auc, fpr95 = eval_ood_measure(
            np.concatenate(anomaly_scores),
            np.concatenate(ood_gts)
        )
        
        self.logger.warning(
            f"Test Results ({dataset_name}) - "
            f"AUROC: {roc_auc:.4f}, "
            f"AUPRC: {prc_auc:.4f}, "
            f"FPR95: {fpr95:.4f}"
        )
        
        return {
            'dataset': dataset_name,
            'AUROC': roc_auc,
            'AUPRC': prc_auc,
            'FPR_TPR95': fpr95
        }

    def test_all(self) -> Dict[str, Dict[str, float]]:
        """
        Test model on all available datasets.
        
        Returns:
            Dictionary containing results for all datasets
        """
        results = {}
        for dataset_name in self.test_datasets.keys():
            try:
                self.logger.warning(f"\n{'='*40}\nTesting on {dataset_name}\n{'='*40}")
                results[dataset_name] = self.test(dataset_name)
            except Exception as e:
                self.logger.error(f"Error testing {dataset_name}: {str(e)}")
                continue
                
        # Print summary table
        self.logger.warning("\n=== FINAL TEST RESULTS ===")
        for dataset, metrics in results.items():
            self.logger.warning(
                f"{dataset:<15} | "
                f"AUROC: {metrics['AUROC']:.4f} | "
                f"AUPRC: {metrics['AUPRC']:.4f} | "
                f"FPR95: {metrics['FPR_TPR95']:.4f}"
            )
        
        return results


if __name__ == '__main__':
    # Initialize directories and random seeds
    random_init(args.seed)
    os.makedirs(opt.model_dir, exist_ok=True)
    os.makedirs(opt.log_dir, exist_ok=True)

    # Set default weight path if not provided
    if not args.weight_path:
        args.weight_path = f'./ckpts/{args.id}/AUPRC_best_model.pth'
        print(f"Loading default weights from: {args.weight_path}")

    # Run testing
    tester = TestM2FOOD()
    if args.test_dataset:  # Test specific dataset
        tester.test(args.test_dataset)
    else:  # Test all datasets
        tester.test_all()