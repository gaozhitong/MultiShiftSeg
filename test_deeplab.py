import os
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from lib.configs.parse_arg import opt, args
from lib.dataset import *
from lib.utils.metric import eval_ood_measure
from train_deeplab import TrainDeepLabOOD
from lib.network.deepv3 import DeepWV3Plus

class TestDeepLabOOD(TrainDeepLabOOD):
    def __init__(self):
        super().__init__() 

    def build_dataset(self):

        self.test_datasets = {
            "RoadAnomaly": RoadAnomaly,
            "RoadAnomaly21": RoadAnomaly21,
            "RoadObstacle21": RoadObstacle21,
            # "MUAD": MUAD,
            # "ACDC_POC": ACDC_POC,
        }

        self.test_tf = Compose([
            ToTensor(),
            Normalize(mean=opt.data.mean, std=opt.data.std),
        ])

        return

    def build_dataloader(self, dataset_name=None):
        
        if dataset_name not in self.test_datasets:
            raise ValueError(f"Unknown test dataset: {dataset_name}")
            
        # Initialize dataset
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
        return

    def build_model(self, class_num=19, parallel=True, weight_path = ''):
        model = DeepWV3Plus(class_num)

        # Use data parallel by default.
        if parallel:
            model = nn.DataParallel(model)

        if weight_path is None:
            self.logger.warning("Please set the weight_path.")
            return model
        
        state_dict = torch.load(weight_path)
        if 'state_dict' in state_dict.keys():
            state_dict = state_dict['state_dict']
        error_message = model.load_state_dict(state_dict, strict=False)
        print(error_message)

        model = model.cuda()
        return model

    def test(self, dataset_name='RoadAnomaly21'):
        """Run testing on specified dataset"""
        self.build_dataloader(dataset_name)
        
        # Ensure model is in eval mode
        if hasattr(self.model, 'module'):
            self.model.module.eval()
        else:
            self.model.eval()
        
        anomaly_scores = []
        ood_gts = []
        
        with torch.no_grad():
            for data in tqdm(self.data_loader, desc=f"Testing {dataset_name}"):
                img, target = data[0].cuda(), data[1].cuda().long()
                
                # Get anomaly scores
                anomaly_score, logit  = self.model(img)
                
                anomaly_scores.append(anomaly_score.cpu().numpy())
                ood_gts.append(target.cpu().numpy())
        
        # Calculate metrics
        metrics = eval_ood_measure(
            np.concatenate(anomaly_scores),
            np.concatenate(ood_gts)
        )
        roc_auc, prc_auc, fpr95 = metrics
        
        # Log results
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

    def test_all(self):
        """Test on all available datasets"""
        results = {}
        for dataset_name in self.test_datasets.keys():
            try:
                self.logger.warning(f"\n{'='*40}\nTesting on {dataset_name}\n{'='*40}")
                results[dataset_name] = self.test(dataset_name)
            except Exception as e:
                self.logger.error(f"Error testing {dataset_name}: {str(e)}")
                continue
                
        # Print summary
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
    # Initialize
    random_init(args.seed)
    
    # Create necessary directories
    os.makedirs(opt.model_dir, exist_ok=True)
    os.makedirs(opt.log_dir, exist_ok=True)
    
    # Set default weight path if not specified
    if not args.weight_path:
        args.weight_path = f'./ckpts/{args.id}/AUPRC_best_model.pth'
        print(f"Loading default weights from: {args.weight_path}")
    
    # Initialize tester
    tester = TestDeepLabOOD()
    
    # Run tests (single dataset or all)
    if args.test_dataset:  # If specific dataset provided
        tester.test(args.test_dataset)
    else:  # Test all datasets by default
        tester.test_all()