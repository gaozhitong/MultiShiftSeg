import glob
import os
from collections import namedtuple
from typing import Any, Callable, Optional, Tuple
from PIL import Image
import torch
from torch.utils.data import Dataset
import random
import numpy as np
from lib.utils.img_utils import *
from lib.utils.utils import *
from typing import List
import glob

class DiverseCityscapes(Dataset):
    """
    The proposed coherent generatived based augmentation (CG-Aug). It consists of both covariate and semantic shifts on the original cityscapes images.

    Cityscapes Dataset http://www.cityscapes-dataset.com/
    Labels based on https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
    """
    CityscapesClass = namedtuple('CityscapesClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                                     'has_instances', 'ignore_in_eval', 'color'])

    labels = [
        CityscapesClass('unlabeled', 0, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('ego vehicle', 1, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('rectification border', 2, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('out of roi', 3, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('static', 4, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('dynamic', 5, 255, 'void', 0, False, True, (111, 74, 0)),
        CityscapesClass('ground', 6, 255, 'void', 0, False, True, (81, 0, 81)),
        CityscapesClass('road', 7, 0, 'flat', 1, False, False, (128, 64, 128)),
        CityscapesClass('sidewalk', 8, 1, 'flat', 1, False, False, (244, 35, 232)),
        CityscapesClass('parking', 9, 255, 'flat', 1, False, True, (250, 170, 160)),
        CityscapesClass('rail track', 10, 255, 'flat', 1, False, True, (230, 150, 140)),
        CityscapesClass('building', 11, 2, 'construction', 2, False, False, (70, 70, 70)),
        CityscapesClass('wall', 12, 3, 'construction', 2, False, False, (102, 102, 156)),
        CityscapesClass('fence', 13, 4, 'construction', 2, False, False, (190, 153, 153)),
        CityscapesClass('guard rail', 14, 255, 'construction', 2, False, True, (180, 165, 180)),
        CityscapesClass('bridge', 15, 255, 'construction', 2, False, True, (150, 100, 100)),
        CityscapesClass('tunnel', 16, 255, 'construction', 2, False, True, (150, 120, 90)),
        CityscapesClass('pole', 17, 5, 'object', 3, False, False, (153, 153, 153)),
        CityscapesClass('polegroup', 18, 255, 'object', 3, False, True, (153, 153, 153)),
        CityscapesClass('traffic light', 19, 6, 'object', 3, False, False, (250, 170, 30)),
        CityscapesClass('traffic sign', 20, 7, 'object', 3, False, False, (220, 220, 0)),
        CityscapesClass('vegetation', 21, 8, 'nature', 4, False, False, (107, 142, 35)),
        CityscapesClass('terrain', 22, 9, 'nature', 4, False, False, (152, 251, 152)),
        CityscapesClass('sky', 23, 10, 'sky', 5, False, False, (70, 130, 180)),
        CityscapesClass('person', 24, 11, 'human', 6, True, False, (220, 20, 60)),
        CityscapesClass('rider', 25, 12, 'human', 6, True, False, (255, 0, 0)),
        CityscapesClass('car', 26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
        CityscapesClass('truck', 27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
        CityscapesClass('bus', 28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
        CityscapesClass('caravan', 29, 255, 'vehicle', 7, True, True, (0, 0, 90)),
        CityscapesClass('trailer', 30, 255, 'vehicle', 7, True, True, (0, 0, 110)),
        CityscapesClass('train', 31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
        CityscapesClass('motorcycle', 32, 17, 'vehicle', 7, True, False, (0, 0, 230)),
        CityscapesClass('bicycle', 33, 18, 'vehicle', 7, True, False, (119, 11, 32)),
        CityscapesClass('license plate', -1, -1, 'vehicle', 7, False, True, (0, 0, 142)),
    ]

    """Normalization parameters"""
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    """Useful information from labels"""
    ignore_in_eval_ids, label_ids, train_ids, train_id2id = [], [], [], []  # empty lists for storing ids
    color_palette_train_ids = [(0, 0, 0) for i in range(256)]
    for i in range(len(labels)):
        if labels[i].ignore_in_eval and labels[i].train_id not in ignore_in_eval_ids:
            ignore_in_eval_ids.append(labels[i].train_id)
    for i in range(len(labels)):
        label_ids.append(labels[i].id)
        if labels[i].train_id not in ignore_in_eval_ids:
            train_ids.append(labels[i].train_id)
            color_palette_train_ids[labels[i].train_id] = labels[i].color
            train_id2id.append(labels[i].id)
    num_label_ids = len(set(label_ids))  # Number of ids
    num_train_ids = len(set(train_ids))  # Number of trainIds
    id2label = {label.id: label for label in labels}
    train_id2label = {label.train_id: label for label in labels}

    def __init__(self, root: str = './datasets/cityscapes', generation_root: str = './datasets/DTWP_ADE_final',
                 coco_root: str = './datasets/coco/coco2017',
                 split: str = "val", mode: str = "gtFine_labelIds",
                 target_type: str = "semantic_train_id", transform: Optional[Callable] = None,
                 anomaly_mix=False, mixup=False, ood_scale_array=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],
                 ) -> None:
        
        self.root = root
        self.generation_root = generation_root
        self.split = split
        self.mode = 'gtFine' if "fine" in mode.lower() else 'gtCoarse'
        self.transform = transform

        self.images_dir = os.path.join(self.root, 'leftImg8bit', self.split)
        self.targets_dir = os.path.join(self.root, 'gtFine', self.split)

        self.generated_images_dirs = os.path.join(self.generation_root,'leftImg8bit', self.split)
        self.generated_targets_dirs = os.path.join(self.generation_root,'gtFine', self.split)

        self.coco_images_dir = os.path.join(coco_root, 'train2017')
        self.coco_targets_dir = os.path.join(coco_root, 'annotations', "oodclass_nocrowd_seg_train2017")

        self.images = []
        self.targets = []
        self.generated_images = []
        self.generated_targets = []
        self.coco_images = []
        self.coco_targets = []
        self.coco_classes = []

        img_dir = self.images_dir
        target_dir = self.targets_dir

        for city_name in os.listdir(img_dir):
            for file_name in os.listdir(os.path.join(img_dir, city_name)):
                if file_name.endswith(".txt"): continue
                target_name = '{}_{}'.format(file_name.split('_leftImg8bit')[0],
                                                self._get_target_suffix(self.mode, target_type))
                gen_img_dir = self.generated_images_dirs
                gen_target_dir = self.generated_targets_dirs
                pattern = os.path.join(gen_img_dir, city_name, "_".join(file_name.split('_')[:3]) + '*')
                matching_files = glob.glob(pattern)
                if len(matching_files) < 1:
                    continue
                sample_idx = random.randint(0, len(matching_files) - 1)
                file_name_ = os.path.basename(matching_files[sample_idx])
                target_name_ = '{}_{}'.format(file_name_.split('_leftImg8bit')[0],
                                                self._get_target_suffix(self.mode, target_type))

                self.generated_images.append(os.path.join(gen_img_dir, city_name, file_name_))
                self.generated_targets.append(os.path.join(gen_target_dir, city_name, target_name_))

                self.images.append(os.path.join(img_dir, city_name, file_name))
                self.targets.append(os.path.join(target_dir, city_name, target_name))


        for root, _, filenames in os.walk(self.coco_targets_dir):
            filenames = sorted(filenames)
            for filename in filenames:
                if os.path.splitext(filename)[-1] == '.png':
                    self.coco_targets.append(os.path.join(root, filename))
                    self.coco_images.append(
                        os.path.join(self.coco_images_dir, (filename.split("_")[1]).split(".")[0] + ".jpg"))
                    self.coco_classes.append(filename.split("_")[0])

        self.anomaly_mix = anomaly_mix
        self.mixup = mixup
        self.ood_scale_array = ood_scale_array

    def __getitem__(self, index: int) -> Tuple:
        image = Image.open(self.images[index]).convert('RGB') 
        gen_image = Image.open(self.generated_images[index]).convert('RGB') 

        target = np.array(Image.open(self.targets[index]), dtype=np.uint8)
        gen_target = np.array(Image.open(self.generated_targets[index]), dtype=np.uint8)

        if self.mixup:
            p = min(random.random(), 0.3) 
            mix_array = (p * np.array(image) + (1 - p) * np.array(gen_image)).astype(np.uint8)
            gen_image = Image.fromarray(mix_array).convert('RGB')

        if self.transform is not None:
            image, target, gen_image, gen_target = self.transform(image, target, gen_image, gen_target)

        if self.anomaly_mix:
            image, target = paste_coco_objects(image, target, self.coco_images, self.coco_targets, self.ood_scale_array)

        return image, target, gen_image, gen_target

    def __len__(self) -> int:
        return len(self.images)

    @staticmethod
    def _get_target_suffix(mode: str, target_type: str) -> str:
        if target_type == 'instance':
            return '{}_instanceIds.png'.format(mode)
        elif target_type == 'semantic_id':
            return '{}_labelIds.png'.format(mode)
        elif target_type == 'semantic_train_id':
            return '{}_labelTrainIds.png'.format(mode)
        elif target_type == 'color':
            return '{}_color.png'.format(mode)
        else:
            print("'%s' is not a valid target type, choose from:\n" % target_type +
                  "['instance', 'semantic_id', 'semantic_train_id', 'color']")
            exit()

