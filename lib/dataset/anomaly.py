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

class RoadAnomaly(Dataset):
    RoadAnomaly_class = namedtuple('RoadAnomalyClass', ['name', 'id', 'train_id', 'hasinstances',
                                                        'ignoreineval', 'color'])
    # --------------------------------------------------------------------------------
    # A list of all Lost & Found labels
    # --------------------------------------------------------------------------------
    labels = [
        RoadAnomaly_class('in-distribution', 0, 0, False, False, (144, 238, 144)),
        RoadAnomaly_class('out-distribution', 1, 1, False, False, (255, 102, 102)),
    ]

    train_id_in = 0
    train_id_out = 1
    num_eval_classes = 19
    label_id_to_name = {label.id: label.name for label in labels}
    train_id_to_name = {label.train_id: label.name for label in labels}
    trainid_to_color = {label.train_id: label.color for label in labels}
    label_name_to_id = {label.name: label.id for label in labels}

    def __init__(self, root='./datasets/road_anomaly', transform=None):
        """Load all filenames."""
        self.transform = transform
        self.root = root
        self.images = []  # list of all raw input images
        self.targets = []  # list of all ground truth TrainIds images
        filenames = sorted(os.listdir(os.path.join(root, 'original')))

        for filename in filenames:
            if os.path.splitext(filename)[1] == '.jpg':
                f_name = os.path.splitext(filename)[0]
                filename_base_img = os.path.join("original", f_name)
                filename_base_labels = os.path.join("labels", f_name)

                self.images.append(os.path.join(self.root, filename_base_img + '.jpg'))
                self.targets.append(os.path.join(self.root, filename_base_labels + '.png'))
        # self.images = sorted(self.images)
        # self.targets = sorted(self.targets)

    def __len__(self):
        """Return number of images in the dataset split."""
        return len(self.images)

    def __getitem__(self, i):
        """Return raw image, trainIds as torch.Tensor or PIL Image"""
        image = Image.open(self.images[i]).convert('RGB')
        target = Image.open(self.targets[i]).convert('L')

        if self.transform is not None:
            image, target = self.transform(image, target)

        f_name = os.path.splitext(os.path.basename(self.images[i]))[0]
        return image, target, f_name

    def __repr__(self):
        """Print some information about dataset."""
        fmt_str = 'Road anomaly Dataset: \n'
        fmt_str += '----Number of images: %d\n' % len(self.images)
        return fmt_str.strip()


class RoadAnomaly21(Dataset):
    RoadAnomaly_class = namedtuple('RoadAnomalyClass', ['name', 'id', 'train_id', 'hasinstances',
                                                        'ignoreineval', 'color'])
    # --------------------------------------------------------------------------------
    # A list of all Lost & Found labels
    # --------------------------------------------------------------------------------
    labels = [
        RoadAnomaly_class('in-distribution', 0, 0, False, False, (144, 238, 144)),
        RoadAnomaly_class('out-distribution', 1, 1, False, False, (255, 102, 102)),
    ]

    train_id_in = 0
    train_id_out = 1
    train_id_ignore = 255
    num_eval_classes = 19
    label_id_to_name = {label.id: label.name for label in labels}
    train_id_to_name = {label.train_id: label.name for label in labels}
    trainid_to_color = {label.train_id: label.color for label in labels}
    label_name_to_id = {label.name: label.id for label in labels}

    def __init__(self, root='./datasets/dataset_AnomalyTrack', transform=None):
        """Load all filenames."""
        self.transform = transform
        self.root = root
        self.images = []  # list of all raw input images
        self.targets = []  # list of all ground truth TrainIds images
        filenames = sorted(os.listdir(os.path.join(root, 'images')))

        for filename in filenames:
            if os.path.splitext(filename)[1] == '.jpg':
                f_name = os.path.splitext(filename)[0]
                filename_base_img = os.path.join("images", f_name)
                filename_base_labels = os.path.join("labels_masks", f_name)

                # Only load Validation Images
                if not os.path.exists(os.path.join(self.root, filename_base_labels + '_labels_semantic.png')):
                    continue

                self.images.append(os.path.join(self.root, filename_base_img + '.jpg'))
                self.targets.append(os.path.join(self.root, filename_base_labels + '_labels_semantic.png'))

    def __len__(self):
        """Return number of images in the dataset split."""
        return len(self.images)

    def __getitem__(self, i):
        """Return raw image, trainIds as torch.Tensor or PIL Image"""
        image = Image.open(self.images[i]).convert('RGB')
        if os.path.exists(self.targets[i]):
            target = Image.open(self.targets[i]).convert('L')
        else:
            # create a new array filled with 255 (white in grayscale)
            image_np = np.array(image)
            target = np.ones_like(image_np)[:, :, 0] * 255
            target = Image.fromarray(target.astype(np.uint8), 'L')

        if self.transform is not None:
            image, target = self.transform(image, target)

        f_name = os.path.splitext(os.path.basename(self.images[i]))[0]
        return image, target, f_name

    def __repr__(self):
        """Print some information about dataset."""
        fmt_str = 'Road anomaly 21 Dataset: \n'
        fmt_str += '----Number of images: %d\n' % len(self.images)
        return fmt_str.strip()


class RoadObstacle21(Dataset):
    RoadAnomaly_class = namedtuple('RoadAnomalyClass', ['name', 'id', 'train_id', 'hasinstances',
                                                        'ignoreineval', 'color'])
    # --------------------------------------------------------------------------------
    # A list of all Lost & Found labels
    # --------------------------------------------------------------------------------
    labels = [
        RoadAnomaly_class('in-distribution', 0, 0, False, False, (144, 238, 144)),
        RoadAnomaly_class('out-distribution', 1, 1, False, False, (255, 102, 102)),
    ]

    train_id_in = 0
    train_id_out = 1
    train_id_ignore = 255
    num_eval_classes = 19
    label_id_to_name = {label.id: label.name for label in labels}
    train_id_to_name = {label.train_id: label.name for label in labels}
    trainid_to_color = {label.train_id: label.color for label in labels}
    label_name_to_id = {label.name: label.id for label in labels}

    def __init__(self, root='./datasets/dataset_ObstacleTrack', transform=None, no_void=False):
        """Load all filenames."""
        self.transform = transform
        self.root = root
        self.images = []  # list of all raw input images
        self.targets = []  # list of all ground truth TrainIds images
        self.no_void = no_void
        filenames = sorted(os.listdir(os.path.join(root, 'images')))

        for filename in filenames:
            if os.path.splitext(filename)[1] == '.webp':
                f_name = os.path.splitext(filename)[0]
                filename_base_img = os.path.join("images", f_name)
                filename_base_labels = os.path.join("labels_masks", f_name)

                # Only load Validation Images
                if not os.path.exists(os.path.join(self.root, filename_base_labels + '_labels_semantic.png')):
                    continue

                self.images.append(os.path.join(self.root, filename_base_img + '.webp'))
                self.targets.append(os.path.join(self.root, filename_base_labels + '_labels_semantic.png'))


    def __len__(self):
        """Return number of images in the dataset split."""
        return len(self.images)

    def __getitem__(self, i):
        """Return raw image, trainIds as torch.Tensor or PIL Image"""
        image = Image.open(self.images[i]).convert('RGB')
        if os.path.exists(self.targets[i]):
            target = Image.open(self.targets[i]).convert('L')
        else:
            # create a new array filled with 255 (white in grayscale)
            image_np = np.array(image)
            target = np.ones_like(image_np)[:, :, 0] * 255
            target = Image.fromarray(target.astype(np.uint8), 'L')

        if self.transform is not None:
            image, target = self.transform(image, target)

        if self.no_void:
            target[target == self.train_id_ignore] = self.train_id_in

        f_name = os.path.splitext(os.path.basename(self.images[i]))[0]
        return image, target, f_name

    def __repr__(self):
        """Print some information about dataset."""
        fmt_str = 'Road Obstacle 21 Dataset: \n'
        fmt_str += '----Number of images: %d\n' % len(self.images)
        return fmt_str.strip()


class MUAD(Dataset):
    def __init__(self, root='./datasets/MUAD_challenge/test_sets/test_OOD', transform=None):
        super(MUAD, self).__init__()
        self.transform = transform
        self.root = root
        self.img_root = os.path.join(self.root, 'leftImg8bit')
        self.gt_root = os.path.join(self.root, 'leftLabel')
        self.images = sorted(glob.glob(os.path.join(self.img_root, "*.png")))  # list of all raw input images
        self.f_names = [os.path.splitext(os.path.basename(img))[0] for img in self.images]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img_dir = self.images[i]
        f_name = self.f_names[i]
        gt_dir = img_dir.replace("leftImg8bit", "leftLabel")
        img = np.array(Image.open(img_dir))
        gt = np.array(Image.open(gt_dir))
        if self.transform:
            img, gt = self.transform(img, gt)
        ood_gt = np.zeros_like(gt)
        ood_gt[gt <= 18] = 0
        ood_gt[(gt == 19) | (gt == 20)] = 1
        ood_gt[gt == 255] = 255
        gt[gt >= 19] = 255
        return img, ood_gt, f_name, gt

    def __repr__(self):
        return f"""
MUAD Dataset
Root: {self.root}
Number of images: {len(self)}
        """


class ACDC_POC(Dataset):
    """
    object_id = {
        'rider,bicycle': '25,33',
        'rider,motorcycle': '25,32',
        'train': 31,
        'bus': 28,
        'person': 24,
        'car': 26,
        'truck': 27,
        'stroller': 100,
       'trolley': 101,
       'garbage bag': 102,
       'wheelie bin': 103,
       'suitcase': 104,
       'skateboard': 105,
       'chair dumped on the street': 106,
       'sofa dumped on the street': 107,
       'furniture dumped on the street': 108,
       'matress dumped on the street': 109,
       'garbage dumped on the street': 110,
       'clothes dumped on the street': 111,
       'cement mixer on the street': 112,
       'cat': 113,
       'dog': 114,
       'bird flying': 115,
       'horse': 116,
       'skunk': 117,
       'sheep': 118,
       'crocodile': 119,
       'alligator': 120,
       'bear': 121,
       'llama': 122,
       'tiger': 123,
       'monkey': 124,

    }
    """

    train_id_in = 0
    train_id_out = 1
    trainId2evalId = [(7, 0),
                      (8, 1),
                      (11, 2),
                      (12, 3),
                      (13, 4),
                      (17, 5),
                      (19, 6),
                      (20, 7),
                      (21, 8),
                      (22, 9),
                      (23, 10),
                      (24, 11),
                      (25, 12),
                      (26, 13),
                      (27, 14),
                      (28, 15),
                      (31, 16),
                      (32, 17),
                      (33, 18)]

    def __init__(self, root="./datasets/acdc_ood/",
                 gt_root="./datasets/acdc_ood/",
                 transform=None,
                 splits=None,
                 domains=None,
                 ):
        """
        For now, ACDC_ODD only support validation set
        splits: valid values are any non-emtpy subset of ['train', 'val', 'test']
        domains: valid values are any non-emtpy subset of ['fog', 'rain', 'snow', 'night']
        """
        if domains is None:
            domains = ['fog', 'rain', 'snow', 'night']
        if splits is None:
            splits = ['val']
        assert isinstance(splits, list), "splits must be a list or None"
        assert isinstance(domains, list), "domains must be a list or None"

        assert len(splits) == 1 and splits[0] == 'val', "ACDC_ODD only support validation set"

        self.transform = transform
        self.root = root
        self.gt_root = gt_root
        self.img_root = os.path.join(self.root, 'rgb_anon_trainvaltest', 'rgb_anon')
        self.gt_root = os.path.join(self.gt_root, 'gt_trainval', 'gt')
        self.images = []
        self.GTs = []  # list of all ground truth images
        self.splits = splits
        self.domains = domains
        self.f_names = []

        for domain in self.domains:
            img_root = os.path.join(self.img_root, domain, splits[0])
            gt_root = os.path.join(self.gt_root, domain, splits[0])
            img_list = sorted(glob.glob(os.path.join(img_root, '*', '*.png')))
            gt_list = sorted(glob.glob(os.path.join(gt_root, '*', '*.png')))
            f_names = [os.path.splitext(os.path.basename(img))[0] for img in img_list]
            self.images += img_list
            self.GTs += gt_list
            self.f_names += f_names

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_dir = self.images[idx]
        gt_dir = self.GTs[idx]
        f_name = self.f_names[idx]
        assert os.path.basename(img_dir) == os.path.basename(gt_dir) == f_name + '.png'
        image = np.array(Image.open(img_dir).convert('RGB'))
        target = np.array(Image.open(gt_dir).convert('L'))
        if self.transform:
            image, target = self.transform(image, target)
        ood_gt = np.zeros_like(target)
        ood_gt[target == 0] = 255
        ood_gt[target > 33] = 1
        target[target > 33] = 255
        # eval_target = np.zeros_like(target)
        eval_target = np.ones_like(target) * 255
        for train_id, eval_id in self.trainId2evalId:
            eval_target[target == train_id] = eval_id
        return image, ood_gt, f_name, eval_target

    def __repr__(self):
        return f"""
            ACDC_ODD Dataset
            Number of images: {len(self)}
            Splits: {self.splits}
            Domains: {self.domains}
                    """
