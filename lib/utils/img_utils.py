import cv2
import random
import numpy as np
import torch
import torchvision.transforms as trans
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode
from abc import ABC, abstractmethod
from functools import partial
from typing import List, Dict
from PIL import Image


class BaseTransformation(ABC):
    def __init__(self):
        self.aug = None
        self.aug_mask = None

    @abstractmethod
    def __call__(self, img, mask, img_gen, mask_gen):
        raise NotImplementedError

    def __repr__(self, *inputs, **kwargs):
        return self.__class__.__name__ + '()'


class Compose:
    """Wraps together multiple image augmentations.
    Should also be used with only one augmentation, as it ensures, that input
    images are of type 'PIL.Image' and handles the augmentation process.
    Args:
        augmentations: List of augmentations to be applied.
    """

    def __init__(self, augmentations):
        """Initializes the composer with the given augmentations."""
        assert isinstance(augmentations, List)
        self.augmentations = augmentations

    def __call__(self, img, mask, img_gen=None, mask_gen=None):
        """Returns images that are augmented with the given augmentations."""
        for a in self.augmentations:
            if isinstance(a, List):
                aug, prob = a
            else:
                aug, prob = a, 1
            if random.random() < prob:
                img, mask, img_gen, mask_gen = aug(img, mask, img_gen, mask_gen)
        if img_gen is not None:
            return img, mask, img_gen, mask_gen
        return img, mask


class SpatialTransformation(BaseTransformation):
    def __call__(self, img, mask, img_gen=None, mask_gen=None, **kwargs):
        """

        Args:
            img: torch.Tensor [C, H, W]
            mask: torch.Tensor [H, W]
            img_gen: torch.Tensor [C, H, W]
            mask_gen: torch.Tensor [H, W]
            **kwargs: arguments for the augmentation
        Returns:
            img_aug: torch.Tensor,
            mask_aug: torch.Tensor,
            img_gen_aug: torch.Tensor if applicable else None,
            mask_gen_aug; torch.Tensor if applicable else None
        """
        if not isinstance(img, torch.Tensor):
            raise TypeError(f'Input image should be a tensor, get {type(img)}')
        img_tensor, mask_tensor = [img], [mask]
        if img_gen is not None:
            img_tensor.append(img_gen)
            mask_tensor.append(mask_gen)
        img_tensor = torch.stack(img_tensor, dim=0)
        mask_tensor = torch.stack(mask_tensor, dim=0)
        img_tensor = self.aug(img_tensor, **kwargs).split(1, dim=0)
        mask_tensor = self.aug_mask(mask_tensor, **kwargs).split(1, dim=0)
        return (img_tensor[0][0], mask_tensor[0][0],
                img_tensor[1][0] if img_gen is not None else img_gen,
                mask_tensor[1][0] if mask_gen is not None else mask_gen)


class NonSpatialTransformation(BaseTransformation):
    def __call__(self, img, mask, img_gen=None, mask_gen=None, **kwargs):
        """

        Args:
            img: torch.Tensor [C, H, W]
            mask: torch.Tensor [H, W]
            img_gen: torch.Tensor [C, H, W]
            mask_gen: torch.Tensor [H, W]
        Returns:
            img_aug: torch.Tensor,
            mask_aug: torch.Tensor (original one),
            img_gen_aug: torch.Tensor if applicable else None,
            mask_gen_aug; torch.Tensor (original one) if applicable else None
        """
        if not isinstance(img, torch.Tensor):
            raise TypeError(f'Input image should be a tensor, get {type(img)}')
        img_tensor = [img]
        if img_gen is not None:
            img_tensor.append(img_gen)
        img_tensor = torch.stack(img_tensor, dim=0)
        img_tensor = self.aug(img_tensor, **kwargs).split(1, dim=0)
        return img_tensor[0][0], mask, img_tensor[1][0] if img_gen is not None else img_gen, mask_gen


class ToTensor(BaseTransformation):
    def __init__(self):
        super().__init__()
        self.aug = trans.ToTensor()
        #self.aug_mask = torch.tensor
        self.aug_mask = ToTensor._to_tensor_mask

    def __call__(self, img, mask, img_gen=None, mask_gen=None, **kwargs):
        img = self.aug(img)
        mask = self.aug_mask(mask)
        if img_gen is not None:
            img_gen = self.aug(img_gen)
            mask_gen = self.aug_mask(mask_gen)
        return img, mask, img_gen, mask_gen

    @staticmethod
    def _to_tensor_mask(mask):
        return torch.tensor(np.array(mask, dtype=np.uint8), dtype=torch.long)


"""
NonSpatialTransformations
"""


class ColorJitter(NonSpatialTransformation):
    def __init__(self):
        super().__init__()
        self.aug = trans.ColorJitter(0.8, 0.8, 0.8, 0.2)


class GaussianBlur(NonSpatialTransformation):
    def __init__(self):
        super().__init__()
        self.aug = trans.GaussianBlur((9, 9), (0.1, 5.0))


class Normalize(NonSpatialTransformation):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean
        self.std = std
        self.aug = trans.Normalize(mean=self.mean, std=self.std)


class Fog(NonSpatialTransformation):
    def __init__(self):
        super().__init__()
        self.aug = self.fog

    def __call__(self, img, mask, img_gen=None, mask_gen=None, **kwargs):
        severity = random.randint(1, 5)
        img, mask, img_gen, mask_gen = super().__call__(img, mask, img_gen, mask_gen, severity=severity)
        return img, mask, img_gen, mask_gen

    @staticmethod
    def fog(img_tensor, severity=1):
        assert len(img_tensor.shape) == 4, 'Input image should be a tensor with shape [B, C, H, W]'
        imgs = list(img_tensor.split(1, dim=0))
        for i, img in enumerate(imgs):
            img = img.squeeze(0)
            img = Fog._fog(img, severity)
            imgs[i] = img.unsqueeze(0)
        return torch.cat(imgs, dim=0)

    @staticmethod
    def _fog(img, severity=1):
        c = [(1.5, 2), (2., 2), (2.5, 1.7), (2.5, 1.5), (3., 1.4)][severity - 1]

        img = np.array(img)
        max_val = img.max()

        img += c[0] * plasma_fractal(mapsize=img.shape[1], wibbledecay=c[1])[:img.shape[0], :img.shape[1]][
            ..., np.newaxis]
        img = np.clip(img * max_val / (max_val + c[0]), 0, 1)
        img = torch.tensor(img)
        return img


class RandSharpness(NonSpatialTransformation):
    def __init__(self):
        super().__init__()
        self.aug = F.adjust_sharpness

    def __call__(self, img, mask, img_gen=None, mask_gen=None, **kwargs):
        sharpness = random.random() * 2  # [0, 2]
        return super().__call__(img, mask, img_gen, mask_gen, sharpness_factor=sharpness)


class AutoContrast(NonSpatialTransformation):
    def __init__(self):
        super().__init__()
        self.aug = F.autocontrast


class Equalize(NonSpatialTransformation):
    def __init__(self):
        super().__init__()
        self.aug = F.equalize

    def __call__(self, img, mask, img_gen=None, mask_gen=None, **kwargs):
        img = (img * 255).to(torch.uint8)
        if img_gen is not None:
            img_gen = (img_gen * 255).to(torch.uint8)
        img, mask, img_gen, mask_gen = super().__call__(img, mask, img_gen, mask_gen)
        img = img.to(torch.float32) / 255
        if img_gen is not None:
            img_gen = img_gen.to(torch.float32) / 255
        return img, mask, img_gen, mask_gen


"""
SpatialTransformations
"""


class Resize(SpatialTransformation):
    def __init__(self, size):
        super().__init__()
        self.aug = trans.Resize(size)
        self.aug_mask = trans.Resize(size, trans.InterpolationMode.NEAREST)


class RandResize(SpatialTransformation):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale
        self.aug = partial(F.resize, interpolation=InterpolationMode.BILINEAR)
        self.aug_mask = partial(F.resize, interpolation=InterpolationMode.NEAREST)

    def __call__(self, img, mask, img_gen=None, mask_gen=None, **kwargs):
        scale = random.choice(self.scale)
        size = (int(img.shape[1] * scale), int(img.shape[2] * scale))
        return super().__call__(img, mask, img_gen, mask_gen, size=size)


class RandCrop(SpatialTransformation):
    def __init__(self, size):
        super().__init__()
        self.size = size
        self.aug = F.crop
        self.aug_mask = F.crop

    def __call__(self, img, mask, img_gen=None, mask_gen=None, **kwargs):
        if img.shape[1] < self.size[0] or img.shape[2] < self.size[1]:
            img = F.resize(img, self.size, interpolation=InterpolationMode.BILINEAR)
        x_start = random.randint(0, img.shape[1] - self.size[0])
        y_start = random.randint(0, img.shape[2] - self.size[1])
        return super().__call__(img, mask, img_gen, mask_gen,
                                top=x_start, left=y_start, height=self.size[0], width=self.size[1])


class RandCropIncludeOOD(SpatialTransformation):
    def __init__(self, size, prob=0.5):
        """

        Args:
            size: Crop Size
            prob: probability that switch between including partial or all OOD pixels in the cropped image
        """

        super().__init__()
        self.size = size
        self.aug = F.crop
        self.aug_mask = F.crop
        self.prob = prob

    def _bbox(self, mask: torch.Tensor):
        """
        Args:
            mask: torch.Tensor [H, W]
        Returns:
            bbox: List[int] [x_start, y_start, x_end, y_end]
        """
        mask = mask.numpy()
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        x_start, x_end = np.where(rows)[0][[0, -1]]
        y_start, y_end = np.where(cols)[0][[0, -1]]
        return [x_start, y_start, x_end, y_end]

    def __call__(self, img, mask, img_gen=None, mask_gen=None, **kwargs):
        assert img_gen is not None and mask_gen is not None, 'Generated image should be provided for RandCropIncludeOOD'
        anomaly_mask = (mask_gen > 100) & (mask < 255)
        h, w = mask_gen.shape
        if anomaly_mask.sum():
            x_start = random.randint(0, img.shape[1] - self.size[0])
            y_start = random.randint(0, img.shape[2] - self.size[1])
            return super().__call__(img, mask, img_gen, mask_gen,
                                    top=x_start, left=y_start, height=self.size[0], width=self.size[1])
        else:
            x_min, y_min, x_max, y_max = self._bbox(anomaly_mask)
            # Note: we assume the crop size is larger than the size of ood object
            if random.random() < self.prob:  # partially include OOD
                lower_x, upper_x = x_min, x_max
                lower_y, upper_y = y_min, y_max
            else:  # completely include OOD
                lower_x, upper_x = x_max, x_min
                lower_y, upper_y = y_max, y_min
            x_start = random.randint(max(0, lower_x - self.size[0]), min(upper_x, h - self.size[0]))
            y_start = random.randint(max(0, lower_y - self.size[1]), min(upper_y, w - self.size[1]))
            return super().__call__(img, mask, img_gen, mask_gen,
                                    top=x_start, left=y_start, height=self.size[0], width=self.size[1])


class RandRotate(SpatialTransformation):
    def __init__(self):
        super().__init__()
        self.aug = partial(F.rotate, interpolation=InterpolationMode.BILINEAR)
        self.aug_mask = partial(F.rotate, interpolation=InterpolationMode.NEAREST)

    def __call__(self, img, mask, img_gen=None, mask_gen=None, **kwargs):
        angle = random.random() * 20 - 10  # [-10, 10]
        return super().__call__(img, mask, img_gen, mask_gen, angle=angle)


class RandHorizontalFlip(SpatialTransformation):
    def __init__(self):
        super().__init__()
        self.aug = F.hflip
        self.aug_mask = F.hflip


class RandVerticalFlip(SpatialTransformation):
    def __init__(self):
        super().__init__()
        self.aug = F.vflip
        self.aug_mask = F.vflip


"""
Helper Functions
"""


def random_scale(img, gt, scales):
    scale = random.choice(scales)
    sh = int(img.shape[0] * scale)
    sw = int(img.shape[1] * scale)
    img = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_LINEAR)
    gt = cv2.resize(gt, (sw, sh), interpolation=cv2.INTER_NEAREST)

    return img, gt, scale


def normalize(img, mean, std):
    # pytorch pretrained model need the input range: 0-1
    img = img.astype(np.float32) / 255.0
    img = img - mean
    img = img / std

    return img

"""
Helper functions for Anomaly Mix (Code adapted from PEBAL)
"""

def paste_coco_objects(image, target, coco_images, coco_targets, ood_scale_array):
    # get coco image
    ood_idx = random.randint(0, len(coco_images) - 1)

    ood_image = np.array(Image.open(coco_images[ood_idx]).convert('RGB'), dtype=np.float32)
    ood_target = np.array(Image.open(coco_targets[ood_idx]).convert('L'), dtype=np.uint8)

    scaled_img, scaled_gt, scale = random_scale(ood_image, ood_target, ood_scale_array)

    city_mix_img, city_mix_gt = mix_func(image, target, scaled_img, scaled_gt)
    return city_mix_img, city_mix_gt

def extract_bboxes(mask):
    boxes = np.zeros([mask.shape[-1], 4], dtype=np.int32)
    for i in range(mask.shape[-1]):
        m = mask[:, :, i]
        # Bounding box.
        horizontal_indicies = np.where(np.any(m, axis=0))[0]
        vertical_indicies = np.where(np.any(m, axis=1))[0]
        if horizontal_indicies.shape[0]:
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            x2 += 1
            y2 += 1
        else:
            x1, x2, y1, y2 = 0, 0, 0, 0
        boxes[i] = np.array([y1, x1, y2, x2])
    return boxes.astype(np.int32)

def mix_func(current_labeled_image=None, current_labeled_mask=None,
                cut_object_image=None, cut_object_mask=None, object_id=254,
                mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)):

    mask = (cut_object_mask != 0) & (cut_object_mask != 255)
    ood_mask = np.expand_dims(mask, axis=2)
    ood_boxes = extract_bboxes(ood_mask)
    ood_boxes = ood_boxes[0, :]
    y1, x1, y2, x2 = ood_boxes[0], ood_boxes[1], ood_boxes[2], ood_boxes[3]
    cut_object_mask = cut_object_mask[y1:y2, x1:x2]
    cut_object_image = cut_object_image[y1:y2, x1:x2, :]
    idx = np.transpose(np.repeat(np.expand_dims(cut_object_mask, axis=0), 3, axis=0), (1, 2, 0))

    h_start_point = random.randint(0, current_labeled_mask.shape[0] - cut_object_mask.shape[0])
    h_end_point = h_start_point + cut_object_mask.shape[0]
    w_start_point = random.randint(0, current_labeled_mask.shape[1] - cut_object_mask.shape[1])
    w_end_point = w_start_point + cut_object_mask.shape[1]

    back_to_tensor = False
    if torch.is_tensor(current_labeled_image):
        back_to_tensor = True

        current_labeled_image = np.transpose(np.array(current_labeled_image), (1, 2, 0))
        current_labeled_mask = np.array(current_labeled_mask)

        cut_object_image = normalize(cut_object_image, mean, std)

    current_labeled_image[h_start_point:h_end_point, w_start_point:w_end_point, :][
        np.where((idx != 0) & (idx != 255))] = \
        cut_object_image[np.where((idx != 0) & (idx != 255))]

    current_labeled_mask[h_start_point:h_end_point, w_start_point:w_end_point][
        np.where((cut_object_mask != 0) & (cut_object_mask != 255))] = \
        cut_object_mask[np.where((cut_object_mask != 0) & (cut_object_mask != 255))]

    if back_to_tensor:
        current_labeled_image = torch.tensor(current_labeled_image).permute(2, 0, 1)
        current_labeled_mask = torch.tensor(current_labeled_mask)

    return current_labeled_image, current_labeled_mask
