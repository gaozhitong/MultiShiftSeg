import os
import random
import cv2
import numpy
import numpy as np
from CGAug.ControlNet.annotator.uniformer.mmseg.datasets import CityscapesDataset, ADE20KDataset
from CGAug.config import Config as cfg
from CGAug.generate_multishift_image import ood_classes_idx
from PIL import Image
import pickle
from typing import List, Tuple


def get_cities():
    """
    Get the cities for the current split and city_batch.
    """
    if cfg.split == "train":
        cities = [
            ['aachen', 'bochum', 'bremen', 'cologne', ],
            ['darmstadt', 'dusseldorf', 'erfurt', 'hamburg', ],
            ['hanover', 'jena', 'krefeld', 'monchengladbach', 'strasbourg', ],
            ['stuttgart', 'tubingen', 'ulm', 'weimar', 'zurich']
        ]
        cities = cities[cfg.city_batch]
    else:
        cities = ["frankfurt", "lindau", "munster"]

    return cities


def prepare_ADE20k():
    """
    Prepare the ADE20K dataset for the generation process.
    """
    # Load ADE20K statistics
    with open(os.path.join(cfg.ADE_root, "ADE20K_2021_17_01", "index_ade20k.pkl"), 'rb') as f:
        ADE20k_static = pickle.load(f)

    # obj2pic indicates whether an object is in the picture.
    ADE20k_static['obj2pic'] = ADE20k_static['objectPresence'] - ADE20k_static['objectIsPart']
    ADE20k_size = len(ADE20k_static['filename'])

    # mapping 3000+ fine-grained ADE20K classes to the 150 semantic categories
    with open("CGAug/static_data/ADE_class_mapping.pkl", "rb") as f:
        ADE_class_mapping = pickle.load(f)

    # Load OOD classes idx
    if cfg.split == "train":
        ood_class_path = "CGAug/static_data/ADE_ood_class_idx.pkl"
    else:
        ood_class_path = "CGAug/static_data/ADE_ood_class_idx_val.pkl"
    with open(ood_class_path, "rb") as f:
        ood_classes_idx = pickle.load(f)
    ood_classes_idx = [idx - 1 for idx in ood_classes_idx]
    return ADE20k_static, ADE20k_size, ADE_class_mapping, ood_classes_idx


def Cityscapes_to_ADE20k(city_label: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    """
    Convert Cityscapes labels to ADE20K labels based on common classes.

    Args:
        city_label: np.ndarray, shape (H, W, 3), RGB image of Cityscapes label.

    return:
        ade_label: np.ndarray, shape (H, W, 3), RGB image of ADE20K label.
        categories: list, the common classes between Cityscapes and ADE20

    """

    city2ade_cate = {
        'unlabeled': "",
        'ego vehicle': "road",
        'rectification border': "",
        'out of roi': "",
        'static': "",
        'dynamic': "",
        'ground': "road",
        'road': "road",
        'sidewalk': "sidewalk",
        'parking': "",
        'rail track': "railing",
        'building': "building",
        'wall': "wall",
        'fence': "fence",
        'guard rail': "fence",
        'bridge': "bridge",
        'tunnel': "",
        'pole': "pole",
        'polegroup': "pole",
        'traffic light': "traffic light",
        'traffic sign': "signboard",
        'vegetation': "tree",
        'terrain': "grass",
        'sky': "sky",
        'person': "person",
        'rider': "person",
        'car': "car",
        'truck': "truck",
        'bus': "bus",
        'caravan': "car",
        'trailer': "truck",
        'train': "bus",
        'motorcycle': "bicycle",
        'bicycle': "bicycle",
        'license plate': "",
    }
    categories = []
    ade_label = city_label.copy()
    for idx, category in enumerate(CityscapesDataset.ALL_CLASSES):
        city_rgb = CityscapesDataset.ALL_PALETTE[idx]
        ade_class = city2ade_cate[category]
        if ade_class != "":
            ade_idx = ADE20KDataset.CLASSES.index(ade_class)
            ade_rgb = ADE20KDataset.PALETTE[ade_idx]
        else:
            ade_rgb = [0, 0, 0]
        mask = np.all(ade_label == city_rgb, axis=-1)
        if mask.sum() > 0:
            categories.append(category)
        ade_label[mask] = ade_rgb

    return ade_label, categories


def paste_on_road(label: np.ndarray,
                  anomaly_mask: np.ndarray,
                  anomaly_rgb: np.ndarray) -> np.ndarray:
    """
    Paste the anomaly mask on the road in the detected map.

    Args:
        label: np.ndarray, shape (H, W, 3), the colored ground truth of the input image.
        anomaly_mask: np.ndarray, shape (H, W), the mask
        anomaly_rgb: np.ndarray, shape (3,), the RGB color of the anomaly.

    return:
        label_: np.ndarray, shape (H, W, 3), the colored ground truth with the anomaly pasted on the road.
    """
    ade_rgb = ADE20KDataset.PALETTE[ADE20KDataset.CLASSES.index('road')]
    road_pixels = np.all(label == ade_rgb, axis=-1)

    # We need to make sure the anomaly won't be generated to close to the edge of the image
    # Here, we initialize a safe margin of 150 pixels
    safe_margin_mark = np.zeros_like(road_pixels)
    safe_margin = 150
    safe_margin_mark[safe_margin: -safe_margin, safe_margin: -safe_margin] = 1
    road_coords = np.column_stack(np.where(road_pixels & safe_margin_mark))

    # If the safe margin is too large, we reduce it by 10 pixels
    # We can at most accept a margin as small as 10 pixels
    while len(road_coords) == 0 and safe_margin > 10:
        print(f"Warning: safe margin {safe_margin} is too large, reduce to {safe_margin - 10} pixels.")
        safe_margin -= 10
        safe_margin_mark = np.zeros_like(road_pixels)
        safe_margin_mark[safe_margin: -safe_margin, safe_margin: -safe_margin] = 1
        road_coords = np.column_stack(np.where(road_pixels & safe_margin_mark))

    if len(road_coords) == 0:
        return label

    center_coord = road_coords[random.randint(0, len(road_coords) - 1)]

    # Calculate the outer rectangle of the anomaly mask
    y_indices, x_indices = np.where(anomaly_mask == 1)
    min_x, min_y = np.min(x_indices), np.min(y_indices)
    max_x, max_y = np.max(x_indices), np.max(y_indices)
    width, height = max_x - min_x + 1, max_y - min_y + 1
    anomaly_mask = anomaly_mask[min_y:max_y + 1, min_x:max_x + 1]

    # Check and adjust the size
    target_size = max(min(500, max(width, height)), 200)
    scale_factor = target_size / max(width, height)
    anomaly_mask_resized = cv2.resize(anomaly_mask, (0, 0), fx=scale_factor, fy=scale_factor,
                                      interpolation=cv2.INTER_NEAREST)

    # Calculate the starting and the ending coordinates
    center_y, center_x = center_coord
    start_y, start_x = max(center_y - anomaly_mask_resized.shape[0] // 2, 0), max(
        center_x - anomaly_mask_resized.shape[1] // 2, 0)
    end_y, end_x = min(start_y + anomaly_mask_resized.shape[0], label.shape[0]), min(
        start_x + anomaly_mask_resized.shape[1], label.shape[1])

    # Calculate starting and ending indices for both anomaly_mask_resized and detected_map
    if start_y == 0:
        mask_start_y = anomaly_mask_resized.shape[0] - end_y
    else:
        mask_start_y = 0

    if end_y == label.shape[0]:
        mask_end_y = label.shape[0] - start_y
    else:
        mask_end_y = anomaly_mask_resized.shape[0]

    if start_x == 0:
        mask_start_x = anomaly_mask_resized.shape[1] - end_x
    else:
        mask_start_x = 0

    if end_x == label.shape[1]:
        mask_end_x = label.shape[1] - start_x
    else:
        mask_end_x = anomaly_mask_resized.shape[1]

    mask = (anomaly_mask_resized == 1)[mask_start_y:mask_end_y, mask_start_x:mask_end_x]
    label_ = label.copy()

    label_[start_y:end_y, start_x:end_x][mask] = anomaly_rgb

    return label_


def paste_anomalies_ade(label: np.ndarray,
                        ADE20k_size: int,
                        ADE20k_static: dict,
                        ADE_class_mapping: dict) -> Tuple[np.ndarray, str, np.ndarray]:
    """
    Randomly select an image from ADE20K, find an OOD class in the image, and paste it on the road.

    Args:
        label: np.ndarray, shape (H, W, 3), the colored ground truth of the input image.
        ADE20k_size: int, the number of images in ADE20K.
        ADE20k_static: dict, the statics of ADE20K.
        ADE_class_mapping: dict, the mapping from fine-grained ADE20K classes to the 150 semantic categories.
    return:
        label_pasted: np.ndarray, shape (H, W, 3), the colored ground truth with the anomaly pasted on the road.
        ood_name: str, the name of the OOD class.
        anomaly_mask: np.ndarray, shape (H, W), the mask of the anomaly.
    """
    while True:
        # randomly select an image from ADE20K
        idx = random.randint(0, ADE20k_size - 1)
        filename = ADE20k_static['filename'][idx]
        folder = ADE20k_static['folder'][idx]
        gt_path = os.path.join(cfg.ADE_root, folder, filename)
        gt_path = gt_path.split('.jpg')[0] + '_seg.png'
        gt = np.array(Image.open(gt_path))

        # get the unique rgb in the image
        unique_rgb = np.unique(gt.reshape(-1, gt.shape[2]), axis=0)
        # get the semantic class index
        unique_class_idx = np.int32(unique_rgb[:, 0] / 10) * 256 + np.int32(unique_rgb[:, 1])
        sem_class_idx = [ADE_class_mapping.get(idx, -1) for idx in unique_class_idx]
        sem_class_idx = [idx - 1 for idx in sem_class_idx]
        # find all the OOD classes in the image
        choices = [i for i, idx in enumerate(sem_class_idx) if idx in ood_classes_idx]
        if len(choices) == 0:
            print(f"Didn't find any OOD classes in the {gt_path}.")
            continue
        # randomly select an OOD class and paste it on the road
        ood_idx = random.choice(choices)
        ood_name = ADE20KDataset.CLASSES[sem_class_idx[ood_idx]]
        anomaly_mask = (gt == unique_rgb[ood_idx]).all(axis=2).astype(np.uint8)
        anomaly_rgb = ADE20KDataset.PALETTE[sem_class_idx[ood_idx]]
        print(f"find {ood_name} in {gt_path}")
        label_pasted = paste_on_road(label, anomaly_mask, anomaly_rgb)
        anomaly_mask = (label_pasted == anomaly_rgb).all(axis=2).astype(np.uint8)
        return label_pasted, ood_name, anomaly_mask


def get_prompt(weathers: List[str],
               places: List[str]) -> Tuple[str, str]:
    """
    Get the prompt for the generation process.

    Args:
        weathers: list, the weather conditions.
        places: list, the places.

    return:
        prompt: str, the prompt for the generation process.
        domain: str, the domain of the image

    """
    p = random.random()
    if p > 0.5:
        template = "An image sampled from various stereo video sequences taken by dash cam."
    else:
        template = "An image sampled from various stereo video sequences taken by dash cam in {PLACE} in a {WEATHER} {TIME}."

    weather_idx = random.randint(0, len(weathers) - 1)
    weather = weathers[weather_idx]
    p_ = random.random()
    if p_ < 0.7:
        time = "day"
    else:
        time = "night"
    place_idx = random.randint(0, len(places) - 1)
    place = places[place_idx]
    domain = "" if p > 0.5 else f"_{weather}_{time}_{place.replace(' ', '_')}"
    return template.format(WEATHER=weather, TIME=time, PLACE=place), domain


def check_anomaly_by_SAM(image: np.ndarray,
                         anomaly_mask: np.ndarray,
                         sam) -> Tuple[np.ndarray, float]:
    """
    Check the anomaly mask by SAM.

    Args:
        image: np.ndarray, the generated image.
        anomaly_mask: np.ndarray, the anomaly mask.
        sam: the SAM model.

    return:
        pred_ood_mask: np.ndarray, the anomaly mask segmented by SAM.
        iou: float, the IoU between the anomaly mask and the SAM segmentation.
    """

    def _bbox(mask: np.array):
        y_indices, x_indices = np.where(mask == 1)
        min_x, min_y = np.min(x_indices), np.min(y_indices)
        max_x, max_y = np.max(x_indices), np.max(y_indices)
        return np.array([[min_x, min_y, max_x, max_y]])

    def _iou(pred, gt):
        intersection = np.logical_and(pred, gt).sum()
        union = np.logical_or(pred, gt).sum()
        return intersection / union

    sam.set_image(image)
    masks, _, _ = sam.predict(
        point_coords=None,
        point_labels=None,
        box=_bbox(anomaly_mask),
        multimask_output=False,
    )

    pred_ood_mask = masks[0]
    iou = _iou(pred_ood_mask, anomaly_mask)

    return pred_ood_mask, iou


def check_anomaly_by_detector(image: np.ndarray,
                              ood_mask: np.ndarray,
                              OODDetector) -> float:
    """
    Check the anomaly mask by OOD detector.

    Args:
        image: np.ndarray, the generated image.
        ood_mask: np.ndarray, the anomaly mask.
        OODDetector: the OOD detector.

    return:
        ood_score: float, the OOD score of the generated image.
    """

    anomaly_score = OODDetector.anomaly_score(image[None])[0]
    ood_score = anomaly_score[ood_mask == 1].mean()

    return ood_score