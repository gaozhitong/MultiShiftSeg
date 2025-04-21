import os
import sys
import cv2
import torch
import einops
import logging
import pathlib

from tqdm import tqdm
from PIL import Image
from glob import glob
from typing import Optional, Tuple
from pytorch_lightning import seed_everything
from segment_anything import sam_model_registry, SamPredictor

from CGAug.config import Config as cfg
from CGAug.ControlNet.annotator.util import resize_image
from CGAug.ControlNet.cldm.model import create_model, load_state_dict
from CGAug.ControlNet.cldm.ddim_hacked import DDIMSampler
from CGAug.generation_utils import Cityscapes_to_ADE20k, paste_anomalies_ade, prepare_ADE20k, get_prompt, get_cities, \
    check_anomaly_by_SAM, check_anomaly_by_detector

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib.utils.img_utils import *
from lib.configs.parse_arg import opt
from lib.method_module import Mask2Anomaly

seed = 214748365
seed_everything(seed)


class Semantic2ImageGenerator:
    def __init__(self, config):
        assert config.split in ["train", "val"]
        assert config.city_batch in range(4)
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(filename=os.path.join(config.log_dir, "log.log"), level=logging.INFO)

        self.img_dir = str(os.path.join(config.img_dir, config.split))
        self.mask_dir = str(os.path.join(config.mask_dir, config.split))
        self.save_img_dir = str(os.path.join(config.save_img_dir, config.split))
        self.save_mask_dir = str(os.path.join(config.save_mask_dir, config.split))
        pathlib.Path(self.save_img_dir).mkdir(parents=True, exist_ok=True)

        # Create CGAug model
        self.model = create_model('CGAug/models/cldm_v15.yaml').cpu()
        self.model.load_state_dict(load_state_dict(config.controlnet_weight_path, location='cuda'))
        self.model = self.model.cuda()
        self.ddim_sampler = DDIMSampler(self.model)

        self.ADE20k_static, self.ADE20k_size, self.ADE_class_mapping, self.ood_classes_idx = prepare_ADE20k()

        # Load SAM model
        self.sam = sam_model_registry["vit_h"](checkpoint=config.SAM_path).to(device="cuda")
        self.predictor = SamPredictor(self.sam)

        # Load OOD detector
        self.OODDetector = Mask2Anomaly(weight_path=config.anomaly_weight_path)

        self.image_transform = Compose([ToTensor(), Normalize(mean=opt.data.mean, std=opt.data.std)])

    @torch.no_grad()
    def generate(self,
                 input_image: np.ndarray,
                 prompt: str,
                 a_prompt: str,
                 n_prompt: str,
                 seed: Optional[int] = -1,
                 image_resolution: Optional[int] = 512,
                 ddim_steps: Optional[int] = 50) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str, float]:
        """
        Generate images with both domain and semantic shift given prompts.

        Args:
            input_image: np.ndarray, shape (H, W, 3), colored ground truth with Cityscapes' palette.
            prompt: str, the prompt for the image generation.
            a_prompt: str, positive prompt.
            n_prompt: str, negative prompt.
            seed: int, the seed for the random number generator.
            image_resolution: int, the resolution of the generated image.
            ddim_steps: int, the number of steps for the DDIM sampling.

        return:
            label: np.ndarray, colored ground truth of the input image,
            label_SAM: np.ndarray, colored ground truth with the anomaly segmented by SAM,
            label_pasted: np.ndarray, colored ground truth with the anomaly pasted on the road,
            generation: np.ndarray, the generated image,
            novel_class: str, the name of the OOD class,
            iou: float, the IoU between the anomaly mask and the SAM segmentation.

        """
        # Convert the palette of Cityscapes to ADE20K
        label, categories = Cityscapes_to_ADE20k(input_image)
        # Paste anomalies on the road
        label_pasted, novel_class, anomaly_mask = paste_anomalies_ade(label, self.ADE20k_size,
                                                                      self.ADE20k_static,
                                                                      self.ADE_class_mapping)
        categories.append(novel_class)

        img = resize_image(input_image, image_resolution)

        H, W, C = img.shape
        semantic_mask = cv2.resize(label_pasted, (W, H), interpolation=cv2.INTER_NEAREST)
        anomaly_mask = cv2.resize(anomaly_mask, (W, H), interpolation=cv2.INTER_NEAREST)
        anomaly_mask.astype('bool')

        # CGAug generation
        control = torch.from_numpy(semantic_mask.copy()).float().cuda() / 255.0
        control = torch.stack([control], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()
        if seed == -1:
            seed = random.randint(0, 65535)

        if cfg.save_memory:
            self.model.low_vram_shift(is_diffusing=False)

        categories = "There is a {} accidentally staying on the road.".format(novel_class)

        cond = {"c_concat": [control],
                "c_crossattn": [self.model.get_learned_conditioning([f"{prompt}, {categories}, {a_prompt}"])]}
        un_cond = {"c_concat": [control], "c_crossattn": [self.model.get_learned_conditioning([n_prompt])]}
        shape = (4, H // 8, W // 8)

        if cfg.save_memory:
            self.model.low_vram_shift(is_diffusing=True)

        self.model.control_scales = [1.0] * 13
        samples, intermediates = self.ddim_sampler.sample(ddim_steps, 1,
                                                          shape, cond, verbose=False, eta=0.0,
                                                          unconditional_guidance_scale=9.0,
                                                          unconditional_conditioning=un_cond)
        if cfg.save_memory:
            self.model.low_vram_shift(is_diffusing=False)

        generation = self.model.decode_first_stage(samples)
        generation = (einops.rearrange(generation, 'b c h w -> b h w c') * 127.5 + 127.5)
        generation = generation.cpu().numpy().clip(0, 255).astype(np.uint8)[0]

        ret = self.auto_filtering(generation, anomaly_mask, label)
        if not ret[0]:
            return self.generate(input_image, prompt, a_prompt, n_prompt, seed, image_resolution, ddim_steps)
        _, label_SAM, iou = ret
        return label, label_SAM, label_pasted, generation, novel_class, iou

    def auto_filtering(self,
                       generation: np.ndarray,
                       anomaly_mask: np.ndarray,
                       label: np.ndarray) -> Tuple[bool, np.ndarray, float] or Tuple[bool]:
        """
        Filter the generated image by SAM and OOD detector.

        Args:
            generation: np.ndarray, the generated image.
            anomaly_mask: np.ndarray, the anomaly mask.
            label: np.ndarray, the colored ground truth of the input image.

        return:
            success: bool, whether the generation is successful.
            label_SAM: np.ndarray, the colored ground truth with the anomaly segmented by SAM.
            iou: float, the IoU between the anomaly mask and the SAM segmentation
        """
        # Check the anomaly generation quality by SAM
        success = False
        pred_ood_mask, iou = check_anomaly_by_SAM(generation, anomaly_mask, self.predictor)
        if iou <= 0.7:
            self.logger.info(f"low IoU ({iou}), regenerate...")
            return (success,)

        # Paste the anomaly mask on the ground truth
        H, W, C = label.shape
        pred_ood_mask = cv2.resize(pred_ood_mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
        label_SAM = label.copy()
        label_SAM[pred_ood_mask == 1, :] = [254, 254, 254]

        # Check the anomaly generation quality by OOD detector
        img_ood = cv2.resize(generation, (W, H))
        img_ood, pred_ood_mask = self.image_transform(Image.fromarray(img_ood), pred_ood_mask)
        ood_score = check_anomaly_by_detector(img_ood, pred_ood_mask, self.OODDetector)
        if ood_score < -3.02:
            self.logger.info(f"low ood score ({ood_score}), regenerate...")
            return (success,)
        self.logger.info(f"Success Generation with IoU: {iou} and OOD score: {ood_score}")
        success = True
        return success, label_SAM, iou


def main():
    generator = Semantic2ImageGenerator(cfg)

    for city_name in get_cities():
        for file in tqdm(os.listdir(os.path.join(generator.img_dir, city_name))):
            generator.logger.info("-" * 40)
            file_name = file.split("_leftImg8bit")[0]
            # Continue from the last processed image
            possible_name = os.path.join(generator.save_mask_dir, city_name, file_name + "*_gtFine_ADEcolor.png")
            if len(glob(possible_name)) > 0:
                generator.logger.info("EXIST! " + possible_name[0])
                continue
            prefix = os.path.join(generator.mask_dir, city_name, file_name)
            label_color = np.array(Image.open(prefix + '_gtFine_color.png'))[:, :, :3]
            label_trainId = np.array(Image.open(prefix + '_gtFine_labelTrainIds.png'))

            prompt_with_domain, domain = get_prompt(cfg.WEATHER_LIST, cfg.PLACE_PROMPT)
            generator.logger.info(f"Prompt: {prompt_with_domain}")

            ret = generator.generate(label_color, prompt_with_domain, cfg.a_prompt, cfg.n_prompt, seed)
            label, label_SAM, label_pasted, generation, novel_class, iou = ret

            domain += f"_{novel_class}_{iou:.2f}"
            label_trainId[np.any(label != label_SAM, axis=2)] = 254

            generation = cv2.resize(generation, (label_color.shape[1], label_color.shape[0]))

            prefix = os.path.join(generator.save_mask_dir, city_name, file_name + domain)

            label_file = prefix + '_gtFine_ADEcolor.png'
            label_paste_file = prefix + '_gtFine_ADEcolor_paste.png'
            trainId_file = prefix + '_gtFine_labelTrainIds.png'
            img_file = prefix.replace(generator.save_mask_dir, generator.save_img_dir) + '_leftImg8bit.png'
            generator.logger.info(img_file)

            os.makedirs(os.path.dirname(label_file), exist_ok=True)
            os.makedirs(os.path.dirname(trainId_file), exist_ok=True)
            os.makedirs(os.path.dirname(img_file), exist_ok=True)

            Image.fromarray(label_SAM).save(label_file)
            Image.fromarray(label_trainId).save(trainId_file)
            Image.fromarray(generation).save(img_file)
            if label_pasted is not None:
                Image.fromarray(label_pasted).save(label_paste_file)


if __name__ == "__main__":
    main()
