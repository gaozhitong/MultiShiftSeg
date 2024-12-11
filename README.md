<h1 align="center">Generalize or Detect? Towards Robust Semantic Segmentation Under Multiple Distribution Shifts</h1>

---

<p align="center">
    <a href="https://gaozhitong.github.io"><strong><ins>Zhitong Gao</ins></strong></a>
    ·
    <a href="https://www.bingnanli.com"><strong><ins>Bingnan Li</ins></strong></a>
    ·
    <a href="https://people.epfl.ch/mathieu.salzmann"><strong><ins>Mathieu Salzmann</ins></strong></a>
    ·
    <a href="https://xmhe.bitbucket.io"><strong><ins>Xuming He</ins></strong></a>
</p>


<p align="center"> 
    [<a href="https://arxiv.org/abs/2411.03829#:~:text=Towards%20Robust%20Semantic%20Segmentation%20Under%20Multiple%20Distribution%20Shifts,-Zhitong%20Gao%2C%20Bingnan&text=In%20open%2Dworld%20scenarios%2C%20where,and%20generalize%20to%20new%20domains.">Arxiv</a>]
    [<a href="">Poster</a>]
    [<a href="">Slides</a>]
    [<a href="https://recorder-v3.slideslive.com/?share=95164&s=dd2ba512-5f4c-47ca-8744-4a6a44ad7479">Video</a>]
</p> 

<h4 align="center">NeurIPS 2024 Proceedings</h3>

![pipline.png](imgs/pipline.png "pipeline")
Figure 1: The overview of the proposed method.

---

### Abstract

> In open-world scenarios, where both novel classes and domains may exist, an ideal segmentation model should detect
> anomaly classes for safety and generalize to new domains. However, existing methods often struggle to distinguish
> between domain-level and semantic-level distribution shifts, leading to poor OOD detection or domain generalization
> performance. In this work, we aim to equip the model to generalize effectively to covariate-shift regions while
> precisely identifying semantic-shift regions. To achieve this, we design a novel generative augmentation method to
> produce coherent images that incorporate both anomaly (or novel) objects and various covariate shifts at both image
> and
> object levels. Furthermore, we introduce a training strategy that recalibrates uncertainty specifically for semantic
> shifts and enhances the feature extractor to align features associated with domain shifts. We validate the
> effectiveness
> of our method across benchmarks featuring both semantic and domain shifts. Our method achieves state-of-the-art
> performance across all benchmarks for both OOD detection and domain generalization.

---

### Environment Setup

```bash
conda env create -f environment.yml 

conda activate MultiShiftSeg
git clone https://github.com/facebookresearch/detectron2.git
pip install -e detectron2
pip install git+https://github.com/cocodataset/panopticapi.git
cd lib/network/mask2former/modeling/pixel_decoder/ops
sh make.sh
```

### Data Preparation

```
datasets
├── cityscapes                      
├── road_anomaly
│   ├── original
│   └── labels
├── dataset_AnomalyTrack            #RA21
│   ├── images
│   └── labels_masks
├── dataset_ObstacleTrack           #RO21
│   ├── images
│   ├── image-sources.txt
│   └── labels_masks
├── MUAD_challenge                  
│   └── test_sets
│       └── test_OOD
│           ├── leftImg8bit
│           └── leftLabel
├── acdc_ood                        #ACDC-POC
│   ├── gt_trainval
│   └── rgb_anon_trainvaltest
├── fs_LostAndFound                 
│   ├── original
│   └── labels 
└── fs_static
    ├── original
    ├── labels 
    └── match.npy
```

Generated data can be downloaded
from [Google Drive](https://drive.google.com/file/d/1PxjH5q-R6kBdVaaC0ssBXwl8Z7JbBWIk/view?usp=share_link)
or [Hugging Face](https://huggingface.co/datasets/Cuttle-fish-my/DTWP_ADE/tree/main).
To generate the data, please refer to the [Generation Instruction](ControlNet/README.md).

For more detailed instructions, please refer to the [Dataset Instruction](datasets/README.md).
### Checkpoint

|             | RoadAnomaly | RoadAnomaly | RoadAnomaly | SMIYC-RA21 | SMIYC-RA21 | SMIYC-RO21 | SMIYC-RO21 |                                                                                                 Weights                                                                                                 |
|:-----------:|:-----------:|:-----------:|:-----------:|:----------:|:----------:|:----------:|:----------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|   Method    |     AUC     |     AP      |     FPR     |     AP     |    FPR     |     AP     |    FPR     |                                                                                                                                                                                                         |
| DeepLab v3+ |    96.40    |    74.60    |    16.08    |   88.06    |    8.21    |   90.71    |    0.26    | [Google Drive](https://drive.google.com/file/d/1EB73bf3w0HJQdNcpFp_vOgWpOctYz7Tr/view?usp=share_link) or [Hugging Face](https://huggingface.co/Cuttle-fish-my/MultiShiftSeg/blob/main/DeepLab_best.pth) |
| Mask2Former |    97.94    |    90.17    |    7.54     |   91.92    |    7.94    |   95.29    |    0.07    |   [Google Drive](https://drive.google.com/file/d/1wH0skkEk6DXMVawegwcFLHhc1mA0Z3p1/view?usp=share_link) or [Hugging Face](https://huggingface.co/Cuttle-fish-my/MultiShiftSeg/blob/main/M2F_best.pth)   |

### Training

Coming Soon.

### Evaluation

Coming Soon.

### BibTeX

```bibtex
@inproceedings{
gao2024generalize,
title={Generalize or Detect? Towards Robust Semantic Segmentation Under Multiple Distribution Shifts},
author={Zhitong Gao and Bingnan Li and Mathieu Salzmann and Xuming He},
booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
year={2024},
url={https://openreview.net/forum?id=h0rbjHyWoa}
}
```