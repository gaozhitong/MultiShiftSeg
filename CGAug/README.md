# Coherent Generative-based Augmentation

This repository is based on [ControlNet](https://github.com/lllyasviel/ControlNet). We adopt it to generate domain and
semantic shift at the same time.


## Useage

### 0. Environment Setup

Please refer to the [Instructions](../README.md#1-environment-setup) to set up the environment.

### 1. Download the pretrained weights

```
conda activate MultiShiftSeg
python download.py
```
### 2. Configure the training parameters

Please check the [``config.py``](config.py) file to set the parameters for generation.

### 2. Generate the augmented images

To generate the augmented images, you may run the following command in the **outer** directory:

```bash
# For Chinese users, you may use the following mirror:
# export HF_ENDPOINT=https://hf-mirror.com
python CGAug/generate_mutishift_image.py --cfg CGAug/static_data/anomaly_detector_config.yaml
```
