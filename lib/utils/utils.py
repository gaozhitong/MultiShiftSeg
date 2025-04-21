import random
import wget
import h5py
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn

def random_init(seed=0):
    cudnn.deterministic = True
    cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

class RunningMeter(object):
    def __init__(self):
        self.running_loss = []

    def update(self, loss):
        self.running_loss.append(loss.detach())

    def get_metric(self):
        avg = 0
        for p in self.running_loss:
            avg += p
        loss_avg = avg*1.0 / len(self.running_loss) if len(self.running_loss)!=0 else None
        return loss_avg

    def reset(self):
        self.running_loss = []

class MultiRunningMeter(object):
    def __init__(self):
        self.running_loss = {}

    def update(self, loss_dict):
        for key in loss_dict:
            if not key in self.running_loss:
                self.running_loss[key] = []
            self.running_loss[key].append(loss_dict[key])

    def get_metric(self):
        tmp = {}
        loss_avg = {}
        for key in self.running_loss:
            tmp[key] = 0.0
            for value in self.running_loss[key]:
                tmp[key] += value
            loss_avg[key] = tmp[key]*1.0 / len(self.running_loss[key]) if len(self.running_loss[key])!=0 else None
        return loss_avg

    def reset(self):
        self.running_loss = {}

def download_checkpoint(url, save_dir):
    print("Download PyTorch checkpoint")
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    filename = wget.download(url, out=str(save_dir))
    return filename


def save_as_hdf5(data, path, var_name='value', compression=9):
    with h5py.File(path, 'w') as hdf5_file_handle:
        if var_name in hdf5_file_handle:
            hdf5_file_handle[var_name][:] = data
        else:
            hdf5_file_handle.create_dataset(var_name, data=data, compression=compression)


def map2citycolor(array):
    from DSSeg_Release.lib.dataset.anomaly import Cityscapes

    """
    :param array: torch.Tensor with shape (B, H, W)
    :return: torch.Tensor with shape (B, H, W, 3)
    """
    # Create a tensor of zeros with the same shape as the input tensor, but with an additional dimension for RGB channels
    array_shape = list(array.shape)
    array_shape.append(3)
    array_rgb = torch.zeros(array_shape, dtype=array.dtype, device=array.device)

    # Loop through each unique training ID in the Cityscapes color palette
    for i, color in enumerate(Cityscapes.color_palette_train_ids):
        color_tensor = torch.tensor(color, dtype=array.dtype, device=array.device).view(1, 1, 1, 3)

        mask = (array == i).unsqueeze(-1)

        array_rgb += color_tensor * mask

    # unknown - > white
    color_tensor = torch.tensor((255,255,255), dtype=array.dtype, device=array.device).view(1, 1, 1, 3)
    mask = (array == 254).unsqueeze(-1)
    array_rgb += color_tensor * mask

    return array_rgb
