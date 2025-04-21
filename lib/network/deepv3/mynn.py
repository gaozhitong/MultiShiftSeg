"""
Custom Norm wrappers to enable sync BN, regular BN and for weight initialization
"""
from lib.configs.parse_arg import opt, args
import torch
import torch.nn as nn

def Norm2d(in_channels):
    """
    Custom Norm Function to allow flexible switching
    """
    return nn.BatchNorm2d(in_channels)

def initialize_weights(*models):
    """
    Initialize Model Weights
    """
    for model in models:
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

def Upsample(x, size):
    """
    Wrapper Around the Upsample Call
    """
    return nn.functional.interpolate(x, size=size, mode='bilinear',
                                     align_corners=True)


# class Norm2d(nn.BatchNorm2d):
#
#     def __init__(self, num_features, eps=1e-5, momentum=0.1,
#                  affine=True, track_running_stats=True, adapt = False):
#         super(Norm2d, self).__init__(
#             num_features, eps, momentum, affine, track_running_stats)
#         self.adapt = adapt
#         self.momentum = momentum
#         self.mean, self.var = None, None
#
#     def forward(self, input):
#
#         self._check_input_dim(input)
#
#         self.mean = input.mean([0, 2, 3]).detach()
#         self.var = input.var([0, 2, 3], unbiased=False).detach()
#
#         exponential_average_factor = 0.0
#         if self.training and self.track_running_stats:
#             if self.num_batches_tracked is not None:
#                 self.num_batches_tracked += 1
#                 if self.momentum is None:  # use cumulative moving average
#                     exponential_average_factor = 1.0 / float(self.num_batches_tracked)
#                 else:  # use exponential moving average
#                     exponential_average_factor = self.momentum
#
#         # calculate running estimates
#         if self.training:
#             n = input.numel() / input.size(1)
#             with torch.no_grad():
#                 self.running_mean = exponential_average_factor * self.mean \
#                                     + (1 - exponential_average_factor) * self.running_mean
#                 # update running_var with unbiased var
#                 self.running_var = exponential_average_factor * self.var * n / (n - 1) \
#                                    + (1 - exponential_average_factor) * self.running_var
#             mean = self.mean
#             var = self.var
#
#         elif self.adapt:
#             exponential_average_factor = self.momentum
#             n = input.numel() / input.size(1)
#             with torch.no_grad():
#                 mix_mean = exponential_average_factor * self.mean \
#                                     + (1 - exponential_average_factor) * self.running_mean.detach()
#                 # update running_var with unbiased var
#                 if n != 1:
#                     mix_var = exponential_average_factor * self.var * n / (n - 1) \
#                                    + (1 - exponential_average_factor) * self.running_var.detach()
#                 else:
#                     mix_var = exponential_average_factor * self.var  \
#                               + (1 - exponential_average_factor) * self.running_var.detach()
#             mean = mix_mean
#             var = mix_var
#
#         else: # test
#             mean = self.running_mean
#             var = self.running_var
#
#         # if mean is None:
#         #     mean = self.mean
#         # if var is None:
#         #     var = self.var
#
#         input = (input - mean[None, :, None, None]) / (torch.sqrt(var[None, :, None, None] + self.eps))
#         if self.affine:
#             input = input * self.weight[None, :, None, None] + self.bias[None, :, None, None]
#         return input