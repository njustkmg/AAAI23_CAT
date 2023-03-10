import torch as ms
from torch.utils.data import Dataset
from torchvision.datasets.folder import pil_loader as p_loader
import pytorch_lightning as pl
from pytorch_lightning import LightningModule as LModule
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import softmax, cross_entropy, normalize, mse_loss
from torch.nn import Module as Cell
from torch.nn import Conv2d, BatchNorm2d, ReLU, MaxPool2d, AdaptiveAvgPool2d, init, GroupNorm, Sequential, Embedding
from pl_bolts.optimizers.lr_scheduler import linear_warmup_decay
# import torch
import torchmetrics as metrics
from torchsummary import summary
from torch.nn import Linear as Dense
import torch.distributed as dist
from classy_vision.generic.distributed_util import (
    convert_to_distributed_tensor,
    convert_to_normal_tensor,
    is_distributed_training_run,
)
from typing import Any, Callable, List, Optional, Type, Union

from torch import Tensor
from torch.hub import load_state_dict_from_url
import torch.nn.functional as F
from torch.optim import SGD
from pl_bolts.optimizers.lars import LARS
import torchvision as vision
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import adjust_brightness, adjust_contrast, adjust_saturation, resized_crop, resize, to_tensor
from classy_vision.dataset.transforms import register_transform
from classy_vision.dataset.transforms.classy_transform import ClassyTransform
from torchvision.transforms import RandomResizedCrop
import einops

model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-f37072fd.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-b627a593.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-0676ba61.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-63fe2227.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-394f9c45.pth",
    "resnext50_32x4d": "https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth",
    "resnext101_32x8d": "https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth",
    "wide_resnet50_2": "https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth",
    "wide_resnet101_2": "https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth",
}