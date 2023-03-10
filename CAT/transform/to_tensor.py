import mindspore
import mindspore.dataset.vision
import mindspore.dataset.transforms as transforms
from typing import Any, Dict

import numpy as np
from misc.utils import *
from PIL import Image

try:
    import accimage
except ImportError:
    accimage = None


# See /pytorch/vision/torchvision/transforms/functional.py
def _is_numpy(img):
    return isinstance(img, np.ndarray)


def _is_pil_image(img):
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)


@register_transform("VideoToTensor")
class VideoToTensor(ClassyTransform):
    def __init__(self, mean=None, std=None, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

        assert self.mean is not None
        assert self.std is not None

    def __to_tensor__(self, img):
        return to_tensor(img)

    def __normalize__(self, img):
        return TF.normalize(img, self.mean, self.std, self.inplace)

    def __call__(self, imgmap):
        assert isinstance(imgmap, list)
        return [self.__normalize__(self.__to_tensor__(img)) for img in imgmap]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "VideoToTensor":
        mean = config.get("mean", [0.485, 0.456, 0.406])
        std = config.get("std", [0.229, 0.224, 0.225])
        return cls(mean=mean, std=std)
