import random
import mindspore
import mindspore.dataset.vision
import mindspore.dataset.transforms as transforms

from typing import Any, Dict
from misc.utils import *


@register_transform("VideoRandomGrayScale")
class VideoRandomGrayScale(ClassyTransform):
    def __init__(self, p=0.2):
        self.p = p
        self.tfm = vision.transforms.Grayscale(num_output_channels=3)

    def __call__(self, imgmap):
        assert isinstance(imgmap, list)
        if random.random() < self.p:
            return [self.tfm(img) for img in imgmap]
        else:
            return imgmap

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "VideoRandomGrayScale":
        p = config.get("p", 0.2)
        return cls(p=p)
