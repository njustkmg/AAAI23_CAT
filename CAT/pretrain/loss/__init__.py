# ------------------------------------------------------------------------------------
# BaSSL
# Copyright (c) 2021 KakaoBrain. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------

from .pretrain_loss import (
    CATLoss,
)


def get_loss(cfg):
    if cfg.LOSS.sampling_method.name == "bassl":
        loss = CATLoss(cfg)
    else:
        raise NotImplementedError
    return loss


__all__ = ["get_loss"]
