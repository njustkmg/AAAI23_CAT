import torch

from .movienet import MovieNetDataset


def get_dataset(cfg, mode, is_train):
    dataset = None
    if cfg.DATASET == "movienet":
        dataset = MovieNetDataset(cfg=cfg, mode=mode, is_train=is_train)
    else:
        raise NotImplementedError("not supported dataset: {}".format(cfg.DATASET))
    assert dataset is not None
    return dataset


def get_collate_fn(cfg):
    default_colalte_fn = torch.utils.data._utils.collate.default_collate
    return default_colalte_fn


__all__ = ["get_dataset", "get_collate_fn"]
