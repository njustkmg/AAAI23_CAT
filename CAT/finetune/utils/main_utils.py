import json
import logging
import os
import sys

from torchsummary import summary
import easydict
import hydra
import pytorch_lightning as pl
import torch
from dataset import get_collate_fn, get_dataset
from finetune.finetune_wrapper import FinetuningWrapper
from finetune.utils.hydra_utils import initialize_config
from model import get_contextual_relation_network, get_shot_encoder, get_audio_encoder


def init_hydra_config(mode: str):
    # set logging level
    logging.getLogger().setLevel(logging.DEBUG)

    overrides = sys.argv[1:]
    logging.info(f"####### overrides: {overrides}")
    with hydra.initialize_config_module(config_module="finetune.cfg"):
        cfg = hydra.compose("default", overrides=overrides)

    cfg = initialize_config(cfg, mode=mode)
    return cfg


def apply_random_seed(cfg):
    if "SEED" in cfg and cfg.SEED >= 0:
        pl.seed_everything(cfg.SEED, workers=True)


def load_pretrained_config(cfg):
    load_from = cfg.PRETRAINED_LOAD_FROM
    ckpt_root = cfg.PRETRAINED_CKPT_PATH

    with open(os.path.join(ckpt_root, load_from, "config.json"), "r") as fopen:
        pretrained_cfg = json.load(fopen)
        pretrained_cfg = easydict.EasyDict(pretrained_cfg)

    # override configuration of pre-trained model
    cfg.MODEL = pretrained_cfg.MODEL
    # set to use contextual relation network
    cfg.MODEL.contextual_relation_network.enabled = True

    # override neighbor size of an input sequence of shots
    sampling = pretrained_cfg.LOSS.sampling_method.name
    cfg.LOSS.sampling_method.params["sbd"][
        "neighbor_size"
    ] = pretrained_cfg.LOSS.sampling_method.params[sampling]["neighbor_size"]

    return cfg


def init_data_loader(cfg, mode, is_train):
    data_loader = torch.utils.data.DataLoader(
        dataset=get_dataset(cfg, mode=mode, is_train=is_train),
        batch_size=cfg.TRAIN.BATCH_SIZE.batch_size_per_proc,
        num_workers=cfg.TRAIN.NUM_WORKERS,
        pin_memory=cfg.TRAIN.PIN_MEMORY,
        persistent_workers=is_train,
        drop_last=is_train,
        shuffle=is_train,
        collate_fn=get_collate_fn(cfg),
    )
    if is_train:
        # need for warmup
        cfg.TRAIN.TRAIN_ITERS_PER_EPOCH = (
            len(data_loader.dataset) // cfg.TRAIN.BATCH_SIZE.effective_batch_size
        )
    return cfg, data_loader


def init_model(cfg):
    use_visual_mode = cfg.MODEL.visual.visual_mode
    use_audio_mode = cfg.MODEL.audio.audio_mode
    
    if use_visual_mode:
        visual_encoder = get_shot_encoder(cfg)
        visual_crn = get_contextual_relation_network(cfg, 'visual')
    else:
        visual_encoder = None
        visual_crn = None
    if use_audio_mode:
        audio_encoder = get_audio_encoder(cfg)
        audio_crn = get_contextual_relation_network(cfg, 'audio')
    else:
        audio_encoder = None
        audio_crn = None
    print("cfg.VISUAL_PRETRAINED_LOAD_FROM is ", cfg.VISUAL_PRETRAINED_LOAD_FROM)
    if "LOAD_FROM" in cfg and len(cfg.LOAD_FROM) > 0:
        # print("LOAD SBD MODEL WEIGHTS FROM: ", cfg.LOAD_FROM)
        model = FinetuningWrapper.load_from_checkpoint(
            cfg=cfg,
            visual_encoder=visual_encoder,
            visual_crn=visual_crn,
            audio_encoder=audio_encoder,
            audio_crn=audio_crn,
            checkpoint_path=os.path.join(cfg.CKPT_PATH, cfg.LOAD_FROM, "model-v1.ckpt"),
            strict=False,
        )
    elif "VISUAL_PRETRAINED_LOAD_FROM" in cfg and len(cfg.VISUAL_PRETRAINED_LOAD_FROM) > 0 and use_visual_mode:
        print("LOAD PRETRAINED MODEL WEIGHTS FROM: ", cfg.VISUAL_PRETRAINED_LOAD_FROM)
        model = FinetuningWrapper.load_from_checkpoint(
            cfg=cfg,
            visual_encoder=visual_encoder,
            visual_crn=visual_crn,
            audio_encoder=audio_encoder,
            audio_crn=audio_crn,
            checkpoint_path=os.path.join(
                cfg.PRETRAINED_CKPT_PATH, cfg.VISUAL_PRETRAINED_LOAD_FROM, "model-v1.ckpt"
            ),
            strict=False,
        )
        if "AUDIO_PRETRAINED_LOAD_FROM" in cfg and len(cfg.AUDIO_PRETRAINED_LOAD_FROM) > 0 and use_audio_mode:
            print("AUDIO_PRETRAINED_LOAD_FROM is : ", cfg.AUDIO_PRETRAINED_LOAD_FROM)
            audio_checkpoint = torch.load(os.path.join(
            cfg.PRETRAINED_CKPT_PATH, cfg.AUDIO_PRETRAINED_LOAD_FROM, "model-v1.ckpt"))
            for name, param in model.audio_encoder.named_parameters():
                param.data.copy_(audio_checkpoint['state_dict'][r"audio_encoder.{}".format(name)].data) 
            for name, param in model.audio_crn.named_parameters():
                param.data.copy_(audio_checkpoint['state_dict'][r"audio_crn.{}".format(name)].data)
    else:
        model = FinetuningWrapper(cfg, visual_encoder, visual_crn, audio_encoder, audio_crn)
    logging.info(f"MODEL: {model}")
    return cfg, model


def init_trainer(cfg):
    # logger
    logger = None
    callbacks = []
    if cfg.MODE == "finetune":
        logs_path = os.path.join(cfg.LOG_PATH, cfg.EXPR_NAME)
        os.makedirs(logs_path, exist_ok=True)
        logger = pl.loggers.TensorBoardLogger(logs_path, version=0)

        # checkpoint callback
        ckpt_path = os.path.join(cfg.CKPT_PATH, cfg.EXPR_NAME)
        os.makedirs(ckpt_path, exist_ok=True)
        callbacks.append(
            pl.callbacks.ModelCheckpoint(
                dirpath=ckpt_path, monitor=None, filename="model"
            )
        )
        # learning rate callback
        callbacks.append(pl.callbacks.LearningRateMonitor(logging_interval="step"))
        # GPU stat callback
        callbacks.append(
            pl.callbacks.GPUStatsMonitor(
                memory_utilization=True,
                gpu_utilization=True,
                intra_step_time=True,
                inter_step_time=True,
            )
        )
    trainer = pl.Trainer(**cfg.TRAINER, callbacks=callbacks, logger=logger)
    return cfg, trainer
