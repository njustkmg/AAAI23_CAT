import json
import logging
import os

import einops
import mindspore
from misc.utils import *
from finetune.utils.hydra_utils import save_config_to_disk
from finetune.utils.metric import (
    AccuracyMetric,
    F1ScoreMetric,
    MovieNetMetric,
    SklearnAPMetric,
    SklearnAUCROCMetric,
)


class FinetuningWrapper(LModule):
    def __init__(self, cfg, visual_encoder, visual_crn, audio_encoder, audio_crn):
        super().__init__()
        self.cfg = cfg
        self.use_visual_mode = self.cfg.MODEL.visual.visual_mode
        self.use_audio_mode = self.cfg.MODEL.audio.audio_mode
        
        # build model components
        self.visual_encoder = visual_encoder
        self.visual_crn = visual_crn
        self.audio_encoder = audio_encoder
        self.audio_crn = audio_crn
        
        crn_name = cfg.MODEL.audio.contextual_relation_network.name
        hdim = cfg.MODEL.audio.contextual_relation_network.params[crn_name]["hidden_size"]

        self.visual_head_sbd = Dense(hdim, 2)
        self.audio_head_sbd = Dense(hdim, 2)

        # define loss
        self.criterion = nn.CrossEntropyLoss(weight=ms.tensor([1.0, 4.0], device=self.device))
        # define metrics
        self.acc_metric = AccuracyMetric()
        self.ap_metric = SklearnAPMetric()
        self.f1_metric = F1ScoreMetric(num_classes=1)
        self.auc_metric = SklearnAUCROCMetric()
        self.movienet_metric = MovieNetMetric()

        self.log_dir = os.path.join(cfg.LOG_PATH, cfg.EXPR_NAME)
        self.use_raw_shot = cfg.USE_RAW_SHOT
        self.eps = 1e-5

    def on_train_start(self) -> None:
        if self.global_rank == 0:
            try:
                save_config_to_disk(self.cfg)
            except Exception as err:
                logging.info(err)

    def extract_shot_representation(self, inputs: ms.Tensor) -> ms.Tensor:
        """ inputs [b s k c h w] -> output [b d] """
        assert len(inputs.shape) == 6  # (B Shot Keyframe C H W)
        b, s, k, c, h, w = inputs.shape

        # we extract feature of each key-frame and average them
        inputs = einops.rearrange(inputs, "b s k c h w -> (b s) k c h w", s=s)
        keyframe_repr = [self.visual_encoder(inputs[:, _k]) for _k in range(k)]
        shot_repr = ms.stack(keyframe_repr).mean(dim=0)  # [k (b s) d] -> [(b s) d]
        shot_repr = einops.rearrange(shot_repr, "(b s) d -> b s d", s=s)
        return shot_repr
    
    def extract_audio_representation(self, inputs, is_train=True):
        """
            inputs [b s k h w] -> output [b d]
        """
        b, s, h, w = inputs.shape
        inputs = einops.rearrange(inputs, "b s h w -> (b s) 1 h w", s=s)
        x = self.audio_encoder(inputs)
        # reshape output to [b s d]
        x = einops.rearrange(x, "(b s) d -> b s d", s=s, b=b)
        return x

    def shared_step(self, visual_inputs: ms.Tensor, audio_inputs: ms.Tensor) -> ms.Tensor:
        with ms.no_grad():
            # infer shot encoder
            if self.use_raw_shot:
                visual_shot_repr = self.extract_shot_representation(visual_inputs)
                audio_shot_repr = self.extract_audio_representation(audio_inputs)
            else:
                visual_shot_repr = visual_inputs
                audio_shot_repr = audio_inputs
            assert len(visual_shot_repr.shape) == 3 and len(audio_shot_repr.shape) == 3

        # infer CRN
        if self.use_visual_mode:
            visual_crn_repr_wo_mask, visual_crn_repr_wo_mask_pool = self.visual_crn(visual_shot_repr, mask=None)
            visual_pred = self.visual_head_sbd(visual_crn_repr_wo_mask_pool)
        if self.use_audio_mode:
            audio_crn_repr_wo_mask, audio_crn_repr_wo_mask_pool = self.audio_crn(audio_shot_repr, mask=None)
            audio_pred = self.audio_head_sbd(audio_crn_repr_wo_mask_pool)
        if self.use_visual_mode and self.use_audio_mode:
            pred = visual_pred*0.7 + audio_pred*0.3
        elif self.use_visual_mode:
            pred = visual_pred
        elif self.use_audio_mode:
            pred = audio_pred
        return pred

    def forward(self, x: ms.Tensor, **kwargs) -> ms.Tensor:
        return self.shared_step(x, **kwargs)

    def training_step(self, batch: ms.Tensor, batch_idx: int) -> ms.Tensor:
        visual_inputs = batch["video_visual"]
        audio_inputs = batch["video_audio"]
        labels = batch["label"].view(-1)
        outputs = self.shared_step(visual_inputs, audio_inputs)
        # compute sbd loss where positive and negative ones are
        # balanced with their numbers
        loss = self.criterion(outputs, labels)
        # write metrics
        preds = ms.argmax(outputs, dim=1)

        gt_one = labels == 1
        gt_zero = labels == 0
        pred_one = preds == 1
        pred_zero = preds == 0

        tp = (gt_one * pred_one).sum()
        fp = (gt_zero * pred_one).sum()
        tn = (gt_zero * pred_zero).sum()
        fn = (gt_one * pred_zero).sum()

        acc0 = 100.0 * tn / (fp + tn + self.eps)
        acc1 = 100.0 * tp / (tp + fn + self.eps)
        tp_tn = tp + tn

        self.log(
            "sbd_train/loss",
            loss,
            on_step=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "sbd_train/tp_batch",
            tp,
            on_step=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "sbd_train/fp_batch",
            fp,
            on_step=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "sbd_train/tn_batch",
            tn,
            on_step=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "sbd_train/fn_batch",
            fn,
            on_step=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "sbd_train/acc0",
            acc0,
            on_step=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "sbd_train/acc1",
            acc1,
            on_step=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "sbd_train/tp_tn",
            tp_tn,
            on_step=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        return loss

    def validation_step(self, batch: ms.Tensor, batch_idx: int):
        vids = batch["vid"]
        sids = batch["sid"]
        visual_inputs = batch["video_visual"]
        audio_inputs = batch["video_audio"]
        labels = batch["label"]
        outputs = self.shared_step(visual_inputs, audio_inputs)

        prob = softmax(outputs, dim=1)
        preds = ms.argmax(prob, dim=1)
        self.acc_metric.update(
            prob[:, 1], labels
        )  # prob[:,1] is confidence score for boundary
        self.ap_metric.update(prob[:, 1], labels)
        self.f1_metric.update(prob[:, 1], labels)
        self.auc_metric.update(prob[:, 1], labels)
        for vid, sid, pred, gt in zip(vids, sids, preds, labels):
            self.movienet_metric.update(vid, sid, pred, gt)

    def validation_epoch_end(self, validation_step_outputs):
        score = {}

        # update acc.
        acc = self.acc_metric.compute()
        ms.cuda.synchronize()
        assert isinstance(acc, dict)
        score.update(acc)
        # update average precision (AP).
        ap, _, _ = self.ap_metric.compute()  # * 100.
        ap *= 100.0
        ms.cuda.synchronize()
        assert isinstance(ap, ms.Tensor)
        score.update({"ap": ap})
        # update AUC-ROC
        auc, _, _ = self.auc_metric.compute()
        auc *= 100.0
        ms.cuda.synchronize()
        assert isinstance(auc, ms.Tensor)
        score.update({"auc": auc})
        # update F1 score.
        f1 = self.f1_metric.compute() * 100.0
        ms.cuda.synchronize()
        assert isinstance(f1, ms.Tensor)
        score.update({"f1": f1})
        # update recall, mIoU score.
        recall, recall_at_3s, miou = self.movienet_metric.compute()
        ms.cuda.synchronize()
        assert isinstance(recall, ms.Tensor)
        assert isinstance(recall_at_3s, ms.Tensor)
        assert isinstance(miou, ms.Tensor)
        score.update({"recall": recall * 100.0})
        score.update({"recall@3s": recall_at_3s * 100})
        score.update({"mIoU": miou * 100})

        # logging
        for k, v in score.items():
            self.log(
                f"sbd_test/{k}",
                v,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )
        score = {k: v.item() for k, v in score.items()}
        self.print(f"\nTest Score: {score}")

        # reset all metrics
        self.acc_metric.reset()
        self.ap_metric.reset()
        self.f1_metric.reset()
        self.auc_metric.reset()
        self.movienet_metric.reset()

        # save last epoch result.
        with open(os.path.join(self.log_dir, "all_score.json"), "w") as fopen:
            json.dump(score, fopen, indent=4, ensure_ascii=False)

    def test_step(self, batch: ms.Tensor, batch_idx: int):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, test_step_outputs):
        return self.validation_epoch_end(test_step_outputs)

    def exclude_from_wt_decay(self, named_params, weight_decay, skip_list):
        params = []
        excluded_params = []

        for name, param in named_params:
            # if not param.requires_grad or "head_sbd" not in name:
            if not param.requires_grad:
                continue
            elif any(layer_name in name for layer_name in skip_list):
                excluded_params.append(param)
                # print(name)
            else:
                params.append(param)
                # print(name)
        # print("params is ", param)
        # print("excluded_params is ", excluded_params)
        return [
            {"params": params, "weight_decay": weight_decay},
            {"params": excluded_params, "weight_decay": 0.0},
        ]

    def configure_optimizers(self):
        # params
        skip_list = []
        weight_decay = self.cfg.TRAIN.OPTIMIZER.weight_decay
        if not self.cfg.TRAIN.OPTIMIZER.regularize_bn:
            skip_list.append("bn")
        if not self.cfg.TRAIN.OPTIMIZER.regularize_bias:
            skip_list.append("bias")
        params = self.exclude_from_wt_decay(
            self.named_parameters(), weight_decay=weight_decay, skip_list=skip_list
        )
        # optimizer
        if self.cfg.TRAIN.OPTIMIZER.name == "sgd":
            optimizer = ms.optim.SGD(
                params,
                lr=self.cfg.TRAIN.OPTIMIZER.lr.scaled_lr,
                momentum=0.9,
                weight_decay=weight_decay,
            )
        elif self.cfg.TRAIN.OPTIMIZER.name == "adam":
            optimizer = ms.optim.Adam(
                params, lr=self.cfg.TRAIN.OPTIMIZER.lr.scaled_lr
            )
        else:
            raise ValueError()

        warmup_steps = int(
            self.cfg.TRAIN.TRAIN_ITERS_PER_EPOCH
            * self.cfg.TRAINER.max_epochs
            * self.cfg.TRAIN.OPTIMIZER.scheduler.warmup
        )
        total_steps = int(
            self.cfg.TRAIN.TRAIN_ITERS_PER_EPOCH * self.cfg.TRAINER.max_epochs
        )

        if self.cfg.TRAIN.OPTIMIZER.scheduler.name == "cosine_with_linear_warmup":
            scheduler = {
                "scheduler": ms.optim.lr_scheduler.LambdaLR(
                    optimizer,
                    linear_warmup_decay(warmup_steps, total_steps, cosine=True),
                ),
                "interval": "step",
                "frequency": 1,
            }
        else:
            raise NotImplementedError

        return [optimizer], [scheduler]
