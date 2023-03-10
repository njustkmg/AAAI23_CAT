import logging

import mindspore
import mindspore.ops as ops

from misc.utils import *
from pretrain.loss.pretext_task import PretextTaskWrapper, SimclrLoss



class CATLoss(PretextTaskWrapper):
    def __init__(self, cfg):
        PretextTaskWrapper.__init__(self, cfg=cfg)
        # to disable debug dump in numba (used by DTW computation)
        numba_logger = logging.getLogger("numba")
        numba_logger.setLevel(logging.WARNING)
        self.use_gsm_loss = cfg.LOSS.global_scene_matching.get("enabled", True)
        self.it = 0

    def forward(self, shot_repr, **kwargs):
        self.it += 1
        b, s, d = shot_repr.shape
        # bnet_shot_repr = self.head_nce(shot_repr)

        loss = {}
        if self.use_crn:
            masking_mask = kwargs.get("mask", None)
            crn = kwargs.get("crn", None)
            shuffled_orders = kwargs.get("shuffled_orders", None)
            orders_targets = kwargs.get("orders_targets", None)
            assert crn is not None
        head_shot_repr = self.head_nce(shot_repr)

        # Get pseudo-scene bounds
        with ms.no_grad():
            scence_idx = ms.tensor([0, s - 1] * b, dtype=ms.long, device=shot_repr.device).view(-1, 2)
            center_idx = s // 2
            normalized_d_emb = normalize(head_shot_repr, dim=-1)
            sim = ms.einsum(
                "bd,btd->bt", normalized_d_emb[:, center_idx], normalized_d_emb
            )

            for i in range(b):
                # start shot
                for j in range(center_idx-1, -1, -1):
                    if sim[i][j] < self.sim_threshold:
                        scence_idx[i][0] = j
                        break
                # end shot
                for j in range(center_idx+1, s, 1):
                    if sim[i][j] < self.sim_threshold:
                        scence_idx[i][1] = j
                        break
        # compute masked shot modeling loss
        if self.use_msm_loss:
            masked_shot_loss = self._compute_msm_loss(
                crn, shot_repr, masking_mask
            )
            loss["msm_loss"] = masked_shot_loss

        if self.use_som_loss:
            shot_order_loss = self._compute_som_loss(
                crn, shot_repr, shuffled_orders, orders_targets
            )
            loss["som_loss"] = shot_order_loss

        # compute shot-scene matching Loss
        if self.use_gsm_loss:
            gsm_loss = self._compute_gsm_loss(head_shot_repr, scence_idx)
            loss["gsm_loss"] = gsm_loss

        # compute contextual group matching loss
        if self.use_lsm_loss:
            crn_repr_wo_mask, center_repr = crn(shot_repr)  # infer CRN without masking
            crn_repr_wo_mask = crn_repr_wo_mask[
                :, 1:
            ].contiguous()  # exclude [CLS] token
            lsm_loss = self._compute_lsm_loss(crn_repr_wo_mask, scence_idx)
            loss["cgm_loss"] = lsm_loss
        return loss
