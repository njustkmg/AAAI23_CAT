import logging
import einops
import numpy as np
import mindspore
import mindspore.nn as nn


from misc.utils import *
from misc.dist_utils import gather_from_all
from model.crn.trn import BertMLMHead
from model.head import MlpHead


class SimclrLoss(Cell):
    def __init__(self, cfg):
        Cell.__init__(self)

        self.cfg = cfg
        self.num_pos = 2  # fixed
        if self.cfg.MODEL.visual.visual_mode:
            pretrain_mode = 'visual'
        elif self.cfg.MODEL.audio.audio_mode:
            pretrain_mode = 'audio'
        else:
            raise NotImplementedError

        gsm_name = cfg.LOSS.global_scene_matching.name
        nce_cfg = cfg.LOSS.global_scene_matching.params[pretrain_mode][gsm_name]
        self.T = nce_cfg["temperature"]

        # create head for nce loss
        self.head_nce = MlpHead(**nce_cfg["head"])
        self.scene_padding = Embedding(1, nce_cfg["head"]["output_dim"])
        # parameters for mask generation
        self.total_instances = (
            self.cfg.TRAIN.BATCH_SIZE.effective_batch_size * self.num_pos
        )
        self.world_size = self.cfg.DISTRIBUTED.WORLD_SIZE
        self.batch_size = self.total_instances // self.world_size
        self.orig_instances = self.batch_size // self.num_pos

    def on_train_start(self, dist_rank, device):
        self.dist_rank = dist_rank
        self.device = device
        logging.info(f"Creating Info-NCE loss on Rank: {self.dist_rank}")
        self.precompute_pos_neg_mask()

    def precompute_pos_neg_mask(self):
        """ we precompute the positive and negative masks to speed up the loss calculation
        """
        # computed once at the begining of training
        pos_mask = ms.zeros(
            self.batch_size, self.total_instances, device=self.device
        )
        neg_mask = ms.zeros(
            self.batch_size, self.total_instances, device=self.device
        )
        all_indices = np.arange(self.total_instances)
        pos_members = self.orig_instances * np.arange(self.num_pos)
        orig_members = ms.arange(self.orig_instances)
        for anchor in np.arange(self.num_pos):
            for img_idx in range(self.orig_instances):
                delete_inds = self.batch_size * self.dist_rank + img_idx + pos_members
                neg_inds = ms.tensor(np.delete(all_indices, delete_inds)).long()
                neg_mask[anchor * self.orig_instances + img_idx, neg_inds] = 1
            for pos in np.delete(np.arange(self.num_pos), anchor):
                pos_inds = (
                    self.batch_size * self.dist_rank
                    + pos * self.orig_instances
                    + orig_members
                )
                pos_mask[
                    ms.arange(
                        anchor * self.orig_instances, (anchor + 1) * self.orig_instances
                    ).long(),
                    pos_inds.long(),
                ] = 1
        self.pos_mask = pos_mask
        self.neg_mask = neg_mask

    def _compute_gsm_loss(self, head_shot_repr, scence):
        b, s, d = head_shot_repr.shape
        # compute scene-level embeddings (average of dense shot features)
        scene_emb = []
        for bi in range(b):
            center_start_idx = scence[bi][0]
            center_end_idx = scence[bi][1]+1
            # center scene embedding
            center_scence_idx = ms.zeros(s, dtype=ms.bool, device=head_shot_repr.device).detach()
            center_scence_idx[center_start_idx: center_end_idx] = 1
            # aligned_dense_idx = dtw_path[bi][:, 1][aligned_dense_mask]
            center_scene_emb = head_shot_repr[bi, center_scence_idx].mean(dim=0)
            scene_emb.append(center_scene_emb)

            # left scene embedding
            if center_start_idx == 0:
                scene_emb.append(center_scene_emb)
            else:
                left_scence_idx = ms.zeros(s, dtype=ms.bool, device=head_shot_repr.device).detach()
                left_scence_idx[0: center_start_idx] = 1
                left_scene_emb = head_shot_repr[bi, left_scence_idx].mean(dim=0)
                scene_emb.append(left_scene_emb)

            # right scene embedding
            if center_end_idx == s:
                scene_emb.append(center_scene_emb)
            else:
                right_scence_idx = ms.zeros(s, dtype=ms.bool, device=head_shot_repr.device).detach()
                right_scence_idx[center_end_idx: s] = 1
                right_scene_emb = head_shot_repr[bi, right_scence_idx].mean(dim=0)
                scene_emb.append(right_scene_emb)

        scene_emb = ms.stack(scene_emb, dim=0)  # [b*3, d]
        scene_emb = normalize(scene_emb, dim=-1)
        scene_emb = einops.rearrange(scene_emb, "(b nscene) d -> b nscene d", b=b)
        # compute contrastive loss for individual aligned pairs
        gsm_loss = 0
        center_idx = s//2
        for i, si in enumerate([center_idx, 0, s-1]):
            sparse_shot = head_shot_repr[:, si]
            scene_shot = scene_emb[:, i]
            paired_emb = ms.cat([sparse_shot, scene_shot], dim=0)  # [b*2 d]
            gsm_loss += self._compute_nce_loss(paired_emb)

        return gsm_loss

    def _compute_nce_loss(self, embedding):
        # Step 1: gather all the embeddings. Shape example: 4096 x 128
        if ms.distributed.is_available() and ms.distributed.is_initialized():
            embeddings_buffer = gather_from_all(embedding)
        else:
            embeddings_buffer = embedding

        # Step 2: matrix multiply: 64 x 128 with 4096 x 128 = 64 x 4096
        # and divide by temperature.
        similarity = ms.exp(ms.mm(embedding, embeddings_buffer.t()) / self.T)
        pos = ms.sum(similarity * self.pos_mask, 1)
        neg = ms.sum(similarity * self.neg_mask, 1)
        loss = -(ms.mean(ms.log(pos / (pos + neg))))
        return loss

    def forward(self, shot_repr, **kwargs):
        # shot_repr shape: [b nview d] -> [(nview b) d]
        shot_repr = ms.cat(ms.unbind(shot_repr, dim=1), dim=0)
        shot_repr = self.head_nce_for_crn(shot_repr)  # [(nview b) d_head]
        return {"simclr_loss": self._compute_nce_loss(shot_repr)}


class PretextTaskWrapper(SimclrLoss):
    def __init__(self, cfg):
        SimclrLoss.__init__(self, cfg=cfg)

        self.sim_threshold = cfg.LOSS.sim_threshold
        self.use_crn = cfg.MODEL.audio.contextual_relation_network.enabled
        self.use_msm_loss = cfg.LOSS.masked_shot_modeling.get("enabled", False)
        self.use_lsm_loss = cfg.LOSS.local_scene_matching.get("enabled", False)
        self.use_som_loss = cfg.LOSS.shot_order_modeling.get("enabled", False)
        

        if self.use_crn:
            # if we use CRN, one of following losses should be used (set to True)
            # assert self.use_msm_loss or self.use_pp_loss or self.use_cgm_loss
            if cfg.MODEL.visual.visual_mode:
                crn_name = cfg.MODEL.visual.contextual_relation_network.name
            else:
                crn_name = cfg.MODEL.audio.contextual_relation_network.name
        else:
            # if we do not use TM, all following losses should not be used (set to False)
            assert (
                (not self.use_msm_loss)
                and (not self.use_som_loss)
                and (not self.use_lsm_loss)
            )

        # masked shot modeling loss
        if self.use_msm_loss:
            if cfg.MODEL.visual.visual_mode:
                msm_params = cfg.MODEL.visual.contextual_relation_network.params[crn_name]
            else:
                msm_params = cfg.MODEL.audio.contextual_relation_network.params[crn_name]
            msm_params["vocab_size"] = msm_params.input_dim
            self.head_msm = BertMLMHead(msm_params)


        if self.use_som_loss:
            if cfg.MODEL.visual.visual_mode:
                som_params = cfg.MODEL.visual.contextual_relation_network.params[crn_name]
            else:
                som_params = cfg.MODEL.audio.contextual_relation_network.params[crn_name]
            sampling_name = cfg.LOSS.sampling_method.name
            som_params["vocab_size"] = 2 * cfg.LOSS.sampling_method.params[sampling_name]["neighbor_size"] + 1
            self.head_som = BertMLMHead(som_params)

        # local scene match loss
        if self.use_lsm_loss:
            if cfg.MODEL.visual.visual_mode:
                crn_dim = cfg.MODEL.visual.contextual_relation_network.params[crn_name][
                    "hidden_size"
                ]
            else:
                crn_dim = cfg.MODEL.audio.contextual_relation_network.params[crn_name][
                    "hidden_size"
                ]
            self.head_lsm = Dense(crn_dim * 2, 2)
            self.lsm_padding = Embedding(1, crn_dim)


    def _compute_masked_hidden(self, hidden, mask):
        """ get only the masked region (don"t compute unnecessary hiddens) """
        mask = mask.unsqueeze(-1).expand_as(hidden).bool()
        hidden_masked = hidden[mask].contiguous().view(-1, hidden.size(-1))
        return hidden_masked

    def _compute_msm_loss(self, crn, shot_repr, masking_mask):
        """ compute Masked Shot Modeling loss """
        # infer CRN with masking
        crn_repr_w_mask, _ = crn(
            shot_repr, masking_mask
        )  # [B S+1 D]; S means # of shots

        # compute masked shot modeling loss
        crn_repr_w_cls = crn_repr_w_mask[
            :, 1:
        ].contiguous()  # exclude [CLS] token; [B S D]
        crn_repr_at_masked = self._compute_masked_hidden(
            crn_repr_w_cls, masking_mask
        )  # [M D]
        logit_at_masked = self.head_msm(crn_repr_at_masked)  # [M D]
        shot_repr_at_masked = self._compute_masked_hidden(
            shot_repr.detach(), masking_mask
        )  # [M D]
        masked_shot_loss = mse_loss(
            logit_at_masked, shot_repr_at_masked
        )  # l2 distance
        return masked_shot_loss


    def _compute_som_loss(self, crn, shot_repr, shuffled_orders, targets):
        """ compute Shot Order Modeling loss """
        # Reshuffle c_v_feats according to targets
        shuffled_orders_expanded = shuffled_orders.unsqueeze(-1).expand_as(
            shot_repr)
        shot_repr_shuffled = ms.zeros_like(
            shot_repr, dtype=shot_repr.dtype, device=shot_repr.device)
        shot_repr_shuffled = shot_repr_shuffled.scatter_(
            1, shuffled_orders_expanded, shot_repr)
        # print(shot_repr_shuffled)
        crn_repr_wo_mask, _ = crn(shot_repr_shuffled)  # infer CRN without masking
        crn_repr_wo_mask = crn_repr_wo_mask[:, 1:].contiguous()
        b, s, d = crn_repr_wo_mask.size()
        crn_repr_wo_mask = crn_repr_wo_mask.view(-1, d)
        shot_reorder_outputs = self.head_som(crn_repr_wo_mask)
        loss = cross_entropy(shot_reorder_outputs, targets.view(-1), ignore_index=-1,
                reduction='mean')
        return loss


    def _compute_lsm_loss(self, crn_repr_wo_mask, scence):
        """
            contextual group mathcing loss
            where we sample two pairs of (center shot, pos_shot), (center shot, neg_shot)
            and predict whether the pairs belong to the same group or not
        """
        assert scence is not None
        B, nshot, _ = crn_repr_wo_mask.shape
        lsm_padding = self.lsm_padding(ms.zeros(B, dtype=ms.long, device=crn_repr_wo_mask.device)).unsqueeze(1)
        crn_repr_wo_mask = ms.cat([crn_repr_wo_mask, lsm_padding], dim=1)
        center_idx = nshot // 2

        # sample shot indices from group 0 and 1
        matched_idx, no_matched_idx = [], []
        for bi in range(B):
            start_idx = scence[bi][0].item()
            end_idx = scence[bi][1].item() + 1
            group_idx = np.arange(start_idx, end_idx)
            group_cand = np.delete(group_idx, group_idx == center_idx)
            sampled_idx = np.random.choice(group_cand, size=1)[0]
            matched_idx.append(sampled_idx)

            if end_idx - start_idx == nshot:
                no_matched_idx.append(end_idx)
            else:
                group_no_cand = np.arange(0, nshot)
                group_no_cand = np.delete(group_no_cand, group_idx)
                sampled_idx = np.random.choice(group_no_cand, size=1)[0]
                no_matched_idx.append(sampled_idx)

        # obtain representations
        b_idx = ms.arange(0, B, device=crn_repr_wo_mask.device)
        center_shot_repr = normalize(crn_repr_wo_mask[:, center_idx], dim=1)  # [B D]
        pos_shot_repr = normalize(
            crn_repr_wo_mask[b_idx, matched_idx], dim=1
        )  # [B D]
        neg_shot_repr = normalize(
            crn_repr_wo_mask[b_idx, no_matched_idx], dim=1
        )  # [B D]

        logit = self.head_lsm(
            ms.cat(
                [
                    ms.cat([center_shot_repr, pos_shot_repr], dim=1),
                    ms.cat([center_shot_repr, neg_shot_repr], dim=1),
                ],
                dim=0,
            )
        )  # [2*B 2]
        label = ms.cat(
            [
                ms.ones(B, dtype=ms.long, device=crn_repr_wo_mask.device),
                ms.zeros(B, dtype=ms.long, device=crn_repr_wo_mask.device),
            ],
            dim=0,
        )  # [2*B]
        cgm_loss = cross_entropy(logit, label)
        return cgm_loss
