import logging
import os
import random
import copy

import einops
import ndjson
import numpy as np
from misc.utils import *
from dataset.base import BaseDataset
import mindspore


class MovieNetDataset(BaseDataset):
    def __init__(self, cfg, mode, is_train):
        super(MovieNetDataset, self).__init__(cfg, mode, is_train)
        self.use_visual_mode = self.cfg.MODEL.visual.visual_mode
        self.use_audio_mode = self.cfg.MODEL.audio.audio_mode
        logging.info(f"Load Dataset: {cfg.DATASET}")
        if mode == "finetune" and not self.use_raw_shot:
            assert len(self.cfg.VISUAL_PRETRAINED_LOAD_FROM) > 0
            self.visual_shot_repr_dir = os.path.join(
                self.cfg.VISUAL_FEAT_PATH, self.cfg.VISUAL_PRETRAINED_LOAD_FROM
            )
            self.audio_shot_repr_dir = os.path.join(
                self.cfg.AUDIO_FEAT_PATH, self.cfg.AUDIO_PRETRAINED_LOAD_FROM
            )

    def load_data(self):
        self.tmpl = "{}/shot_{}_img_{}.jpg"  # video_id, shot_id, shot_num
        self.tmpa = "{}.pkl"  # video_id, shot_id, shot_num
        if self.mode == "extract_shot":
            with open(
                os.path.join(self.cfg.ANNO_PATH, "anno.trainvaltest.ndjson"), "r"
            ) as f:
                self.anno_data = ndjson.load(f)

        elif self.mode == "pretrain":
            if self.is_train:
                with open(
                    os.path.join(self.cfg.ANNO_PATH, "anno.pretrain.ndjson"), "r"
                ) as f:
                    self.anno_data = ndjson.load(f)
            else:
                with open(
                    os.path.join(self.cfg.ANNO_PATH, "anno.test.ndjson"), "r"
                ) as f:
                    self.anno_data = ndjson.load(f)

        elif self.mode == "finetune":
            if self.is_train:
                with open(
                    os.path.join(self.cfg.ANNO_PATH, "anno.train.ndjson"), "r"
                ) as f:
                    self.anno_data = ndjson.load(f)

                self.vidsid2label = {
                    f"{it['video_id']}_{it['shot_id']}": it["boundary_label"]
                    for it in self.anno_data
                }

            else:
                with open(
                    os.path.join(self.cfg.ANNO_PATH, "anno.test.ndjson"), "r"
                ) as f:
                    self.anno_data = ndjson.load(f)

            self.use_raw_shot = self.cfg.USE_RAW_SHOT
            if not self.use_raw_shot:
                self.tmpl = "{}/shot_{}.npy"  # video_id, shot_id

    def _getitem_for_pretrain(self, idx: int):
        data = self.anno_data[
            idx
        ]  # contain {"video_id", "shot_id", "num_shot", "boundary_label"}
        vid = data["video_id"]
        sid = data["shot_id"]
        num_shot = data["num_shot"]
        payload = {"idx": idx, "vid": vid, "sid": sid}

        if self.sampling_method in ["instance", "temporal"]:
            # This is for two shot-level pre-training baselines:
            # 1) SimCLR (instance) and 2) SimCLR (temporal)
            keyframes, nshot = self.load_shot(vid, sid)
            view1 = self.apply_transform(keyframes)
            view1 = einops.rearrange(view1, "(s k) c ... -> s (k c) ...", s=nshot)

            new_sid = self.shot_sampler(int(sid), num_shot)
            if not new_sid == int(sid):
                keyframes, nshot = self.load_shot(vid, sid)
            view2 = self.apply_transform(keyframes)
            view2 = einops.rearrange(view2, "(s k) c ... -> s (k c) ...", s=nshot)

            # video shape: [nView=2,S,C,H,W]
            video = ms.stack([view1, view2])
            payload["video_visual"] = video

        elif self.sampling_method in ["shotcol", "bassl+shotcol", "bassl"]:
            sparse_method = "edge" if self.sampling_method == "bassl" else "edge+center"
            sparse_idx_to_dense, dense_idx = self.shot_sampler(
                    int(sid), num_shot, sparse_method=sparse_method
            )
            if self.use_visual_mode:
                _dense_video = self.load_shot_list(vid, dense_idx)
                dense_video = self.apply_transform(_dense_video)
                dense_video = dense_video.view(
                    len(dense_idx), -1, 224, 224)
                video = dense_video[:, None, :]  # [T,S=1,C,H,W]
                payload["video_visual"] = video
            if self.use_audio_mode:
                aud_feats = self.load_aud_list(vid, dense_idx)
                payload["video_audio"] = aud_feats
            payload["dense_idx"] = dense_idx
            payload["mask"] = self._get_mask(len(dense_idx))  # for MSM pretext task
            payload["shuffled_orders"], payload["orders_targets"] = self._get_random_reorder(
                list(range(len(dense_idx))))  # for SOM pretext task

        else:
            raise ValueError
        
        assert "video_visual" in payload or "video_audio" in payload
        return payload

    def _get_mask(self, N: int):
        mask = np.zeros(N).astype(np.float)

        for i in range(N):
            prob = random.random()
            # mask token with 15% probability
            if prob < 0.15:
                mask[i] = 1.0

        if (mask == 0).all():
            # at least mask 1
            ridx = random.choice(list(range(0, N)))
            mask[ridx] = 1.0
        return mask

    def _get_random_reorder(self, pos_ids, random_reorder_p=0.15):
        """
        random reorder frame positions
        """
        selected_pos = []
        target_pos = []
        for i, pos_id in enumerate(pos_ids):
            prob = random.random()
            # mask token with 15% probability
            if prob < random_reorder_p:
                selected_pos.append(i)
                target_pos.append(pos_id)
        target_pos_shuffled = copy.deepcopy(target_pos)
        random.shuffle(target_pos_shuffled)
        output_order = copy.deepcopy(pos_ids)
        output_target = [-1] * len(output_order)
        for i, pos in enumerate(selected_pos):
            output_order[pos] = target_pos_shuffled[i]
            output_target[target_pos_shuffled[i]] = pos
        return ms.tensor(output_order), ms.tensor(output_target)

    def _getitem_for_knn_val(self, idx: int):
        data = self.anno_data[
            idx
        ]  # {"video_id", "shot_id", "num_shot", "boundary_label"}
        vid = data["video_id"]
        sid = data["shot_id"]
        payload = {
            "global_video_id": data["global_video_id"],
            "sid": sid,
            "invideo_scene_id": data["invideo_scene_id"],
            "global_scene_id": data["global_scene_id"],
        }
        if self.use_visual_mode: 
            video, s = self.load_shot(vid, sid)
            video = self.apply_transform(video)
            video = einops.rearrange(video, "(s k) c ... -> s k c ...", s=s)
            payload["video_visual"] = video
        if self.use_audio_mode:
            aud_feat, s = self.load_aud(vid, sid)
            payload["video_audio"] = aud_feat
        
        assert "video_visual" in payload or "video_audio" in payload
        return payload

    def _getitem_for_extract_shot(self, idx: int):
        data = self.anno_data[
            idx
        ]  # {"video_id", "shot_id", "num_shot", "boundary_label"}
        vid = data["video_id"]
        sid = data["shot_id"]
        payload = {"vid": vid, "sid": sid}
        
        video, s = self.load_shot(vid, sid)
        video = self.apply_transform(video)
        video = einops.rearrange(video, "(s k) c ... -> s k c ...", s=s)
        payload["video_visual"] = video  # [s=1 k c h w]
        
        aud_feat, s = self.load_aud(vid, sid)
        payload["video_audio"] = aud_feat
        assert "video_visual" in payload or "video_audio" in payload
        return payload

    def _getitem_for_finetune(self, idx: int):
        data = self.anno_data[
            idx
        ]  # {"video_id", "shot_id", "num_shot", "boundary_label"}
        vid, sid = data["video_id"], data["shot_id"]
        num_shot = data["num_shot"]

        shot_idx = self.shot_sampler(int(sid), num_shot)

        if self.use_raw_shot:
            video, s = self.load_shot_list(vid, shot_idx)
            video = self.apply_transform(video)
            video = video.view(
                len(shot_idx), 1, -1, 224, 224
            )  # the shape is [S,1,C,H,W]
            aud_feats = self.load_aud_list(vid, dense_idx)

        else:
            _video = []
            _video_audio = []
            for sidx in shot_idx:
                shot_feat_path = os.path.join(
                    self.visual_shot_repr_dir, self.tmpl.format(vid, f"{sidx:04d}")
                )
                shot = np.load(shot_feat_path)
                shot = ms.from_numpy(shot)
                if len(shot.shape) > 1:
                    shot = shot.mean(0)

                _video.append(shot)
                aud_feat_path = os.path.join(
                    self.audio_shot_repr_dir, self.tmpl.format(vid, f"{sidx:04d}")
                )
                aud_feat = np.load(aud_feat_path)
                aud_feat = ms.from_numpy(aud_feat)
                if len(aud_feat.shape) > 1:
                    aud_feat = aud_feat.mean(0)
                _video_audio.append(aud_feat)
            video = ms.stack(_video, dim=0)
            video_audio = ms.stack(_video_audio, dim=0)
        payload = {
            "idx": idx,
            "vid": vid,
            "sid": sid,
            "video_visual": video,
            "video_audio": video_audio,
            "label": abs(data["boundary_label"]),  # ignore -1 label.
        }
        return payload

    def _getitem_for_sbd_eval(self, idx: int):
        return self._getitem_for_finetune(idx)

    def __getitem__(self, idx: int):
        if self.mode == "extract_shot":
            return self._getitem_for_extract_shot(idx)

        elif self.mode == "pretrain":
            if self.is_train:
                return self._getitem_for_pretrain(idx)
            else:
                return self._getitem_for_knn_val(idx)

        elif self.mode == "finetune":
            if self.is_train:
                return self._getitem_for_finetune(idx)
            else:
                return self._getitem_for_sbd_eval(idx)
