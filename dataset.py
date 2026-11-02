import json
import os
import random
import cv2
import numpy as np
import torch
from torch.utils import data


class FrameOneDataset(data.Dataset):

    def __init__(self, json_path, root_dir, cfg, split="train"):
        self.root_dir = root_dir
        self.cfg = cfg
        self.split = split
        self.fixed_length = cfg["dataset"]["fixed_length"]
        self.resize = cfg["dataset"]["resize"]
        self.is_aug_image = cfg["dataset"]["is_aug_image"]
        with open(json_path, encoding="utf-8") as handle:
            self.samples = json.load(handle)

    def __len__(self):
        return len(self.samples)

    def label_base(self, meta):
        if meta.get("labels_zero_based", False):
            return 0
        cfg = self.cfg["dataset"]
        base = cfg.get("label_index_base_by_dataset", {}).get(
            meta.get("dataset_name", "Unknown"), cfg.get("label_index_base", 0)
        )
        for marker, value in cfg.get("label_index_base_by_path", {}).items():
            if marker in meta["file_path"]:
                base = value
                break
        if base not in (0, 1):
            raise ValueError("label_index_base must be 0 or 1.")
        return int(base)

    def __getitem__(self, idx):
        meta = self.samples[idx]
        video_path = meta["file_path"]
        if not os.path.isabs(video_path):
            video_path = os.path.join(self.root_dir, video_path)
        video = self._load_video(video_path)
        base = self.label_base(meta)
        es = int(meta["keyframe"]["ES"]) - base
        ed = int(meta["keyframe"]["ED"]) - base
        if not (0 <= es < len(video) and 0 <= ed < len(video)):
            raise ValueError(
                f"Invalid keyframes for {video_path}: ES={es}, ED={ed}, length={len(video)}"
            )
        if self.split == "train":
            (video, es, ed) = self._build_repeat_sample(video, es, ed, meta["view"])
            if self.is_aug_image:
                video = self._aug_video(video)
        if self.resize is not None:
            video = np.stack(
                [
                    cv2.resize(
                        frame.transpose(1, 2, 0), (self.resize, self.resize)
                    ).transpose(2, 0, 1)
                    for frame in video
                ]
            )
        view_idx = self.cfg["dataset"]["view_list"].index(meta["view"])
        return {
            "patient_id": meta["patient_id"],
            "video": torch.from_numpy(video.copy()).float(),
            "label_view": meta["view"],
            "view_idx": torch.tensor(view_idx),
            "label_ES": es,
            "label_ED": ed,
            "dataset_name": meta.get("dataset_name", "Unknown"),
        }

    def _load_video(self, path):
        cap = cv2.VideoCapture(path)
        frames = []
        try:
            while True:
                (ok, frame) = cap.read()
                if not ok:
                    break
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        finally:
            cap.release()
        if not frames:
            raise ValueError(f"No video frames could be read: {path}")
        return (np.stack(frames).astype(np.float32) / 255.0).transpose(0, 3, 1, 2)

    def _build_repeat_sample(self, video, ES, ED, view_name):
        T = video.shape[0]
        dist = abs(ED - ES)
        aug_spacing_views = self.cfg["dataset"].get("aug_spacing_views", ["PSAX"])
        if self.cfg["dataset"]["is_aug_spacing"] and view_name in aug_spacing_views:
            min_spacing = self.cfg["dataset"]["min_spacing"]
            p_extra_downsample = self.cfg["dataset"].get("p_extra_downsample", 0.3)
            min_video_len = 8
            while dist > min_spacing and T > min_video_len:
                video = video[::2]
                ES //= 2
                ED //= 2
                T = video.shape[0]
                dist = abs(ED - ES)
            min_spacing_stop = self.cfg["dataset"].get("min_spacing_stop", 5)
            while T > min_video_len and dist > min_spacing_stop:
                if random.random() > p_extra_downsample:
                    break
                video = video[::2]
                ES //= 2
                ED //= 2
                T = video.shape[0]
                dist = abs(ED - ES)
        else:
            while dist > self.fixed_length // 2:
                video = video[::2]
                ES //= 2
                ED //= 2
                dist = abs(ED - ES)
        first = min(ES, ED)
        last = max(ES, ED)
        period_video = []
        period_ES = []
        period_ED = []
        period_video.append(video[first])
        period_ES.append(first == ES)
        period_ED.append(first == ED)
        for i in range(first + 1, last):
            period_video.append(video[i])
            period_ES.append(i == ES)
            period_ED.append(i == ED)
        period_video.append(video[last])
        period_ES.append(last == ES)
        period_ED.append(last == ED)
        for i in range(last - 1, first, -1):
            period_video.append(video[i])
            period_ES.append(i == ES)
            period_ED.append(i == ED)
        period_video = np.stack(period_video)
        period_ES = np.array(period_ES, dtype=bool)
        period_ED = np.array(period_ED, dtype=bool)
        num_repeats = 3
        rep_video = np.tile(period_video, (num_repeats, 1, 1, 1))
        rep_ES = np.tile(period_ES, num_repeats)
        rep_ED = np.tile(period_ED, num_repeats)
        max_start = rep_video.shape[0] - self.fixed_length
        if max_start < 0:
            pad_length = self.fixed_length - rep_video.shape[0]
            clip = np.pad(
                rep_video, ((0, pad_length), (0, 0), (0, 0), (0, 0)), mode="edge"
            )
            clip_ES = np.pad(
                rep_ES, (0, pad_length), mode="constant", constant_values=False
            )
            clip_ED = np.pad(
                rep_ED, (0, pad_length), mode="constant", constant_values=False
            )
        else:
            start_idx = np.random.randint(0, max_start + 1)
            clip = rep_video[start_idx : start_idx + self.fixed_length]
            clip_ES = rep_ES[start_idx : start_idx + self.fixed_length]
            clip_ED = rep_ED[start_idx : start_idx + self.fixed_length]
        if not clip_ES.any() or not clip_ED.any():
            raise ValueError("No ES or ED frame found in the sampled clip.")
        return (clip, clip_ES.astype(np.float32), clip_ED.astype(np.float32))

    def _aug_video(
        self,
        video,
        p_flip=0.5,
        p_rotate=0.5,
        max_angle=10,
        p_scale=0.5,
        scale_range=(0.9, 1.1),
        p_noise=0.5,
        noise_std=0.01,
    ):
        (T, C, H, W) = video.shape
        video_aug = video
        if random.random() < p_flip:
            np.copyto(video_aug, video_aug[:, :, :, ::-1])
        angle = 0
        scale = 1.0
        if random.random() < p_rotate:
            angle = random.uniform(-max_angle, max_angle)
        if random.random() < p_scale:
            scale = random.uniform(scale_range[0], scale_range[1])
        if angle != 0 or scale != 1.0:
            video_aug = np.transpose(video_aug, (0, 2, 3, 1))
            center = (W / 2, H / 2)
            M = cv2.getRotationMatrix2D(center, angle, scale)

            def warp_frame(frame):
                return cv2.warpAffine(
                    frame,
                    M,
                    (W, H),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT,
                )

            video_aug = np.stack([warp_frame(frame) for frame in video_aug], axis=0)
            video_aug = np.transpose(video_aug, (0, 3, 1, 2))
        if random.random() < p_noise:
            noise = np.random.normal(0, noise_std, video_aug.shape)
            video_aug += noise
            np.clip(video_aug, 0.0, 1.0, out=video_aug)
        return video_aug
