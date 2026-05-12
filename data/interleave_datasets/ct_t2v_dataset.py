# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import traceback

import numpy as np
from PIL import Image, ImageFile, PngImagePlugin

from .interleave_t2i_dataset import InterleavedBaseIterableDataset


Image.MAX_IMAGE_PIXELS = 200000000
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2 ** 20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte


class CTText2VideoIterableDataset(InterleavedBaseIterableDataset):
    """
    JSONL sample format (one dict per line):
    {"prompt": "...", "img_path": "/abs/path/to/xxx.npy"}

    npy shape expected:
    - (1, D, H, W), e.g. (1, 32, 384, 384), or
    - (D, H, W)

    Frame grouping rule:
    - non-overlap groups of 3 slices: [0,1,2], [3,4,5], ...
    - if remainder == 2, append one overlapped tail frame: [D-3, D-2, D-1]
      For D=32 -> 10 non-overlap + 1 tail overlap = 11 frames
    """

    def __init__(
        self,
        dataset_name,
        transform,
        tokenizer,
        jsonl_path_list,
        num_used_data,
        data_dir_list=None,
        local_rank=0,
        world_size=1,
        num_workers=8,
        data_status=None,
        shuffle_lines=False,
        shuffle_seed=0,
        **kwargs, 
    ):
        super().__init__(dataset_name, local_rank, world_size, num_workers)
        self.transform = transform
        self.tokenizer = tokenizer
        self.data_status = data_status
        self.data_paths = self.get_data_paths(
            jsonl_path_list=jsonl_path_list,
            num_used_data=num_used_data,
            shuffle_lines=shuffle_lines,
            shuffle_seed=shuffle_seed,
        )
        if len(self.data_paths) == 0:
            raise ValueError("ct_t2v: data_paths is empty, check json/jsonl path and num_used_data.")
        self.set_epoch()

    def get_data_paths(self, jsonl_path_list, num_used_data, shuffle_lines, shuffle_seed):
        data_paths = []
        for path, n in zip(jsonl_path_list, num_used_data):
            with open(path, "r", encoding="utf-8") as f:
                txt = f.read().strip()

            if txt.startswith("["):
                # json list
                raw = json.loads(txt)  # list[dict]
                if shuffle_lines:
                    self.rng.seed(shuffle_seed)
                    self.rng.shuffle(raw)
                raw = raw[:n]
                # 关键：转成 str，兼容 set_epoch
                data_paths.extend([json.dumps(x, ensure_ascii=False) for x in raw])
            else:
                # jsonl: each line is a json object
                lines = txt.splitlines()
                if shuffle_lines:
                    self.rng.seed(shuffle_seed)
                    self.rng.shuffle(lines)
                lines = lines[:n]
                data_paths.extend(lines)

        return data_paths

    @staticmethod
    def _ensure_volume_shape(arr: np.ndarray) -> np.ndarray:
        """Return (D, H, W)."""
        if arr.ndim == 4:
            # expected (1, D, H, W)
            if arr.shape[0] != 1:
                raise ValueError(f"Expected first dim == 1 for 4D input, got {arr.shape}")
            arr = arr[0]
        elif arr.ndim != 3:
            raise ValueError(f"Expected 3D or 4D array, got shape={arr.shape}")
        return arr

    @staticmethod
    def _to_uint8(x: np.ndarray) -> np.ndarray:
        """x in [0,1] -> uint8"""
        x = np.clip(x, 0.0, 1.0)
        return (x * 255.0 + 0.5).astype(np.uint8)

    def _build_rgb_frames_no_overlap_with_tail(self, volume_dhw: np.ndarray):
        """
        volume_dhw: (D, H, W), values in [0,1]
        returns:
          frames: list[PIL.Image RGB]
          frame_indexes: list[int]  # use first-slice index as temporal index
        """
        D, H, W = volume_dhw.shape
        if D < 3:
            raise ValueError(f"Need at least 3 slices, got D={D}")

        frames = []
        frame_indexes = []

        # Non-overlap triplets: 0,3,6,...
        starts = list(range(0, D - 2, 3))
        for s in starts:
            # only full triplet
            if s + 2 < D:
                rgb = np.stack(
                    [volume_dhw[s], volume_dhw[s + 1], volume_dhw[s + 2]],
                    axis=-1
                )  # (H, W, 3)
                img = Image.fromarray(self._to_uint8(rgb), mode="RGB")
                frames.append(img)
                frame_indexes.append(s)

        # Remainder handling:
        # If D % 3 == 2 (e.g. D=32), append tail overlap frame [D-3, D-2, D-1]
        if D % 3 == 2:
            s = D - 3
            if len(frame_indexes) == 0 or frame_indexes[-1] != s:
                rgb = np.stack(
                    [volume_dhw[s], volume_dhw[s + 1], volume_dhw[s + 2]],
                    axis=-1
                )
                img = Image.fromarray(self._to_uint8(rgb), mode="RGB")
                frames.append(img)
                frame_indexes.append(s)

        return frames, frame_indexes

    def __iter__(self):
        data_paths_per_worker, worker_id = self.get_data_paths_per_worker()
        if self.data_status is not None:
            row_start_id = self.data_status[worker_id] + 1
        else:
            row_start_id = 0

        print(
            f"rank-{self.local_rank} worker-{worker_id} dataset-{self.dataset_name}: "
            f"resuming data at row#{row_start_id}"
        )

        while True:
            data_paths_per_worker_ = data_paths_per_worker[row_start_id:]
            for row_idx, line in enumerate(data_paths_per_worker_, start=row_start_id):
                try:
                    item = json.loads(line)
                    prompt = item["prompt"]         # key fixed as requested
                    img_path = item["img_path"]     # absolute path
                except Exception:
                    traceback.print_exc()
                    continue

                try:
                    vol = np.load(img_path)  # (1,D,H,W) or (D,H,W), in [0,1]
                    vol = self._ensure_volume_shape(vol)
                    frames, frame_indexes = self._build_rgb_frames_no_overlap_with_tail(vol)
                except Exception:
                    traceback.print_exc()
                    continue

                if len(frames) == 0:
                    continue

                data = self._init_data()

                # text condition (no text loss for t2v generation condition)
                data = self._add_text(
                    data=data,
                    text=prompt,
                    need_loss=False,
                    enable_cfg=True,
                )

                # video target (MSE loss on vae tokens)
                data = self._add_video(
                    data=data,
                    frames=frames,
                    frame_indexes=frame_indexes,
                    need_loss=True,
                    need_vae=False,
                    enable_cfg=False,
                )

                yield dict(
                    image_tensor_list=data["image_tensor_list"],
                    text_ids_list=data["text_ids_list"],
                    sequence_plan=data["sequence_plan"],
                    num_tokens=data["num_tokens"],
                    data_indexes={
                        "data_indexes": row_idx,
                        "worker_id": worker_id,
                        "dataset_name": self.dataset_name,
                        "img_path": img_path,
                    },
                )

            row_start_id = 0
            print(f"{self.dataset_name} repeat in rank-{self.local_rank} worker-{worker_id}")
