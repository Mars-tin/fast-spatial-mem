#
# This file is derived from code released under the MIT License.
#
# Original copyright:
# Copyright (c) 2025 Tianyuan Zhang, Hao Tan
#
# Modifications copyright:
# Copyright (c) 2026 Martin Ziqiao Ma
#
# This file is distributed under the licensing terms provided in the
# repository LICENSE and NOTICE files.
#
import json
import os
import random
from typing import List, Dict

from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from .utils import (
    resize_and_crop, 
    normalize_with_mean_pose
)

# Decord for frame-level video IO
try:
    from decord import VideoReader, cpu
except Exception as e:
    raise ImportError(f"Failed to import decord. Install with `pip install decord`.\n{e}")

class NVSVideoDataset(Dataset):
    """
    Video-based variant of NVSDataset.

    Expects `data_path` to be a JSON file that lists relative paths to *per-video* JSON files.
    Each per-video JSON has fields:
      {
        "scene_name": "...",
        "video_path": "/abs/or/rel/path/to/clip.mp4",
        "frames": [
           {
             "frame_idx": 0,
             "w2c": [[...4],[...4],[...4],[...4]],
             "fx": ..., "fy": ..., "cx": ..., "cy": ...,
             "h": H, "w": W,
             # optional "timestamp": ...
           },
           ...
        ]
      }
    """
    def __init__(
        self,
        data_path: str,
        num_views: int,
        image_size,
        sorted_indices: bool = False,
        scene_pose_normalize: bool = False,
        max_frame_time: int = 256,
        num_input_views: int = 8,
        window_size: int = 128,
        is_inference: bool = False
    ):
        """
        image_size is (h, w) or an int (square).
        """
        self.base_dir = os.path.dirname(data_path)
        self.data_point_paths: List[str] = json.load(open(data_path, "r"))
        self.sorted_indices = sorted_indices
        self.scene_pose_normalize = scene_pose_normalize
        self.num_views = num_views
        self.num_input_views = num_input_views
        self.max_frame_time = max_frame_time
        self.window_size = window_size
        self.image_size = (image_size, image_size) if isinstance(image_size, int) else image_size

        # cache per-worker decord readers to avoid reopening for repeated items
        self._vr_cache: Dict[str, VideoReader] = {}

        # simple image to tensor
        self._to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.data_point_paths)

    def _load_scene_json(self, path: str):
        """
        Load a scene file that might be:
        - pretty JSON (multi-line object or array)
        - JSONL (one object per line)
        """
        with open(path, "r", encoding="utf-8-sig") as f:
            first_nonempty = None
            for line in f:
                s = line.strip()
                if s:
                    first_nonempty = s
                    break
            if first_nonempty is None:
                raise ValueError(f"{path} is empty")

        # If the file only had one object (JSONL with one line), this is fine
        try:
            return json.loads(first_nonempty)
        except json.JSONDecodeError:
            # Otherwise, try reading the whole file as JSON
            with open(path, "r", encoding="utf-8-sig") as f:
                return json.load(f)

    def _get_vr(self, video_path: str) -> VideoReader:
        # Absolute path preferred to stabilize cache key
        video_path = os.path.abspath(video_path)
        vr = VideoReader(video_path, ctx=cpu(0))
        return vr

    def __getitem__(self, index):
        index = index % len(self.data_point_paths)
        data_point_path = self.data_point_paths[index]
        with open(data_point_path, "r", encoding="utf-8") as f:
            scene_obj = json.load(f)
        
        # support older structure that nested under "frames"
        frames_meta = scene_obj["frames"] if "frames" in scene_obj else scene_obj
        total_len = len(frames_meta)
        if total_len < self.num_views:
            # resample another scene
            new_index = random.randint(0, len(self.data_point_paths) - 1)
            return self.__getitem__(new_index)

        # sample a continuous window then pick num_views within that window
        window_size = min(self.window_size, total_len)  # your hardcoded value
        max_start = total_len - window_size
        start_index = random.randint(0, max_start) if max_start > 0 else 0
        candidate_indices = list(range(start_index, start_index + window_size))
        chosen_indices = random.sample(candidate_indices, self.num_views)
        if self.sorted_indices:
            chosen_indices = sorted(chosen_indices[:self.num_input_views]) + chosen_indices[self.num_input_views:]

        video_path = scene_obj["video_path"]
        vr = self._get_vr(video_path)
            
        fxfycxcy_list, c2w_list, image_list, frame_time_list = [], [], [], []

        for i in chosen_indices:
            info = frames_meta[i]

            # intrinsics
            fxfycxcy = [info["fx"], info["fy"], info["cx"], info["cy"]]

            # frame time: prefer provided timestamp, otherwise use frame_idx; modulo cap
            f_time = int(info["frame_idx"])

            # pose: JSON stores w2c; convert to c2w
            w2c = torch.tensor(info["w2c"], dtype=torch.float32)  # (4,4)
            c2w = torch.inverse(w2c)

            # read frame from video (HWC, uint8, RGB)
            frame_idx = int(info["frame_idx"]) if "frame_idx" in info else int(i)
            frame_np = vr[frame_idx].asnumpy()  # (H, W, 3)
            pil_img = Image.fromarray(frame_np)

            # apply resize+crop and adjust intrinsics
            pil_img, fxfycxcy = resize_and_crop(pil_img, self.image_size, fxfycxcy)

            # color mode normalization
            if pil_img.mode == 'RGBA':
                bg = Image.new('RGB', pil_img.size, (255, 255, 255))
                bg.paste(pil_img, mask=pil_img.split()[-1])
                pil_img = bg
            elif pil_img.mode != 'RGB':
                pil_img = pil_img.convert('RGB')

            # collect
            fxfycxcy_list.append(fxfycxcy)
            c2w_list.append(c2w)
            img_tensor = self._to_tensor(pil_img)
            image_list.append(img_tensor)
            frame_time_list.append(f_time)

        c2ws = torch.stack(c2w_list, dim=0)
        frame_times = torch.tensor(frame_time_list, dtype=torch.long)
        t_min = frame_times.min()
        t_max = frame_times.max()
        if t_max > t_min:
            frame_times = ((frame_times - t_min) / (t_max - t_min) * (self.max_frame_time - 1)).long()
        else:
            frame_times = torch.zeros_like(frame_times).long()
            
        if self.scene_pose_normalize:
            c2ws = normalize_with_mean_pose(c2ws)
            if torch.isnan(c2ws).any():
                print('Getting nan on data path:', data_point_path, 'resampling data point...')
                new_index = random.randint(0, len(self.data_point_paths) - 1)
                return self.__getitem__(new_index)

        return {
            "fxfycxcy": torch.tensor(fxfycxcy_list, dtype=torch.float32),  # (V, 4)
            "c2w": c2ws,                                                   # (V, 4, 4)
            "image": torch.stack(image_list, dim=0),                       # (V, 3, H, W)
            "frame_time": frame_times                                      # (V,)
        }
