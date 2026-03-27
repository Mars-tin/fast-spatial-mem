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

from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from .utils import (
    resize_and_crop, 
    normalize_with_mean_pose
)

class NVSDatasetInference(Dataset):
    def __init__(self, 
        data_path, num_views, image_size, 
        sorted_indices=False, 
        scene_pose_normalize=False,
        max_frame_time=256,
        num_input_views=8,
        window_size=128,
        is_inference=False,
    ):
        """
        image_size is (h, w) or just a int (as size).
        """
        self.base_dir = os.path.dirname(data_path)
        self.data_point_paths = json.load(open(data_path, "r"))
        self.sorted_indices = sorted_indices
        self.num_input_views = num_input_views
        self.scene_pose_normalize = scene_pose_normalize

        self.num_views = num_views
        self.max_frame_time = max_frame_time
        self.window_size = window_size
        self.is_inference = is_inference
        
        if isinstance(image_size, int):
            self.image_size = (image_size, image_size)
        else:
            self.image_size = image_size

    def __len__(self):
        return len(self.data_point_paths)
    
    def __getitem__(self, index):
        index = index % len(self.data_point_paths)  # Ensure index is within bounds
        data_point_path = os.path.join(self.base_dir, self.data_point_paths[index])
        data_point_base_dir = os.path.dirname(data_point_path)
        with open(data_point_path, "r") as f:
            full_json = json.load(f)

        # Support backward compatibility
        images_info = full_json["frames"] if "frames" in full_json else full_json
        total_len = len(images_info)
        json_id = self.data_point_paths[index]   # include this

        if self.is_inference and isinstance(full_json, dict) and ("index_plan" in full_json):
            
            print('using fixed index')
            plan = full_json["index_plan"]
            input_idx = plan.get("input_index", [])
            target_idx = plan.get("target_index", [])

            # Strictly trust your precomputed indices (no filtering).
            indices = input_idx + target_idx

        else:
            if total_len < self.num_views:
                new_index = random.randint(0, len(self.data_point_paths) - 1)
                return self.__getitem__(new_index)

            # Select a continuous window
            window_size = self.window_size
            if window_size > total_len:
                window_size = total_len

            max_start_index = total_len - window_size
            start_index = random.randint(0, max_start_index)
            candidate_indices = list(range(start_index, start_index + window_size))

            # Sample num_views frames from the window
            indices = random.sample(candidate_indices, self.num_views)
            if self.sorted_indices:
                indices = sorted(indices[:self.num_input_views]) + indices[self.num_input_views:]
        
        fxfycxcy_list = []
        c2w_list = []
        image_list = []
        frame_time_list = []

        for index in indices:
            info = images_info[index]
            fxfycxcy = [info["fx"], info["fy"], info["cx"], info["cy"]]
            if 'frame_time' in info:
                f_time = info["frame_time"]
            else:
                # DL3DV, get frame_time from the file name frame_xxxxx.png
                f_time = int(os.path.splitext(info["file_path"])[0].split("_")[-1])
                
            if f_time >= self.max_frame_time:
                f_time %= self.max_frame_time
            
            w2c = torch.tensor(info["w2c"])
            c2w = torch.inverse(w2c)
            c2w_list.append(c2w)
            frame_time_list.append(f_time)
            
            # Load image from file_path using PIL and convert to torch tensor
            image_path = os.path.join(data_point_base_dir, info["file_path"])
            image = Image.open(image_path)
            
            image, fxfycxcy = resize_and_crop(image, self.image_size, fxfycxcy)

            # Convert RGBA to RGB if needed
            if image.mode == 'RGBA':
                # Create a white background and paste the RGBA image on it
                rgb_image = Image.new('RGB', image.size, (255, 255, 255))
                rgb_image.paste(image, mask=image.split()[-1])  # Use alpha channel as mask
                image = rgb_image
            elif image.mode != 'RGB':
                # Convert any other mode to RGB
                image = image.convert('RGB')
            
            fxfycxcy_list.append(fxfycxcy)
            image_list.append(transforms.ToTensor()(image))
        
        c2ws = torch.stack(c2w_list)
        frame_times = torch.tensor(frame_time_list)
            
        if self.scene_pose_normalize:
            c2ws = normalize_with_mean_pose(c2ws)
            if torch.isnan(c2ws).any():
                print('Getting nan on data path:', data_point_path)
                new_index = random.randint(0, len(self.data_point_paths) - 1)
                return self.__getitem__(new_index)

        return {
            "fxfycxcy": torch.tensor(fxfycxcy_list),
            "c2w": c2ws,
            "image": torch.stack(image_list),
            "frame_time": frame_times,
            "json_id": json_id,   
        }
