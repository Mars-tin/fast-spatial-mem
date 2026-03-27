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
import torch
from PIL import Image


def resize_and_crop(image, target_size, fxfycxcy):
    """Resize an image to cover the target size, then center-crop it.

    The camera intrinsics are adjusted to match the resize-and-crop operation.

    Args:
        image: Input PIL image.
        target_size: Target image size as (height, width).
        fxfycxcy: Camera intrinsics as [fx, fy, cx, cy].

    Returns:
        tuple: A tuple of:
            - The resized and center-cropped PIL image.
            - The adjusted camera intrinsics [fx, fy, cx, cy].
    """
    original_width, original_height = image.size  # PIL image is (width, height)
    target_height, target_width = target_size
    fx, fy, cx, cy = fxfycxcy
    
    # Calculate scale factor to fill target size (resize to cover)
    scale_x = target_width / original_width
    scale_y = target_height / original_height
    scale = max(scale_x, scale_y)  # Use larger scale to ensure it covers the target area
    
    # Resize image
    new_width = int(round(original_width * scale))
    new_height = int(round(original_height * scale))
    resized_image = image.resize((new_width, new_height), Image.LANCZOS)
    
    # Calculate crop box for center crop
    left = (new_width - target_width) // 2
    top = (new_height - target_height) // 2
    right = left + target_width
    bottom = top + target_height
    
    # Crop image
    cropped_image = resized_image.crop((left, top, right, bottom))
    
    # Adjust camera parameters
    # Scale focal lengths and principal points
    new_fx = fx * scale
    new_fy = fy * scale
    new_cx = cx * scale - left
    new_cy = cy * scale - top
    
    return cropped_image, [new_fx, new_fy, new_cx, new_cy]


def normalize(x):
    """Normalize a tensor by its Euclidean norm.

    Args:
        x: Input tensor.

    Returns:
        Normalized tensor.
    """
    return x / x.norm()


def normalize_with_mean_pose(c2ws: torch.Tensor):
    """Normalize camera poses using the mean camera frame.

    This historical utility recenters the scene around the mean camera pose and
    rescales translations into approximately [-1, 1].

    Cr. Zexiang Xu and collaborators for the original scene normalization code

    Args:
        c2ws: Camera-to-world matrices of shape [N, 4, 4].

    Returns:
        torch.Tensor: Normalized camera-to-world matrices.
    """
    # Compute the mean camera center and orientation axes.
    center = c2ws[:, :3, 3].mean(0)
    vec2 = c2ws[:, :3, 2].mean(0)
    up = c2ws[:, :3, 1].mean(0)

    # Build the mean camera frame.
    vec2 = normalize(vec2)
    vec0 = normalize(torch.cross(up, vec2, dim=-1))
    vec1 = normalize(torch.cross(vec2, vec0, dim=-1))
    mean_pose = torch.stack([vec0, vec1, vec2, center], dim=1)

    # Extend the mean pose to a 4x4 transform.
    avg_pos = c2ws.new_zeros(4, 4)
    avg_pos[3, 3] = 1.0
    avg_pos[:3] = mean_pose

    # Align all cameras to the mean camera frame.
    c2ws = torch.linalg.inv(avg_pos) @ c2ws

    # Rescale scene translations to approximately [-1, 1].
    scene_scale = torch.max(torch.abs(c2ws[:, :3, 3]))
    c2ws[:, :3, 3] /= scene_scale

    return c2ws


def normalize_intrinsics(fxfycxcy, w, h):
    """Normalize camera intrinsics by image width and height.

    Args:
        fxfycxcy: Tensor containing intrinsics in the order [fx, fy, cx, cy].
        w: Image width.
        h: Image height.

    Returns:
        torch.Tensor: Normalized intrinsics [fx / w, fy / h, cx / w, cy / h].
    """
    fx, fy, cx, cy = fxfycxcy.unbind(-1)
    return torch.stack([fx / w, fy / h, cx / w, cy / h], dim=-1)
