#
#  Apache 2.0 License License
#  Copyright (c) 2026 Martin Ziqiao Ma
#
import json
import bisect
from pathlib import Path

import torch.distributed as dist
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler

from .dataset import NVSDataset
from .dataset_fromvid import NVSVideoDataset
from .dataset_inference import NVSDatasetInference
from .dataset_inference_fromvid import NVSVideoDatasetInference

MAGIC_SEED = 42
PATH_MAP = "path2data.json"


def load_dataset_paths(json_path: str | Path):
    """Load dataset paths from a JSON file.

    Args:
        json_path: Path to the JSON file.

    Returns:
        The parsed JSON content.
    """
    with open(json_path, "r") as f:
        return json.load(f)


def get_dataset(
    dataset_ratios: dict,
    num_views,
    img_size,
    num_input_views,
    max_frame_time=256,
    scene_pose_normalize=True,
    window_size=128,
    sorted_indices=True,
    drop_last=True
):
    """
    Load datasets listed in dataset_names and build a mixed dataset.

    Args:
        dataset_ratios (dict): dataset names and their mixing ratio to load.
        num_views (int): Number of views to sample.
        img_size (tuple): Image size (H, W).
        scene_pose_normalize: Whether to normalize camera poses per scene.
    """

    with open(PATH_MAP, "r") as f:
        dataset_paths = json.load(f)

    # helper to pick dataset class based on known dataset name
    def make_dataset(name, path):
        if name in ["re10k", "stereo4d", "multicam", "example"]:
            return TagDataset(NVSVideoDataset(path, num_views, img_size,
                                              scene_pose_normalize=scene_pose_normalize, 
                                              sorted_indices=sorted_indices, 
                                              max_frame_time=max_frame_time,
                                              num_input_views=num_input_views,
                                              window_size=window_size), name)
        else:
            return TagDataset(NVSDataset(path, num_views, img_size,
                                         scene_pose_normalize=scene_pose_normalize, 
                                         sorted_indices=sorted_indices, 
                                         max_frame_time=max_frame_time,
                                         num_input_views=num_input_views,
                                         window_size=window_size), name)

    # build component datasets
    component_datasets = []
    for name in dataset_ratios.keys():
        path = dataset_paths.get(name, None)
        try:
            component_datasets.append(make_dataset(name, path))
        except Exception as e:
            print(f"Skipping {name} due to error: {e}")

    name_order = [ds.tag for ds in component_datasets]
    repeats = [int(dataset_ratios[name]) for name in name_order]  # must be >=1
    merged_dataset = RepeatedConcatDataset(component_datasets, repeats)

    print(f"Loaded {len(component_datasets)} datasets (repeated merge):")
    for ds, rep in zip(component_datasets, repeats):
        print(f"  • {ds.tag:10s} repeat={rep}  base_len={len(ds)}  contrib={rep*len(ds)}")
    print(f"Total merged length: {len(merged_dataset)}")

    # simple distributed sampler w/ shuffle
    if dist.is_available() and dist.is_initialized():
        sampler = DistributedSampler(
            merged_dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=True,
            drop_last=drop_last,
            seed=MAGIC_SEED,
        )
    else:
        sampler = None  # single-process: you can just use shuffle=True in DataLoader

    return merged_dataset, sampler


def get_dataset_inference(
    dataset_ratios: dict,
    num_views,
    img_size,
    num_input_views,
    max_frame_time=256,
    scene_pose_normalize=True,
    window_size=128,
    is_inference=True,
    sorted_indices=True
):
    """
    Load datasets listed in dataset_names and build a mixed dataset.

    Args:
        dataset_ratios (dict): dataset names and their mixing ratio to load.
        num_views (int): Number of views to sample.
        img_size (tuple): Image size (H, W).
        scene_pose_normalize: Whether to normalize camera poses per scene.
    """

    with open(PATH_MAP, "r") as f:
        dataset_paths = json.load(f)

    # helper to pick dataset class based on known dataset name
    def make_dataset(name, path):
        if name in ["re10k_test", "stereo4d_test", "example"]:
            return TagDataset(NVSVideoDatasetInference(path, num_views, img_size,
                                              scene_pose_normalize=scene_pose_normalize, 
                                              sorted_indices=sorted_indices, 
                                              max_frame_time=max_frame_time,
                                              num_input_views=num_input_views,
                                              window_size=window_size, 
                                              is_inference=is_inference), name)
        else:
            return TagDataset(NVSDatasetInference(path, num_views, img_size,
                                         scene_pose_normalize=scene_pose_normalize, 
                                         sorted_indices=sorted_indices, 
                                         max_frame_time=max_frame_time,
                                         num_input_views=num_input_views,
                                         window_size=window_size,
                                         is_inference=is_inference), name)

    # build component datasets
    component_datasets = []
    for name in dataset_ratios.keys():
        path = dataset_paths.get(name, None)
        try:
            component_datasets.append(make_dataset(name, path))
        except Exception as e:
            print(f"Skipping {name} due to error: {e}")

    name_order = [ds.tag for ds in component_datasets]
    repeats = [int(dataset_ratios[name]) for name in name_order]  # must be >=1
    merged_dataset = RepeatedConcatDataset(component_datasets, repeats)

    print(f"Loaded {len(component_datasets)} datasets (repeated merge):")
    for ds, rep in zip(component_datasets, repeats):
        print(f"  • {ds.tag:10s} repeat={rep}  base_len={len(ds)}  contrib={rep*len(ds)}")
    print(f"Total merged length: {len(merged_dataset)}")

    # simple distributed sampler w/ shuffle
    if dist.is_available() and dist.is_initialized():
        sampler = DistributedSampler(
            merged_dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=False,
            drop_last=False,
            seed=MAGIC_SEED,
        )
    else:
        sampler = None  # single-process: you can just use shuffle=True in DataLoader

    return merged_dataset, sampler


class RepeatedConcatDataset(Dataset):
    """Concatenate datasets with per-dataset virtual repetition.

    The total length is:

        sum(repeats[i] * len(datasets[i]) for i in range(len(datasets)))

    Each dataset is repeated virtually rather than copied. Indexing first
    selects the dataset block, then maps into that dataset by modulo its
    original length.
    """

    def __init__(self, datasets: list[Dataset], repeats: list[int]):
        """Initialize the repeated concatenation wrapper.

        Args:
            datasets: List of datasets to concatenate.
            repeats: Number of virtual repeats for each dataset.

        Raises:
            ValueError: If any repeat count is less than 1.
        """
        assert len(datasets) == len(repeats)

        self.datasets = datasets
        self.repeats = [int(r) for r in repeats]

        for repeat in self.repeats:
            if repeat <= 0:
                raise ValueError(f"repeat must be >= 1, got {repeat}")

        self.base_lens = [len(dataset) for dataset in self.datasets]
        self.block_lens = [
            repeat * base_len
            for repeat, base_len in zip(self.repeats, self.base_lens)
        ]

        # Prefix sums over repeated dataset blocks, used for block lookup.
        self.cum = []
        total = 0
        for block_len in self.block_lens:
            total += block_len
            self.cum.append(total)

    def __len__(self):
        """Return the total virtual length of the concatenated dataset."""
        return self.cum[-1] if self.cum else 0

    def __getitem__(self, idx: int):
        """Return the sample at the given virtual index.

        Args:
            idx: Dataset index. Negative indices are supported.

        Returns:
            A sample from the selected underlying dataset.

        Raises:
            IndexError: If the index is out of range.
        """
        if idx < 0:
            idx = len(self) + idx

        if idx < 0 or idx >= len(self):
            raise IndexError(idx)

        dataset_idx = bisect.bisect_right(self.cum, idx)
        prev_cum = 0 if dataset_idx == 0 else self.cum[dataset_idx - 1]
        offset_in_block = idx - prev_cum
        local_idx = offset_in_block % self.base_lens[dataset_idx]
        return self.datasets[dataset_idx][local_idx]


class TagDataset(Dataset):
    """Wrap a dataset and attach a source tag to each sample."""

    def __init__(self, dataset: Dataset, tag: str):
        """Initialize the tagged dataset wrapper.

        Args:
            dataset: Underlying dataset.
            tag: Source tag attached to each sample under the "src" key.
        """
        self.dataset = dataset
        self.tag = tag

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.dataset)

    def __getitem__(self, i):
        """Return a sample with the source tag attached.

        Args:
            i: Sample index.

        Returns:
            The dataset sample with an added "src" field.
        """
        sample = self.dataset[i]
        sample["src"] = self.tag
        return sample
