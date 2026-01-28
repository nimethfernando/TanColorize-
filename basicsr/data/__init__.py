# Import all datasets to register them
from .lab_dataset import LabDataset

from basicsr.utils.registry import DATASET_REGISTRY
from torch.utils.data import DataLoader
import torch


def build_dataset(dataset_opt):
    """Build dataset from registry.

    Args:
        dataset_opt (dict): Dataset configuration. It must contain the key "type".

    Returns:
        Dataset: The built dataset.
    """
    dataset_type = dataset_opt.pop('type')
    dataset = DATASET_REGISTRY.get(dataset_type)(dataset_opt)
    return dataset


def build_dataloader(dataset, dataset_opt, num_gpu=1, dist=False, sampler=None, seed=None):
    """Build dataloader.

    Args:
        dataset (torch.utils.data.Dataset): Dataset.
        dataset_opt (dict): Dataset options. It contains the following keys:
            phase (str): 'train' or 'val'.
            num_worker_per_gpu (int): Number of workers for each GPU.
            batch_size_per_gpu (int): Training batch size for each GPU.
            pin_memory (bool): Whether to use pin_memory.
        num_gpu (int): Number of GPUs. Used only in the train phase.
            Default: 1.
        dist (bool): Whether in distributed training. Default: False.
        sampler (torch.utils.data.sampler): Data sampler. Default: None.
        seed (int | None): Seed. Default: None

    Returns:
        torch.utils.data.DataLoader: DataLoader.
    """
    phase = dataset_opt['phase']
    rank = 0
    world_size = 1
    if dist:
        from basicsr.utils.dist_util import get_dist_info
        rank, world_size = get_dist_info()

    if phase == 'train':
        num_workers = dataset_opt.get('num_worker_per_gpu', 0)
        batch_size = dataset_opt['batch_size_per_gpu']
        pin_memory = dataset_opt.get('pin_memory', False)
    else:  # val
        num_workers = dataset_opt.get('num_worker_per_gpu', 0)
        batch_size = dataset_opt.get('batch_size_per_gpu', 1)
        pin_memory = dataset_opt.get('pin_memory', False)

    if sampler is not None:
        shuffle = False
    else:
        shuffle = dataset_opt.get('use_shuffle', False)

    if dist:
        batch_size = batch_size
        num_workers = num_workers

    dataloader_args = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        sampler=sampler,
        drop_last=False,
        pin_memory=pin_memory)

    if sampler is None:
        dataloader_args['shuffle'] = shuffle
    else:
        dataloader_args['shuffle'] = False

    return DataLoader(**dataloader_args)
