from pathlib import Path

import albumentations as albu
import cv2
import pytorch_lightning as pl
import torch

from torch.utils.data import DataLoader

from src.datasets.text_zoom import ConcatDataset, TextZoomDataset
from src.utils.config_reader import Config


def denormalize(tensors, means, stds, max_value=1.0):
    """ Denormalizes image tensors using mean and std """
    if not isinstance(means, torch.Tensor):
        means = torch.Tensor(means).type_as(tensors) * max_value

    if not isinstance(stds, torch.Tensor):
        stds = torch.Tensor(stds).type_as(tensors) * max_value

    for c in range(3):
        tensors[:, c].mul_(max_value).mul_(stds[c]).add_(means[c])

    return torch.clamp(tensors, 0, max_value)


class LitDataModule(pl.LightningDataModule):
    def __init__(self, config:Config):
        self._cfg = config
        self._root_dir = Path(self._cfg.dataset_dir)
        self._hr_img_size = config.hr_img_size[::-1] # hxw to wxh
        self._train_dst_dirs = [p for p in self._root_dir.glob('train*') if p.is_dir()]
        self._val_dst_dir = self._root_dir / 'test' / self._cfg.val_split_complexity
        self._test_dst_dir = self._root_dir / 'test' / self._cfg.test_split_complexity
        self._train_dst, self._val_dst, self._test_dst = None, None, None

        width, height = self._hr_img_size
        self._lr_transforms = albu.Compose([
            albu.Resize(height // config.scale_factor, width // config.scale_factor, cv2.INTER_CUBIC),
            albu.ToFloat(),
            albu.Normalize(config.norm_means, config.norm_stds),
        ])
        self._hr_transforms = albu.Compose([
            albu.Resize(height, width, cv2.INTER_CUBIC),
            albu.ToFloat(),
            albu.Normalize(config.norm_means, config.norm_stds),
        ])

    def setup(self, stage):
        # called on every process in DDP
        if stage is None or stage == 'fit':
            train_splits = [TextZoomDataset(p, self._lr_transforms, self._hr_transforms) for p in self._train_dst_dirs]
            self._train_dst = ConcatDataset(train_splits)
            self._val_dst = TextZoomDataset(self._val_dst_dir, self._lr_transforms, self._hr_transforms)

        if stage == 'test':
            self._test_dst = TextZoomDataset(self._test_dst_dir, self._lr_transforms, self._hr_transforms)

    def train_dataloader(self):
        return DataLoader(self._train_dst, 
                          self._cfg.batch_size, 
                          self._cfg.shuffle, 
                          num_workers=self._cfg.num_workers,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self._val_dst, 
                          self._cfg.batch_size, 
                          self._cfg.shuffle, 
                          num_workers=self._cfg.num_workers,
                          pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self._test_dst, 
                          self._cfg.batch_size, 
                          shuffle=False, 
                          num_workers=self._cfg.num_workers,
                          pin_memory=True)