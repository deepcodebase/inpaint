import random
from pathlib import Path
from typing import Any, Callable, Optional, Dict

import torch
from hydra.utils import instantiate
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from pytorch_lightning import LightningDataModule

from .image import ImageDataset


def collate_fn(batch):
    batch = list(filter(lambda x: x[0] is not None and x[1] is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


class InpaintDataset(Dataset):

    def __init__(
            self, data: Dataset, mask: Dataset, random_mask=False):
        self.data = data
        self.mask = mask
        self.random_mask = random_mask
        self.n_mask = len(self.mask)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        if self.random_mask:
            mask_idx = random.randrange(self.n_mask)
        else:
            mask_idx = idx % self.n_mask
        mask = self.mask[mask_idx]
        if mask is not None:
            mask = mask[:1]
        return img, mask


class InpaintDataModule(LightningDataModule):

    def __init__(
            self, data: Dict[str, Any], mask: Dict[str, Any],
            height: int = 512, width: int = 512,
            batch_size: int = 32, num_workers: int = 6,
            pin_memory: bool = False):

        super().__init__()

        self.data = data
        self.mask = mask

        self.height = height
        self.width = width
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage: Optional[str] = None):
        pass

    def data_transform(self, split):
        if split == 'train':
            return transforms.Compose([
                transforms.Resize((self.height, self.width)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
        else:
            return transforms.Compose([
                transforms.Resize((self.height, self.width)),
                transforms.ToTensor(),
            ])

    def mask_transform(self, split):
        if split == 'train':
            return transforms.Compose([
                transforms.Resize((self.height, self.width)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(
                    degrees=20, scale=(0.5, 2.), translate=(0.2, 0.2), shear=20,
                    fill=255),
                transforms.ToTensor()
            ])
        else:
            return transforms.Compose([
                transforms.Resize((self.height, self.width)),
                transforms.ToTensor()
            ])

    def _dataloader(self, split):
        dataset = InpaintDataset(
            ImageDataset(
                self.data[f'{split}_dir'],
                transform=self.data_transform(split)),
            ImageDataset(
                self.mask[f'{split}_dir'],
                transform=self.mask_transform(split)),
            random_mask=(split == 'train')
        )
        loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=(split == 'train'),
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn
        )
        return loader

    def train_dataloader(self):
        return self._dataloader('train')

    def val_dataloader(self):
        return self._dataloader('val')

    def test_dataloader(self):
        return self._dataloader('test')
