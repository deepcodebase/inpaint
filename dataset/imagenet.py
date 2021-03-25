from pathlib import Path
from typing import Any, Callable, Optional

import torch
from torchvision import datasets, transforms
from pytorch_lightning import LightningDataModule


class ImageNetDataModule(LightningDataModule):

    def __init__(
            self, data_dir: str = "path/to/dir", batch_size: int = 32,
            num_workers: int = 6, shuffle: bool = True,
            pin_memory: bool = False):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory

    def setup(self, stage: Optional[str] = None):
        self.train_dir = Path(self.data_dir) / 'train'
        self.val_dir = Path(self.data_dir) / 'val'
        if not self.train_dir.is_dir() or not self.val_dir.is_dir():
            raise FileNotFoundError(
                """
            Imagenet is no longer automatically downloaded by PyTorch.
            To get imagenet:
            download yourself from http://www.image-net.org/challenges/LSVRC/2012/downloads
            """
            )

    @property
    def imagenet_normalization(self):
        return transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    @property
    def train_transform(self):
        return transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            self.imagenet_normalization,
        ])
    
    @property
    def val_transform(self):
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            self.imagenet_normalization,
        ])

    def train_dataloader(self):
        train_dataset = datasets.ImageFolder(
            self.train_dir, self.train_transform)
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
        return train_loader

    def val_dataloader(self):
        val_dataset = datasets.ImageFolder(self.val_dir, self.val_transform)
        val_loader = torch.utils.data.DataLoader(
            dataset=val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
        return val_loader

    def test_dataloader(self):
        return self.val_dataloader()
