from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader, default_collate
from torchvision import datasets, transforms
from torchvision.transforms import v2 as transforms_v2


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(
    image_size: int = 224,
    use_augmentation: bool = True,
    randaugment: bool = False,
    randaugment_num_ops: int = 2,
    randaugment_magnitude: int = 9,
    random_erasing: float = 0.0,
) -> Tuple[transforms.Compose, transforms.Compose]:
    """Return train and eval transforms."""
    if use_augmentation:
        if randaugment:
            aug_transforms = [
                transforms.Resize(256),
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandAugment(
                    num_ops=randaugment_num_ops,
                    magnitude=randaugment_magnitude,
                ),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        else:
            aug_transforms = [
                transforms.Resize(256),
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02
                ),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        if random_erasing > 0:
            aug_transforms.append(transforms.RandomErasing(p=random_erasing))
        train_tf = transforms.Compose(aug_transforms)
    else:
        train_tf = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

    eval_tf = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_tf, eval_tf


def build_mixup_cutmix_collate(
    num_classes: int,
    mixup_alpha: float = 0.2,
    cutmix_alpha: float = 1.0,
) -> Callable:
    """Return a collate function that applies MixUp/CutMix randomly."""
    mix = transforms_v2.RandomChoice(
        [
            transforms_v2.MixUp(alpha=mixup_alpha, num_classes=num_classes),
            transforms_v2.CutMix(alpha=cutmix_alpha, num_classes=num_classes),
        ]
    )

    def collate_fn(batch):
        return mix(default_collate(batch))

    return collate_fn


def class_distribution(dataset: datasets.ImageFolder) -> Dict[int, int]:
    targets = dataset.targets
    counts = Counter(targets)
    return dict(sorted(counts.items(), key=lambda item: item[0]))


def build_dataloaders(
    data_dir: str | Path,
    batch_size: int = 32,
    num_workers: int = 0,
    image_size: int = 224,
    use_augmentation: bool = True,
    collate_fn: Optional[Callable] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, datasets.ImageFolder, datasets.ImageFolder, datasets.ImageFolder]:
    base = Path(data_dir)
    train_tf, eval_tf = build_transforms(
        image_size=image_size, use_augmentation=use_augmentation
    )

    train_ds = datasets.ImageFolder(base / "train", transform=train_tf)
    valid_ds = datasets.ImageFolder(base / "valid", transform=eval_tf)
    test_ds = datasets.ImageFolder(base / "test", transform=eval_tf)

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, valid_loader, test_loader, train_ds, valid_ds, test_ds

