"""
Module for loading the CIFAR-10 dataset.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torchvision
import torchvision.transforms as transforms


DEFAULT_BATCH_SIZE = 64

def prepare_cifar10(
    batch_size: int = DEFAULT_BATCH_SIZE,
    test_batch_size: int = DEFAULT_BATCH_SIZE,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Loads the CIFAR-10 dataset and returns the train and test data loaders.
    """
    
    # CIFAR-10 transform: normalize to mean/std per channel
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465), 
            std=(0.2023, 0.1994, 0.2010)
        )
    ])
    
    try:
        train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    except Exception as e:
        print(f"Failed to download CIFAR-10: {e}. Trying to load from local './data' directory.")
        try:
            train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
            test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
        except Exception as e_local:
            print(f"Failed to load CIFAR-10 from local directory: {e_local}. Exiting.")
            return

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=(device.type == "cuda"))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=2, pin_memory=(device.type == "cuda"))

    return train_loader, test_loader
