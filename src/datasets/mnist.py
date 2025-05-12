"""
Module for loading the MNIST dataset.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torchvision
import torchvision.transforms as transforms


DEFAULT_BATCH_SIZE = 64
    
def prepare_mnist(
    batch_size: int = DEFAULT_BATCH_SIZE,
    test_batch_size: int = DEFAULT_BATCH_SIZE,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Loads the MNIST dataset and returns the train and test data loaders.
    """
    
    # MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    try:
        train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    except Exception as e:
        print(f"Failed to download MNIST: {e}. Trying to load from local './data' directory.")
        try:
            train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=False, transform=transform)
            test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=False, transform=transform)
        except Exception as e_local:
            print(f"Failed to load MNIST from local directory: {e_local}. Exiting.")
            return

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True if device=="cuda" else False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=2, pin_memory=True if device=="cuda" else False)

    return train_loader, test_loader