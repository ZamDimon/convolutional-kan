"""
Module for MNIST classification architecture specification using KANConv2d
layer.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from kan import *

from src.layers.ckan_layer import KANConv2d


class MNISTMixedCKAN(nn.Module):
    """
    A simple CNN model for MNIST classification using the KANConv2d layer.
    """
    
    def __init__(self) -> None:
        """
        Initializes the KANConv2d model for MNIST classification.
        """
        
        super(MNISTMixedCKAN, self).__init__()
        
        self.conv1 = KANConv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1, spline_points=3, spline_degree=3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = KANConv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1, spline_points=5, spline_degree=3)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = KANConv2d(in_channels=16, out_channels=24, kernel_size=3, stride=1, padding=1, spline_points=5, spline_degree=3)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(8*8*24, 10)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the forward pass of the model.
        """
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.fc(x)
        
        return x