"""
Module for MNIST classification architecture specification using 
the regular MLP model.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MNISTMLP(nn.Module):
    """
    A simple CNN model for MNIST classification using the KANConv2d layer.
    """
    
    def __init__(self) -> None:
        """
        Initializes the KANConv2d model for MNIST classification.
        """
        
        super(MNISTMLP, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu4 = nn.ReLU()
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2, stride=1, padding=1)
        self.relu5 = nn.ReLU()
        self.global_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(128, 10)  # Adjusted for 2 max pooling layers


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the forward pass of the model.
        """
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool1(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.maxpool2(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.maxpool3(x)
        x = self.conv5(x)
        x = self.relu5(x)
        x = self.global_pooling(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        
        return x