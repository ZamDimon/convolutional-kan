"""
Module for visualizing intermediate outputs of the KANConv2d model.
"""

from __future__ import annotations

import os
import random
from typing import List
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils

class ConvolutionVisualizer:
    """
    Class to visualize the intermediate outputs of a convolutional neural network.
    """

    def __init__(
        self, 
        model: nn.Module, 
        test_dataset: torch.utils.data.Dataset
    ) -> None:
        """
        Initializes the ConvolutionVisualizer with the model and test dataset.
        """
        self.model = model
        self.test_dataset = test_dataset


    def visualize(self, save_path: Path, layers: List[str]) -> None:
        """
        Visualizes the intermediate outputs of the model.
        """
        
        if not save_path.exists():
            print(f"Creating directory: {save_path}...")
            os.makedirs(save_path, exist_ok=True)

        # Set model to evaluation mode
        self.model.eval()
        device = next(self.model.parameters()).device

        random_idx = random.randint(0, len(self.test_dataset) - 1)
        image, label = self.test_dataset.dataset[random_idx]
        image_unsqueezed = image.unsqueeze(0).to(device)

        print(f"Selected random image for inspection: Index {random_idx}, Label: {label}")

        activation_outputs = {}
        def get_activation(name: str):
            def hook(model, input, output):
                activation_outputs[name] = output.detach()
            return hook

        hooks = []
        for layer in layers:
            torch_layer = getattr(self.model, layer)
            hook = torch_layer.register_forward_hook(get_activation(layer))
            hooks.append(hook)

        with torch.no_grad():
            _ = self.model(image_unsqueezed)

        for hook in hooks:
            hook.remove()
            
        # --- Save/Visualize conv3 outputs ---
        for layer in layers:
            if layer in activation_outputs:
                layer_out = activation_outputs[layer].cpu()
                print(f"conv1 output shape: {layer_out.shape}")

                num_channels = layer_out.shape[1]
                for channel_idx in range(num_channels):
                    # Extract single channel image: shape (H, W)
                    channel_img = layer_out[0, channel_idx, :, :]
                    
                    # Reshape for F.interpolate: (1, 1, H, W)
                    # (Batch_size=1, Channels=1, Height, Width)
                    img_to_upscale = channel_img.unsqueeze(0).unsqueeze(0)
                    
                    # Upscale using nearest-neighbor interpolation
                    upscaled_img = F.interpolate(img_to_upscale, 
                                                size=(256, 256), 
                                                mode='nearest') # or 'nearest-exact' for newer PyTorch
                    
                    # Remove batch dimension for save_image if needed, it expects (C,H,W) or (B,C,H,W)
                    # upscaled_img is (1, 1, 256, 256), squeeze to (1, 256, 256) for save_image
                    os.makedirs(save_path / f"{layer}_channel_outputs", exist_ok=True)
                    vutils.save_image(upscaled_img.squeeze(0), 
                                    save_path / f"{layer}_channel_outputs/channel_{channel_idx+1}_256x256.png",
                                    normalize=True)
                    
                print(f"Saved {layer} upscaled (256x256) channel images from conv1 to 'conv1_channel_outputs'")
                
        # Save the original image
        vutils.save_image(image_unsqueezed.squeeze(0),
                        save_path / "original_image.png",
                        normalize=True)

