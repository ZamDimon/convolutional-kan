"""
Main module for running the KANConv2d model on MNIST dataset.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from src.models.fully_kan_mnist import MNISTFullyCKAN
from src.models.mlp_mnist import MNISTMLP

from src.datasets.mnist import prepare_mnist
from src.datasets.cifar10 import prepare_cifar10
from src.train import ModelTrainer
from src.visualize import ConvolutionVisualizer


def main() -> None:
    """
    Main script that sets up the MNIST dataset, model, optimizer, and loss function.
    It trains the model and saves the trained model's state_dict.
    """
    
    # Hyperparameters
    LEARNING_RATE = 0.001 # For Adam
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Instantiate the model, dataset optimizer, and loss function
    model = MNISTFullyCKAN().to(device)
    #model.load_state_dict(torch.load('models/mlp-mnist/model_epoch_10.pth', weights_only=True))
    train, test = prepare_cifar10(batch_size=32, test_batch_size=32, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    trainer = ModelTrainer(
        model=model,
        train_loader=train,
        test_loader=test,
        optimizer=optimizer,
        criterion=criterion,
        device=device
    )
    trainer.train(epochs=10, 
                  save_path=Path('./models/mlp-mnist'),
                  max_train_batches=50,
                  max_test_batches=5,)
    
    visualizer = ConvolutionVisualizer(model=model, test_dataset=test)
    save_path = Path("./images")
    visualizer.visualize(save_path=save_path, layers=["conv1", "conv2", "conv3", "conv4"])
    
    
if __name__ == '__main__':
    main()