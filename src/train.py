"""
Model for training and evaluating a PyTorch model with enhanced logging and plotting.
"""

import time
import json # To save metrics dictionary
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.utils.data import DataLoader, TensorDataset # Added for example usage

import matplotlib.pyplot as plt
import numpy as np # Needed for metric calculation inputs
from sklearn.metrics import classification_report # For detailed metrics

class ModelTrainer:
    """
    Class to handle the training and evaluation of a model,
    including detailed metric reporting and plotting.
    """

    DEFAULT_LOG_INTERVAL: int = 5

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ) -> None:
        """
        Initializes the ModelTrainer with the model, data loaders, optimizer, and loss function.
        """
        
        self.model = model.to(device) # Ensure model is on the correct device
        self.device = device
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.criterion = criterion

        # For tracking training progress over epochs
        self.history = {
            'epoch_train_losses': [],
            'epoch_test_losses': [],
            'epoch_accuracies': [],
            'all_batch_losses': [] # Store losses for every batch across all epochs
        }


    def train_epoch(
        self,
        epoch: int,
        log_interval: int = DEFAULT_LOG_INTERVAL,
        max_batches: int | None = None
    ) -> tuple[list[float], float]:
        """
        Trains the model for one epoch.

        Returns:
            tuple[list[float], float]: List of losses for each batch in the epoch,
                                       Average loss for the epoch.
        """
        
        self.model.train()
        running_loss = 0.0
        samples_processed = 0
        start_time = time.time()

        batch_losses = []
        num_batches_to_run = len(self.train_loader) if max_batches is None else min(max_batches, len(self.train_loader))

        for batch_idx, (data, target) in enumerate(self.train_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            current_loss = loss.item()
            batch_losses.append(current_loss)

            batch_size = data.size(0)
            running_loss += current_loss * batch_size
            samples_processed += batch_size

            if batch_idx % log_interval == 0 and batch_idx > 0:
                elapsed_time = time.time() - start_time
                samples_so_far = (batch_idx + 1) * batch_size # Approximation if batch sizes vary
                total_samples = len(self.train_loader.dataset) if max_batches is None else num_batches_to_run * batch_size
                speed = samples_so_far / elapsed_time if elapsed_time > 0 else float('inf')
                print(f'Train Epoch: {epoch} [{samples_so_far}/{total_samples} '
                      f'({100. * (batch_idx + 1) / num_batches_to_run:.0f}%)]\t'
                      f'Loss: {current_loss:.6f}\tSpeed: {speed:.2f} samples/sec')

        epoch_loss = running_loss / samples_processed if samples_processed > 0 else 0.0
        epoch_time = time.time() - start_time
        print(f'====> Epoch: {epoch} Average loss: {epoch_loss:.4f}, Time: {epoch_time:.2f}s')

        # Store batch losses for the detailed plot later
        self.history['all_batch_losses'].extend(batch_losses)

        return batch_losses, epoch_loss


    def evaluate(
        self,
        max_batches: int | None = None
    ) -> tuple[float, float, dict]:
        """
        Evaluates the model on the test set and calculates detailed metrics.

        Returns:
            tuple[float, float, dict]: Accuracy, Average test loss, Classification report dictionary.
        """
        self.model.eval()
        test_loss = 0
        correct = 0
        all_targets = []
        all_preds = []
        num_test_samples = 0

        num_batches_to_run = len(self.test_loader) if max_batches is None else min(max_batches, len(self.test_loader))

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.test_loader):
                if max_batches is not None and batch_idx >= max_batches:
                    break

                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += self.criterion(output, target).item() * data.size(0)  # sum up batch loss weighted by batch size
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability

                correct += pred.eq(target.view_as(pred)).sum().item()
                num_test_samples += data.size(0)

                # Store targets and predictions for detailed metrics
                all_targets.extend(target.view_as(pred).cpu().numpy())
                all_preds.extend(pred.cpu().numpy())

        test_loss /= num_test_samples if num_test_samples > 0 else 1.0
        accuracy = 100. * correct / num_test_samples if num_test_samples > 0 else 0.0

        print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{num_test_samples} ({accuracy:.2f}%)')

        # Calculate detailed metrics if we have predictions
        metrics_report = {}
        if num_test_samples > 0 and len(all_targets) > 0:
             # Get unique labels present in targets, handling potential unseen classes during limited evaluation
            unique_labels = np.unique(all_targets + all_preds)
            target_names = [f"Class {i}" for i in unique_labels] # Basic target names

            # Ensure target_names aligns with unique_labels if you need specific names
            # If your targets are 0, 1, 2, target_names=['Class 0', 'Class 1', 'Class 2']
            # Use zero_division=0 to avoid warnings when a class has no predictions/support
            try:
                metrics_report = classification_report(
                    all_targets,
                    all_preds,
                    labels=unique_labels, # Explicitly provide labels seen
                    target_names=target_names,
                    output_dict=True,
                    zero_division=0
                )
                print("Classification Report:")
                # Print a formatted string version for quick console view
                print(classification_report(
                    all_targets,
                    all_preds,
                    labels=unique_labels,
                    target_names=target_names,
                    zero_division=0
                ))
            except ValueError as e:
                print(f"Could not generate classification report: {e}")
                print("Targets:", np.unique(all_targets))
                print("Predictions:", np.unique(all_preds))
                metrics_report = {"error": str(e)}


        print("-" * 30) # Separator

        return accuracy, test_loss, metrics_report


    def plot_training_curves(self, save_path: Path | None = None):
        """ Plots and optionally saves the training curves. """
        epochs_range = range(1, len(self.history['epoch_train_losses']) + 1)

        # --- Plot 1: Batch Losses ---
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1) # Create subplot 1

        # Calculate epoch boundaries for average loss plotting
        batches_per_epoch = len(self.history['all_batch_losses']) // len(self.history['epoch_train_losses']) if len(self.history['epoch_train_losses']) > 0 else 1
        avg_loss_indices = [i * batches_per_epoch + batches_per_epoch // 2 for i in range(len(self.history['epoch_train_losses']))]
        # Ensure indices are within bounds
        avg_loss_indices = [min(idx, len(self.history['all_batch_losses'])-1) for idx in avg_loss_indices]


        plt.plot(self.history['all_batch_losses'], label='Batch Loss', alpha=0.7)
        if avg_loss_indices and self.history['epoch_train_losses']: # Ensure we have data
             plt.plot(avg_loss_indices, self.history['epoch_train_losses'], 'r-o', label='Avg Epoch Train Loss', markersize=4)

        plt.xlabel("Batch Number")
        plt.ylabel("Loss")
        plt.title("Training Loss per Batch")
        plt.legend()
        plt.grid(True)


        # --- Plot 2: Epoch Metrics ---
        plt.subplot(1, 2, 2) # Create subplot 2
        ax1 = plt.gca() # Get current axes (left y-axis)

        p1, = ax1.plot(epochs_range, self.history['epoch_train_losses'], 'b-o', label='Train Loss')
        p2, = ax1.plot(epochs_range, self.history['epoch_test_losses'], 'r-o', label='Test Loss')
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("Loss", color='black')
        ax1.tick_params(axis='y', labelcolor='black')
        ax1.grid(True)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        p3, = ax2.plot(epochs_range, self.history['epoch_accuracies'], 'g-s', label='Test Accuracy')
        ax2.set_ylabel('Accuracy (%)', color='g')  # we already handled the x-label with ax1
        ax2.tick_params(axis='y', labelcolor='g')
        ax2.set_ylim(0, 105) # Accuracy typically 0-100

        plt.title("Epoch Metrics: Loss & Accuracy")
        # Combine legends from both axes
        lines = [p1, p2, p3]
        ax1.legend(lines, [l.get_label() for l in lines], loc='center right')

        plt.tight_layout() # Adjust layout to prevent overlap

        if save_path is not None:
            plot_filename = save_path / "training_curves.pdf"
            plt.savefig(plot_filename, dpi=300)
            print(f"Training curves plot saved to {plot_filename}")
            plt.close() # Close the figure after saving to avoid displaying it
        else:
            plt.show()


    def train(
        self,
        epochs: int,
        max_train_batches: int | None = None,
        max_test_batches: int | None = None,
        log_interval: int = DEFAULT_LOG_INTERVAL,
        save_path: Path | None = None,
    ) -> nn.Module:
        """
        Runs the training loop for the specified number of epochs.
        Saves the model, metrics, and plots if save_path is provided.
        """

        if save_path is not None:
            save_path.mkdir(parents=True, exist_ok=True)
            print(f"Results will be saved to: {save_path}")

        start_total_time = time.time()

        for epoch in range(1, epochs + 1):
            # Train for one epoch
            _, avg_train_loss = self.train_epoch(
                epoch,
                log_interval=log_interval,
                max_batches=max_train_batches
            )

            # Evaluate on the test set
            accuracy, avg_test_loss, metrics_report = self.evaluate(
                max_batches=max_test_batches
            )

            # Store epoch results
            self.history['epoch_train_losses'].append(avg_train_loss)
            self.history['epoch_test_losses'].append(avg_test_loss)
            self.history['epoch_accuracies'].append(accuracy)

            # Save model and metrics if path is provided
            if save_path is not None:
                # Save model state dictionary
                model_save_path = save_path / f"model_epoch_{epoch}.pth"
                torch.save(self.model.state_dict(), model_save_path)
                print(f"Model saved to {model_save_path}")

                # Save metrics report as JSON
                metrics_save_path = save_path / f"metrics_epoch_{epoch}.json"
                try:
                    with open(metrics_save_path, 'w') as f:
                        json.dump(metrics_report, f, indent=4)
                    print(f"Metrics report saved to {metrics_save_path}")
                except Exception as e:
                     print(f"Could not save metrics report for epoch {epoch}: {e}")


        end_total_time = time.time()
        print(f"\nTraining finished. Total time: {end_total_time - start_total_time:.2f}s")

        # Plot final curves after all epochs are done
        self.plot_training_curves(save_path)

        return self.model
