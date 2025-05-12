"""
Model for training and evaluating a PyTorch model.
"""

import time
from pathlib import Path

import torch

import matplotlib.pyplot as plt


class ModelTrainer:
    """
    Class to handle the training and evaluation of a model.
    """
    
    DEFAULT_LOG_INTERVAL: int = 5
    
    
    def __init__(
        self, 
        model: torch.nn.Module, 
        train_loader: torch.utils.data.DataLoader, 
        test_loader: torch.utils.data.DataLoader, 
        optimizer: torch.optim.Optimizer, 
        criterion: torch.nn.Module,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ) -> None:
        """
        Initializes the ModelTrainer with the model, data loaders, optimizer, and loss function.
        """
        
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.criterion = criterion
        
        # For training curve
        self.train_losses = []
        self.test_losses = []
        self.accuracies = []


    def train_epoch(
        self, 
        epoch, 
        log_interval=DEFAULT_LOG_INTERVAL,
        max_batches: int | None = None
    ) -> None:
        """
        Trains the model for one epoch.
        """
        
        self.model.train()
        running_loss = 0.0
        samples_processed = 0
        start_time = time.time()
        
        batch_losses = []
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break  # ✅ Exit early
            
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            batch_losses.append(loss.item())
            
            running_loss += loss.item() * data.size(0)
            samples_processed += data.size(0)
            
            if batch_idx % log_interval == 0 and batch_idx > 0:
                elapsed_time = time.time() - start_time
                speed = (batch_idx * self.train_loader.batch_size) / elapsed_time if elapsed_time > 0 else float('inf')
                total_batches = len(self.train_loader) if max_batches is None else max_batches
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(data) * total_batches} ({100. * batch_idx / total_batches:.0f}%)]\tLoss: {loss.item():.6f}\tSpeed: {speed:.2f} samples/sec')
        
        epoch_loss = running_loss / samples_processed
        epoch_time = time.time() - start_time
        print(f'====> Epoch: {epoch} Average loss: {epoch_loss:.4f}, Time: {epoch_time:.2f}s')
        
        return batch_losses, epoch_loss
            

    def evaluate(self, max_batches: int | None = None) -> None:
        """
        Evaluates the model on the test set.
        """
        
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            num_test_samples = 0
            
            for batch_idx, (data, target) in enumerate(self.test_loader):
                if max_batches is not None and batch_idx >= max_batches:
                    break
                
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += self.criterion(output, target).item() * data.size(0)  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                
                # Calculating statistics
                correct += pred.eq(target.view_as(pred)).sum().item()
                num_test_samples += len(data)

        test_loss /= num_test_samples
        accuracy = 100. * correct / num_test_samples

        print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{num_test_samples} ({accuracy:.2f}%)\n')
        
        return accuracy, test_loss


    def train(
        self, 
        epochs: int, 
        max_train_batches: int | None = None,
        max_test_batches: int | None = None,
        log_interval: int = DEFAULT_LOG_INTERVAL,
        save_path: Path | None = None,
    ) -> torch.nn.Module:
        """
        Runs the training loop for the specified number of epochs.
        """
        
        all_batch_losses = []
        average_losses = []
        
        for epoch in range(1, epochs + 1):
            batch_losses, avg_loss = self.train_epoch(epoch, 
                                                      log_interval=log_interval,
                                                      max_batches=max_train_batches)
            
            all_batch_losses.extend(batch_losses)
            average_losses.append(avg_loss)
            
            accuracy, test_loss = self.evaluate(max_batches=max_test_batches)
            
        # Plotting the batch losses
        plt.figure(figsize=(10, 5))
        plt.plot(all_batch_losses, label="Batch Loss", color='b')
        plt.plot(range(0, len(all_batch_losses), len(all_batch_losses) // len(average_losses)), average_losses, label="Average Loss", color='r', linestyle='dashed')
        plt.xlabel("Batch number")
        plt.ylabel("Loss")
        plt.title("Training Loss per Batch")
        plt.legend()
        plt.grid(True)
            
        if save_path is not None:
            save_path.mkdir(parents=True, exist_ok=True)
            torch.save(self.model.state_dict(), save_path / f"model_epoch_{epoch}.pth")
            plt.savefig(save_path / "batch_loss_curve.pdf", dpi=600)
            print(f"Model saved to {save_path / f'model_epoch_{epoch}.pth'}")
        else:
            plt.show()
        
        return self.model
        