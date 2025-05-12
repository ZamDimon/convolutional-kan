import torch.nn as nn
from collections import OrderedDict # To store layer info nicely

def print_model_summary(model: nn.Module):
    """
    Prints a summary of the model's layers and parameter counts.

    Args:
        model: The PyTorch model (instance of nn.Module).
    """
    
    print("-" * 60)
    print(f"Model Summary: {model.__class__.__name__}")
    print("-" * 60)
    print(f"{'Layer (type)':<30} {'Output Shape':<20} {'Param #':<10}")
    print("=" * 60)

    total_params = 0
    layer_summary = OrderedDict() # Store info to print nicely

    # Use named_modules to iterate through layers including nested ones
    for name, module in model.named_modules():
        # Skip the top-level container module itself if it's the only one
        if name == "" and len(list(model.children())) > 0:
            continue

        # Calculate parameters directly in this module (not sub-modules)
        # Only count parameters that require gradients (trainable)
        num_params = sum(p.numel() for p in module.parameters(recurse=False) if p.requires_grad)
        total_params += num_params

        # Store layer info - Getting output shape requires a forward pass,
        # which is complex without input data. We'll omit it for this generic snippet.
        # A common practice is to show the module's string representation.
        layer_info = {
            "name": name if name else model.__class__.__name__, # Use class name for top level
            "type": module.__class__.__name__,
            "params": num_params
        }

        # Only print layers that have parameters or are basic building blocks
        # Avoid printing container modules like Sequential if they don't add params themselves
        # Or print all modules if desired (comment out the 'if' below)
        if num_params > 0 or not list(module.children()): # Print if it has params or is a leaf module
            print(f"{layer_info['name']} ({layer_info['type']})".ljust(50) + f"{layer_info['params']:,}".rjust(10))


    print("=" * 60)
    print(f"Total trainable parameters: {total_params:,}")
    print("-" * 60)