import torch

def accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    """Compute the accuracy of the model."""
    acc = (output.argmax(dim=1) == target).float()

    return acc.sum().item(), acc
