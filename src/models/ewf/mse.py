import torch

def mse(output: torch.Tensor, target: torch.Tensor) -> float:
    mse = (output.reshape(-1) - target) ** 2

    return mse.sum().item(), mse