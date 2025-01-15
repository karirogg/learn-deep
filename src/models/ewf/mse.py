import torch

def mse(output: torch.Tensor, target: torch.Tensor) -> float:
    '''
    Calculate the mean squared error between the output and the target.
    '''
    mse = (output.reshape(-1) - target) ** 2

    return mse.sum().item(), mse