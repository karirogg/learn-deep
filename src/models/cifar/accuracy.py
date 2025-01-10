import torch

def accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    # mask output to only consider the classes present in the target
    # output_masked = -torch.inf * torch.ones_like(output)
    # output_masked[:, available_targets] = output[:, available_targets]
    acc = (output.argmax(dim=1) == target).float()

    return acc.sum().item(), acc
