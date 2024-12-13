import torch

def accuracy(output: torch.Tensor, target: torch.Tensor, available_targets: list[int]) -> float:
    # mask output to only consider the classes present in the target
    output_masked = -torch.inf * torch.ones_like(output)
    output_masked[:, available_targets] = output[:, available_targets]
    acc = (output_masked.argmax(dim=1) == target).float()
    return acc.mean().item(), acc