import torch

def uniform_replay_buffer(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, X_list: list[torch.tensor], y_list: list[torch.tensor], samples_to_add: int = 10000):
    """
    A simple replay buffer that samples uniformly from a dataloader.
    """

    num_samples_added = 0

    # Assuming that the dataloader is shuffled, this is a uniform sampling strategy
    for X, y, _ in dataloader:
        # detach the tensors since otherwise they keep their gradients
        new_replay_inputs = X.detach().to(torch.float32)

        X_list.append(new_replay_inputs)
        y_list.append(y.detach())

        num_samples_added += len(y)

        if num_samples_added >= samples_to_add:
            break

    return X_list, y_list

