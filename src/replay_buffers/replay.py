import torch
import numpy as np
import pdb


class Replay:

    def __init__(self, params=None):
        self.params = params

    def uniform(self, model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, X_list: list[torch.tensor], y_list: list[torch.tensor], metrics=None, samples_to_add: int = 10000):
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
    
    def simple_sorted(self, model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, X_list: list[torch.tensor], y_list: list[torch.tensor], metrics, samples_to_add: int = 10000, vog_weight=0.5):
        print("populating replay buffer...", end= " ")
        # collect inputs
        input_images, labels = map(torch.cat, zip(*[(img, labels) for img, labels, _ in dataloader]))
        input_images = input_images.detach().to(torch.float32)
        # select indices
        sorted_idcs = sorted(torch.arange(labels.shape[0]), key=lambda i : vog_weight * metrics["vog"][i] + (1-vog_weight) * metrics["learning_speeds"][i])
        lower_boundary = self.params["remove_lower_percent"] * len(sorted_idcs) // 100
        upper_boundary = len(sorted_idcs) - self.params["remove_upper_percent"] * len(sorted_idcs) // 100
        filtered_idcs = sorted_idcs[lower_boundary:upper_boundary]
        selected_idcs = np.random.choice(filtered_idcs, size=min(int(samples_to_add), len(filtered_idcs)), replace=False)
        # return in correct format (list of tensors of shape [batch_size, :, :, :] and [batch_size])
        for i in range(0, len(selected_idcs), dataloader.batch_size):
            idcs = selected_idcs[i : i+dataloader.batch_size]
            X_list.append(input_images[idcs])
            y_list.append(labels[idcs])
        print("done")
        return X_list, y_list
