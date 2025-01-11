import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import pdb

class Replay:

    def __init__(self, params=None, strategy=None, batch_size=8, samples_to_add=10000, num_tasks=2, weights={}):
        self.params = params

        self.X_list = []
        self.task_list = []
        self.y_list = []

        self.buffer = None
        self.weights = self.normalize_weights(weights)
        
        if strategy == "uniform":
            self.strategy = self.uniform
        elif strategy == "simple_sorted":
            self.strategy = self.simple_sorted
        else:
            self.strategy = None

        self.batch_size = batch_size
        self.num_tasks = num_tasks
        self.samples_to_add = samples_to_add

    def reset(self):
        if len(self.X_list) > 0:
            self.buffer = iter(DataLoader(TensorDataset(torch.cat(self.X_list, dim=0), torch.cat(self.task_list, dim=0), torch.cat(self.y_list, dim=0)), batch_size=self.batch_size, shuffle=True))

    def normalize_weights(self, weights):
        print("normalizing weights...", end="")
        weight_sum = sum([v for v in weights.values()])
        for key, value in weights.items():
            weights[key] = value / weight_sum
        print("done | new weights:", weights)
        return weights

    def sample(self):
        inputs = None
        labels = None
        task_ids = None

        out = []
        for i in range(self.num_tasks):
            out.append((None, None))

        if self.buffer is None:
            return out

        try:
            inputs, task_ids, labels = next(self.buffer)
        except StopIteration:
            self.reset()
            inputs, task_ids, labels = next(self.buffer)

        for i in range(self.num_tasks):
            out[i] = (inputs[task_ids == i], labels[task_ids == i])

        return out

    def uniform(self, model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, task_id: int, metrics=None):
        """
        A simple replay buffer that samples uniformly from a dataloader.
        """

        num_samples_added = 0

        # Assuming that the dataloader is shuffled, this is a uniform sampling strategy
        for X, y, _ in dataloader:
            # detach the tensors since otherwise they keep their gradients
            new_replay_inputs = X.detach().to(torch.float32)

            if num_samples_added + len(y) > self.samples_to_add:
                num_samples_to_add = self.samples_to_add - num_samples_added
                new_replay_inputs = new_replay_inputs[:num_samples_to_add]
                y = y[:num_samples_to_add]

            self.X_list.append(new_replay_inputs)
            self.task_list.append(torch.full((len(y),), task_id, dtype=torch.long))
            self.y_list.append(y.detach())

            num_samples_added += len(y)

            if num_samples_added >= self.samples_to_add:
                break    

    def simple_sorted(self, model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, task_id: int, metrics):
        print("populating replay buffer...", end= " ")

        # collect inputs
        unsorted_input_images, unsorted_labels, idx_list = map(torch.cat, zip(*[(img, labels, idcs) for img, labels, idcs in dataloader]))
        unsorted_input_images = unsorted_input_images.detach().to(torch.float32)
        input_images, labels = torch.zeros_like(unsorted_input_images), torch.zeros_like(unsorted_labels)
        for i, idx in enumerate(idx_list):
            input_images[idx] = unsorted_input_images[i]
            labels[idx] = unsorted_labels[i]
        # sort by metrics
        idcs_sum = 0
        for metric, weight in self.weights.items():
            if metrics.get(metric, None) is None or metrics[metric].ndim == 0:
                continue
            # pdb.set_trace()
            if metric not in ["learning_speed", "vog", "variance_of_gradients"]:
                sorted_idcs = torch.Tensor(sorted(torch.arange(labels.shape[0]), key=lambda i : metrics[metric][i]))
            else:
                sorted_idcs = torch.Tensor(sorted(torch.arange(labels.shape[0]), key=lambda i : metrics[metric][i], reverse=True))
            idcs_sum += weight * sorted_idcs
        _, mapped_idcs = torch.unique(idcs_sum, sorted=True, return_inverse=True) # map indices back to the correct interval after summing 
        sorted_idcs = mapped_idcs.argsort()
        # select indices
        lower_boundary = self.params["remove_lower_percent"] * len(sorted_idcs) // 100
        upper_boundary = len(sorted_idcs) - self.params["remove_upper_percent"] * len(sorted_idcs) // 100
        filtered_idcs = sorted_idcs[lower_boundary:upper_boundary]
        selected_idcs = np.random.choice(filtered_idcs, size=min(int(self.samples_to_add), len(filtered_idcs)), replace=False)
        # return in correct format (list of tensors of shape [batch_size, :, :, :] and [batch_size])
        for i in range(0, len(selected_idcs), dataloader.batch_size):
            idcs = selected_idcs[i : i+dataloader.batch_size]
            self.X_list.append(input_images[idcs])
            self.task_list.append(torch.full((len(idcs),), task_id, dtype=torch.long))
            self.y_list.append(labels[idcs])
        print("done")
        return
        
    # def validated(self)
