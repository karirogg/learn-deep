import torch
import pickle
from torch.utils.data import DataLoader, TensorDataset

def preprocess_cifar(num_classes: int, n: int, batch_size: int, device: torch.device):
    train_tasks = []
    test_tasks = []

    for i in range(1, n + 1):
        with open(f"../data/cifar-{num_classes}-{n}/train/task_{i}", "rb") as f:
            task_train, task_labels = pickle.load(f)

        with open(f"../data/cifar-{num_classes}-{n}/test/task_{i}", "rb") as f:
            task_test, task_test_labels = pickle.load(f)

        task_train_tensor = torch.tensor(task_train, dtype=torch.float32)
        task_test_tensor = torch.tensor(task_test, dtype=torch.float32)

        task_train_tensor = task_train_tensor / 255.0 
        task_test_tensor = task_test_tensor / 255.0 

        mean = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32).view(1, 3, 1, 1)

        task_train_tensor = (task_train_tensor - mean) / std
        task_test_tensor = (task_test_tensor - mean) / std

        train_tasks.append(
            DataLoader(
                TensorDataset(
                    task_train_tensor,
                    torch.tensor(task_labels, dtype=torch.long, device=device),
                    torch.arange(len(task_labels), device=device), # this is to keep track of the order of the samples
                ),
                batch_size=batch_size,
                shuffle=True,
            )
        )
        test_tasks.append(
            DataLoader(
                TensorDataset(
                    task_test_tensor,
                    torch.tensor(task_test_labels, dtype=torch.long, device=device),
                    torch.arange(len(task_test_labels), device=device),
                ),
                batch_size=batch_size,
                shuffle=False,
            )
        )

    return train_tasks, test_tasks
