import torch
import pickle
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import torchvision


class TaskDataset(Dataset):
    def __init__(self, data, labels, transform, device):
        self.data = data
        self.labels = labels
        self.transform = transform
        self.device = device

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]

        image = self.transform(image)

        return image.to(self.device), label.to(self.device), idx


def preprocess_cifar(num_classes: int, n: int, batch_size: int, device: torch.device):
    train_tasks = []
    test_tasks = []

    # Define transformations
    if num_classes == 10:
        MEAN, STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
    elif num_classes == 100:
        MEAN, STD = (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)
    else:
        raise ValueError("Invalid number of classes, expected 10 or 100.")

    train_transforms = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    test_transforms = transforms.Compose(
        [
            transforms.Normalize(MEAN, STD),
        ]
    )

    for i in range(1, n + 1):
        with open(f"../data/cifar-{num_classes}-{n}/train/task_{i}", "rb") as f:
            task_train, task_labels = pickle.load(f)

        with open(f"../data/cifar-{num_classes}-{n}/test/task_{i}", "rb") as f:
            task_test, task_test_labels = pickle.load(f)

        task_train_tensor = torch.tensor(task_train, dtype=torch.float32).div(255.0)
        task_test_tensor = torch.tensor(task_test, dtype=torch.float32).div(255.0)

        # Create training and testing datasets
        train_dataset = TaskDataset(
            task_train_tensor,
            torch.tensor(task_labels, dtype=torch.long),
            transform=train_transforms,
            device=device,
        )
        test_dataset = TaskDataset(
            task_test_tensor,
            torch.tensor(task_test_labels, dtype=torch.long),
            transform=test_transforms,
            device=device,
        )

        # Create DataLoaders
        train_tasks.append(
            DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        )
        test_tasks.append(
            DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        )

        # Verify that transformations are correctly applied
        # def imshow(img, mean, std):
        #     mean = torch.tensor(mean).view(3, 1, 1)
        #     std = torch.tensor(std).view(3, 1, 1)
        #     img = img * std + mean
        #     npimg = img.numpy()
        #     plt.imshow(np.transpose(npimg, (1, 2, 0)))
        #     plt.show()

        # dataiter = iter(DataLoader(train_dataset, batch_size=batch_size, shuffle=True))
        # images, labels, _ = next(dataiter)
        # imshow(torchvision.utils.make_grid(images[:4]), MEAN, STD)

    return train_tasks, test_tasks
