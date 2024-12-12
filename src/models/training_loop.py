import torch
from typing import Callable
import pickle
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from torch import nn
import argparse
import numpy as np
from torchvision import transforms

from models.custom_cnn import CIFAR_CNN

from metrics.vog import compute_VoG, visualize_VoG

def training_loop(
        train_tasks: list[torch.utils.data.DataLoader], 
        test_tasks: list[torch.utils.data.DataLoader], 
        model: torch.nn.Module, 
        optimizer: torch.optim.Optimizer, 
        criterion: torch.nn.Module, 
        metric: Callable[[float], float],
        evaluate: Callable[[torch.nn.Module, torch.utils.data.DataLoader, torch.nn.Module, Callable[[float], float]], tuple[float, float]],
        epochs_per_task: int,
        num_checkpoints : int
    ) -> list[float]:
    """
    The function trains the model on each of the different tasks sequentially using continual learning and uses a replay buffer to store the data from the previous tasks.
    """

    # Ignore replay buffer for now
    replay_buffer_X_list = []
    replay_buffer_y_list = []

    task_test_losses = []
    task_test_accuracies = []

    for _ in range(len(train_tasks)):
        task_test_losses.append([])
        task_test_accuracies.append([])

    for i, task in enumerate(train_tasks):
        # here the batch size is 16 = 64/4, so we should expect the replay buffer to include 4 times less data than the current task

        print(f'Training on task {i + 1}')
        grad_matrices = []
        checkpoints = np.linspace(0, epochs_per_task, num_checkpoints, endpoint=False, dtype=np.int32)
        # pdb.set_trace()
        # input_images = torch.cat([img for img, _ in task], dim=0)
        # epoch_labels = torch.cat([labels for _, labels in task], dim=0)
        input_images, epoch_labels = map(torch.cat, zip(*[(img, labels) for img, labels in task]))

        for epoch in tqdm(range(epochs_per_task)):
            replay_buffer = None
            if len(replay_buffer_X_list):
                replay_buffer = iter(DataLoader(TensorDataset(torch.cat(replay_buffer_X_list, dim=0), torch.cat(replay_buffer_y_list, dim=0)), batch_size=16, shuffle=True))
            else:
                replay_buffer = iter([])

            grad_matrices_epoch = []
            for inputs, labels in tqdm(task):
                # print(inputs.shape)
                try:
                    replay_inputs, replay_labels = next(replay_buffer)

                    inputs = torch.cat([inputs, replay_inputs], dim=0)
                    labels = torch.cat([labels, replay_labels], dim=0)

                    inputs.requires_grad_()  # Enable gradient tracking
                    labels.requires_grad_()  # Enable gradient tracking

                except StopIteration:
                    if i > 0:
                        print("Replay buffer exhausted")
                    pass

                if epoch == epochs_per_task - 1:
                    # toss a coin w probability 1/4 to decide whether to use each sample in the inputs in the replay buffer
                    # TODO: replace with a more sophisticated replay strategy
                    mask = torch.rand((inputs.shape[0],))
                    # print(mask)
                    new_replay_inputs = inputs[mask <= 0.5].detach()
                    new_replay_labels = labels[mask <= 0.5].detach()

                    # Convert to float32 before enabling gradients
                    new_replay_inputs = new_replay_inputs.to(torch.float32)
                    new_replay_labels = new_replay_labels.to(torch.float32)

                    replay_buffer_X_list.append(new_replay_inputs)
                    replay_buffer_y_list.append(new_replay_labels)

                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                # inputs.retain_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()

                # if epoch in checkpoints:
                    # pixel_grads = inputs.grad.mean(axis=1)
                    # grad_matrices_epoch.append(pixel_grads.clone())

                optimizer.step()
            # if epoch in checkpoints:
                # grad_matrices.append(torch.concat(grad_matrices_epoch, axis=0))

        # grad_variances = compute_VoG(grad_matrices, epoch_labels, checkpoints)
        # visualize_VoG(grad_variances, input_images, epoch_labels)
        
        print(f'Results after training on task {i + 1}')

        with torch.no_grad():
            for i, task_test in enumerate(test_tasks):
                test_loss, test_accuracy = evaluate(model, task_test, criterion, metric)

                task_test_losses[i].append(test_loss)
                task_test_accuracies[i].append(test_accuracy)

                print(f'Task {i+1} test loss: {test_loss}')
                print(f'Task {i+1} test accuracy: {test_accuracy}')


    return task_test_losses, task_test_accuracies

# Define the evaluation metric
def accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    return (output.argmax(1) == target).float().mean().item()

# Define the evaluation function
def evaluate(model: torch.nn.Module, test_loader: torch.utils.data.DataLoader, criterion: torch.nn.Module, metric: Callable[[float], float]) -> tuple[float, float]:
    model.eval()
    test_loss = 0
    test_accuracy = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            test_loss += criterion(outputs, labels).item()
            test_accuracy += metric(outputs, labels)

    test_loss /= len(test_loader)
    test_accuracy /= len(test_loader)

    return test_loss, test_accuracy


if __name__ == "__main__":
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    num_classes = 10

    # squeeze_net = torch.hub.load(
    #     "pytorch/vision:v0.10.0",
    #     "squeezenet1_0",
    #     pretrained = True
    # )

    model = CIFAR_CNN(num_classes=num_classes)

    # model.classifier = (
    #     nn.Sequential(  # modify classifier layer to return correct number of classes
    #         nn.Dropout(p=0.5, inplace=False),
    #         nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1)),
    #         nn.ReLU(inplace=True),
    #         nn.AdaptiveAvgPool2d(output_size=(1, 1)),
    #     )
    # )

    model = model.to(device)

    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    # Define the loss function
    criterion = torch.nn.CrossEntropyLoss()

    # Define the number of epochs per task
    epochs_per_task = 3
    batch_size = 64
    num_checkpoints = 5

    # Define the training and testing tasks
    train_tasks = []
    test_tasks = []

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n", action="store", type=int, default=5, help="Number of tasks"
    )
    args = parser.parse_args()

    n = args.n

    print(n)

    for i in range(1, n + 1):
        with open(f"../data/cifar-10-{n}/train/task_{i}", "rb") as f:
            task_train, task_labels = pickle.load(f)

        with open(f"../data/cifar-10-{n}/test/task_{i}", "rb") as f:
            task_test, task_test_labels = pickle.load(f)


        # Custom preprocessing function to handle batch processing of the dataset
        # Convert the numpy array into a tensor (for all images in the batch)
        task_train_tensor = torch.tensor(task_train, dtype=torch.float32)
        task_test_tensor = torch.tensor(task_test, dtype=torch.float32)

        # Normalize the images (scale them to [-1, 1] range)
        # CIFAR images are typically in the range [0, 255] and need to be scaled down to [0, 1] first.
        task_train_tensor = task_train_tensor / 255.0  # Scale from [0, 255] to [0, 1]
        task_test_tensor = task_test_tensor / 255.0  # Scale from [0, 255] to [0, 1]

        # Normalize the images using ImageNet means and standard deviations (adapt as needed)
        # CIFAR mean and std could also be used here, but we stick with standard normalization for now.
        mean = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32).view(1, 3, 1, 1)

        task_train_tensor = (task_train_tensor - mean) / std
        task_test_tensor = (task_test_tensor - mean) / std

        train_tasks.append(
            DataLoader(
                TensorDataset(
                    task_train_tensor,
                    torch.tensor(task_labels, dtype=torch.long, device=device),
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
                ),
                batch_size=batch_size,
                shuffle=False,
            )
        )

    task_test_losses, task_test_accuracies = training_loop(
        train_tasks,
        test_tasks,
        model,
        optimizer,
        criterion,
        accuracy,
        evaluate,
        epochs_per_task,
        num_checkpoints,
    )

    for i, (losses, accuracies) in enumerate(
        zip(task_test_losses, task_test_accuracies)
    ):
        print(f"Task {i+1} test loss: {losses}")
        print(f"Task {i+1} test accuracy: {accuracies}")
