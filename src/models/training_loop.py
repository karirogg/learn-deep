import torch
from typing import Callable, Optional
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from torch import nn
import wandb
import numpy as np
import pdb

from metrics.vog import compute_VoG, visualize_VoG
from metrics.learning_speed import calculate_learning_speed


def training_loop(
    train_tasks: list[torch.utils.data.DataLoader],
    test_tasks: list[torch.utils.data.DataLoader],
    unique_labels: list[list[int]],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    criterion: torch.nn.Module,
    device: torch.device,
    metric: Callable[[float], float],
    evaluate: Callable[
        [
            torch.nn.Module,
            torch.utils.data.DataLoader,
            torch.nn.Module,
            torch.device,
            Callable[[float], float],
            list[str],
        ],
        tuple[float, float],
    ],
    replay_buffer_strategy: Optional[
        Callable[
            [
                torch.nn.Module,
                torch.utils.data.DataLoader,
                list[torch.Tensor],
                list[torch.Tensor],
                int,
            ],
            tuple[list[torch.Tensor], list[torch.Tensor]],
        ]
    ],
    max_replay_buffer_size: int,
    epochs_per_task: int,
    num_checkpoints: int,
) -> list[float]:
    """
    The function trains the model on each of the different tasks sequentially using continual learning and uses a replay buffer to store the data from the previous tasks.
    """

    replay_buffer_X_list = []
    replay_buffer_y_list = []

    task_test_losses = []
    task_test_accuracies = []

    epoch_wise_classification_matrices = []

    optimizer_initial_state = optimizer.state_dict()
    scheduler_initial_state = scheduler.state_dict()

    for task in train_tasks:
        task_test_losses.append([])
        task_test_accuracies.append([])

        task_classification_matrix = torch.zeros((len(task.dataset), len(train_tasks), epochs_per_task))
        epoch_wise_classification_matrices.append(task_classification_matrix)

    for i, task in enumerate(train_tasks):
        print(f'Training on task {i + 1}')
        vog_data = {
            "gradient_matrices" : [], # container for storing gradients
            "checkpoints" : np.linspace(0, epochs_per_task, num_checkpoints, endpoint=False, dtype=np.int32), # iterations at which to store gradients
            "input_data" : map(torch.cat, zip(*[(img, labels) for img, labels, _ in task])) # stores input images and labels
        }

        for epoch in tqdm(range(epochs_per_task)):
            grad_matrices_epoch = [] # stores gradients for this epoch
            replay_buffer = None

            if replay_buffer_strategy and len(replay_buffer_X_list):
                replay_buffer = iter(DataLoader(TensorDataset(torch.cat(replay_buffer_X_list, dim=0), torch.cat(replay_buffer_y_list, dim=0)), batch_size=8, shuffle=True))

            for inputs, labels, _ in task:
                replay_inputs = None
                replay_labels = None

                if replay_buffer_strategy and len(replay_buffer_X_list):
                    try:
                        replay_inputs, replay_labels = next(replay_buffer)
                    except StopIteration:
                        replay_buffer = iter(DataLoader(TensorDataset(torch.cat(replay_buffer_X_list, dim=0), torch.cat(replay_buffer_y_list, dim=0), batch_size=8, shuffle=True)))
                        replay_inputs, replay_labels = next(replay_buffer)

                    if replay_inputs is not None:
                        inputs = torch.cat([inputs, replay_inputs], dim=0)
                        labels = torch.cat([labels, replay_labels], dim=0)

                inputs.requires_grad_()

                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                inputs.retain_grad() # for VoG
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                if epoch in vog_data["checkpoints"]:
                    pixel_grads = inputs.grad[:task.batch_size].mean(axis=1)
                    grad_matrices_epoch.append(pixel_grads.clone())
                optimizer.step()
                wandb.log({f"train-loss_task-{i}": loss})
            if epoch in vog_data["checkpoints"]:
                vog_data["gradient_matrices"].append(torch.concat(grad_matrices_epoch, axis=0))

            for j, task_train in enumerate(train_tasks):
                _, _, sample_wise_accuracy = evaluate(model, task_train, criterion, device, metric, unique_labels[j])

                epoch_wise_classification_matrices[j][:, i, epoch] = sample_wise_accuracy

            scheduler.step()

        grad_variances = compute_VoG(vog_data)
        input_images, labels = map(torch.cat, zip(*[(img, labels) for img, labels, _ in task]))
        visualize_VoG(grad_variances, input_images, labels)
        learning_speeds = calculate_learning_speed(epoch_wise_classification_matrices)
        pdb.set_trace()

        if replay_buffer_strategy:
            metrics = {"vog" : torch.hstack(grad_variances), "learning_speeds" : learning_speeds}
            replay_buffer_X_list, replay_buffer_y_list = replay_buffer_strategy(model, task, replay_buffer_X_list, replay_buffer_y_list, metrics, max_replay_buffer_size / (len(train_tasks) - 1))

        print(f'Results after training on task {i + 1}')

        with torch.no_grad():
            for i, task_test in enumerate(test_tasks):
                test_loss, test_accuracy, _ = evaluate(model, task_test, criterion, device, metric, unique_labels[i])

                task_test_losses[i].append(test_loss)
                task_test_accuracies[i].append(test_accuracy)
                # wandb.log({f"test-loss_task-{i}": test_loss})
                wandb.log({f"test-accuracy_task-{i}": test_accuracy})

                print(f'Task {i+1} test loss: {test_loss}')
                print(f'Task {i+1} test accuracy: {test_accuracy}')

        # Reset optimizer and scheduler for training on the next task
        optimizer.load_state_dict(optimizer_initial_state)
        scheduler.load_state_dict(scheduler_initial_state)

    return task_test_losses, task_test_accuracies, epoch_wise_classification_matrices
