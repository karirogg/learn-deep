import torch
from typing import Callable, Optional
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from torch import nn
import wandb
import numpy as np

def training_loop(
        train_tasks: list[torch.utils.data.DataLoader], 
        test_tasks: list[torch.utils.data.DataLoader], 
        unique_labels: list[list[int]],
        model: torch.nn.Module, 
        optimizer: torch.optim.Optimizer, 
        criterion: torch.nn.Module, 
        device: torch.device,
        metric: Callable[[float], float],
        evaluate: Callable[[torch.nn.Module, torch.utils.data.DataLoader, torch.nn.Module, torch.device, Callable[[float], float], list[str]], tuple[float, float]],
        replay_buffer_strategy: Optional[Callable[[torch.nn.Module, torch.utils.data.DataLoader, list[torch.Tensor], list[torch.Tensor], int], tuple[list[torch.Tensor], list[torch.Tensor]]]],
        max_replay_buffer_size: int,
        epochs_per_task: int,
    ) -> list[float]:
    """
    The function trains the model on each of the different tasks sequentially using continual learning and uses a replay buffer to store the data from the previous tasks.
    """

    replay_buffer_X_list = []
    replay_buffer_y_list = []

    task_test_losses = []
    task_test_accuracies = []

    epoch_wise_classification_matrices = []

    for task in train_tasks:
        task_test_losses.append([])
        task_test_accuracies.append([])

        task_classification_matrix = torch.zeros((len(task.dataset), len(train_tasks), epochs_per_task))
        epoch_wise_classification_matrices.append(task_classification_matrix)

    for i, task in enumerate(train_tasks):
        print(f'Training on task {i + 1}')

        for epoch in tqdm(range(epochs_per_task)):
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
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                wandb.log({f"train-loss_task-{i}": loss})

            for j, task_train in enumerate(train_tasks):
                _, _, sample_wise_accuracy = evaluate(model, task_train, criterion, device, metric, unique_labels[j])

                epoch_wise_classification_matrices[j][:, i, epoch] = sample_wise_accuracy

        if replay_buffer_strategy:
            replay_buffer_X_list, replay_buffer_y_list = replay_buffer_strategy(model, task, replay_buffer_X_list, replay_buffer_y_list, max_replay_buffer_size / (len(train_tasks) - 1))
    
        print(f'Results after training on task {i + 1}')

        with torch.no_grad():
            for i, task_test in enumerate(test_tasks):
                test_loss, test_accuracy, _ = evaluate(model, task_test, criterion, device, metric, unique_labels[i])
                # pdb.set_trace()

                task_test_losses[i].append(test_loss)
                task_test_accuracies[i].append(test_accuracy)
                # wandb.log({f"test-loss_task-{i}": test_loss})
                wandb.log({f"test-accuracy_task-{i}": test_accuracy})

                print(f'Task {i+1} test loss: {test_loss}')
                print(f'Task {i+1} test accuracy: {test_accuracy}')

    return task_test_losses, task_test_accuracies, epoch_wise_classification_matrices