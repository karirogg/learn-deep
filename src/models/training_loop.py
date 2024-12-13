import torch
from typing import Callable
import pickle
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from torch import nn
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import wandb

from replay_buffers.uniform_replay_buffer import uniform_replay_buffer

from models.cifar.accuracy import accuracy
from models.cifar.evaluate import evaluate
from models.cifar.task_preprocessing import preprocess_cifar

from utils.fix_seed import fix_seed

from models.custom_cnn import CIFAR_CNN

from metrics.vog import compute_VoG, visualize_VoG

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
        replay_buffer_strategy: Callable[[torch.nn.Module, torch.utils.data.DataLoader, list[torch.Tensor], list[torch.Tensor], int], tuple[list[torch.Tensor], list[torch.Tensor]]],
        max_replay_buffer_size: int,
        epochs_per_task: int,
    ) -> list[float]:
    """
    The function trains the model on each of the different tasks sequentially using continual learning and uses a replay buffer to store the data from the previous tasks.
    """

    # Ignore replay buffer for now
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

            if len(replay_buffer_X_list):
                replay_buffer = iter(DataLoader(TensorDataset(torch.cat(replay_buffer_X_list, dim=0), torch.cat(replay_buffer_y_list, dim=0)), batch_size=8, shuffle=True))

            for inputs, labels, _ in task:
                replay_inputs = None
                replay_labels = None

                if len(replay_buffer_X_list):
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



if __name__ == "__main__":
    fix_seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument("--n", action="store", type=int, default=2, help="Number of tasks")
    parser.add_argument("--epochs", action="store", type=int, default=10, help="Number of epochs")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--classes", action="store", type=int, default=10, help="Number of classes")
    
    args = parser.parse_args()
    n = args.n
    epochs_per_task = args.epochs
    num_classes = args.classes

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model_config = {"num_classes" : 10}
    model = CIFAR_CNN(cfg=model_config)
    model = model.to(device)
    wandb.init(project="learn-deep", config=model_config, mode="online" if args.wandb else "disabled")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    criterion = torch.nn.CrossEntropyLoss()

    batch_size = 64
    num_checkpoints = 5

    train_tasks, test_tasks, unique_labels = preprocess_cifar(num_classes, n, batch_size, device)

    task_test_losses, task_test_accuracies, epoch_wise_classification_matrices = training_loop(
        train_tasks=train_tasks,
        test_tasks=test_tasks,
        unique_labels=unique_labels,
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        metric=accuracy,
        evaluate=evaluate,
        replay_buffer_strategy=uniform_replay_buffer,
        max_replay_buffer_size=5000,
        epochs_per_task=epochs_per_task,
    )

    wandb.finish()

    for i, task in enumerate(train_tasks):
        task_progression = []
        for j in range(len(train_tasks)):
            task_progression.append(torch.mean(epoch_wise_classification_matrices[i][:, j, :], dim=0))

        plt.plot(np.arange(len(train_tasks) * epochs_per_task), np.concatenate(task_progression, axis=0), label=f'Task {i+1}')

    plt.legend()

    if not os.path.exists('../img/'):
        os.mkdir('../img')

    if not os.path.exists('../img/task_progression'):
        os.mkdir('../img/task_progression')
        os.mkdir('../img/heatmaps')

    plt.xlabel('Epoch')
    plt.ylabel('Classification Accuracy')

    plt.xlim(0, len(train_tasks) * epochs_per_task)
    plt.ylim(0, 1)

    plt.savefig(f'../img/task_progression/cifar_{num_classes}_n_{n}_epochs_{epochs_per_task}.png')
    plt.close()

    for i, task in enumerate(train_tasks):
        # plot heatmap of classification accuracy per sample

        order = torch.argsort(torch.mean(epoch_wise_classification_matrices[i][:,i,:], axis=1), descending=False)

        concat_task_progression = torch.cat([epoch_wise_classification_matrices[i][:, j, :] for j in range(n)], dim=1)[order]

        plt.figure(figsize=(5 * n, 5))
        plt.imshow(concat_task_progression.cpu().numpy(), cmap='cividis', interpolation='nearest', aspect='auto')

        plt.savefig(f'../img/heatmaps/cifar_{num_classes}_n_{n}_task_{i+1}_epochs_{epochs_per_task}.png')

    for i, (losses, accuracies) in enumerate(
        zip(task_test_losses, task_test_accuracies)
    ):
        print(f"Task {i+1} test loss: {losses}")
        print(f"Task {i+1} test accuracy: {accuracies}")
