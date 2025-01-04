import torch
from typing import Callable, Optional
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from torch import nn
import wandb
import numpy as np
import pdb

from replay_buffers.replay import Replay

from metrics.vog import compute_VoG, visualize_VoG
from metrics.learning_speed import calculate_learning_speed
from metrics.mc_dropout import mc_dropout_inference


def training_loop(
    train_tasks: list[torch.utils.data.DataLoader],
    test_tasks: list[torch.utils.data.DataLoader],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    # scheduler: torch.optim.lr_scheduler._LRScheduler,
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
    replay_buffer: Optional[
        Replay
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

    for task in train_tasks:
        task_test_losses.append([])
        task_test_accuracies.append([])

        task_classification_matrix = torch.zeros((len(task.dataset), len(train_tasks), epochs_per_task))
        epoch_wise_classification_matrices.append(task_classification_matrix)

    for task_id, task in enumerate(train_tasks):
        print(f"Training on task {task_id + 1}")
        vog_data = {
            "gradient_matrices" : [], # container for storing gradients
            "checkpoints" : np.linspace(0, epochs_per_task, num_checkpoints, endpoint=False, dtype=np.int32), # iterations at which to store gradients
            "input_data" : map(torch.cat, zip(*[(img, labels) for img, labels, _ in task])) # stores input images and labels
        }

        for epoch in tqdm(range(epochs_per_task)):
            grad_matrices_epoch = [] # stores gradients for this epoch
            replay_buffer.reset()

            for inputs, labels, _ in task:
                input_length = inputs.shape[0]
                
                inputs = inputs.to(device)
                labels = labels.to(device)

                replay_inputs = replay_buffer.sample()

                replay_inputs[task_id] = (inputs, labels)


                inputs.requires_grad_()

                optimizer.zero_grad()
                inputs.retain_grad()  # for VoG

                loss = 0

                for i, (inp, lab) in enumerate(replay_inputs):
                    if inp is None or len(inp) == 0:
                        continue

                    inp = inp.to(device)
                    lab = lab.to(device)

                    outputs = model(inp, i)

                    loss += criterion(outputs, lab)

                loss.backward()

                if epoch in vog_data["checkpoints"]:
                    pixel_grads = inputs.grad[:input_length].mean(axis=1)
                    grad_matrices_epoch.append(pixel_grads.clone())

                optimizer.step()
                wandb.log({f"train-loss_task-{task_id}": loss})
            if epoch in vog_data["checkpoints"]:
                vog_data["gradient_matrices"].append(torch.concat(grad_matrices_epoch, axis=0))

            for j, task_train in enumerate(train_tasks):
                _, _, sample_wise_accuracy = evaluate(model, task_train, criterion, device, metric, j)

                epoch_wise_classification_matrices[j][
                    :, task_id, epoch
                ] = sample_wise_accuracy

            # scheduler.step()


        mc_dropout_df = mc_dropout_inference(model, task, task_id, device, replay_buffer.weights, num_samples=100, classification=True)
        grad_variances = compute_VoG(vog_data)
        input_images, labels = map(torch.cat, zip(*[(img, labels) for img, labels, _ in task]))
        # visualize_VoG(grad_variances, input_images, labels)
        learning_speeds = calculate_learning_speed(epoch_wise_classification_matrices)

        if replay_buffer.strategy is not None:
            metrics = {
                "vog": torch.hstack(grad_variances),
                "learning_speed": learning_speeds[task_id][: labels.shape[0]],
                "mc_entropy" : torch.tensor(mc_dropout_df["Predictive_Entropy"].values),
                "mc_mutual_information" : torch.tensor(mc_dropout_df["Mutual_Information"].values),
                "mc_variation_ratio" : torch.tensor(mc_dropout_df["Variation_Ratio"].values),
                "mc_mean_std" : torch.tensor(mc_dropout_df["Mean_Std_Deviation"].values),
            }

            replay_buffer.strategy(model, task, task_id, metrics, max_replay_buffer_size / (len(train_tasks) - 1))

        print(f"Results after training on task {task_id + 1}")

        with torch.no_grad():
            for j, task_test in enumerate(test_tasks):
                test_loss, test_accuracy, _ = evaluate(model, task_test, criterion, device, metric, j)

                task_test_losses[j].append(test_loss)
                task_test_accuracies[j].append(test_accuracy)
                # wandb.log({f"test-loss_task-{task_id}": test_loss})
                wandb.log({f"test-accuracy_task-{j}": test_accuracy})

                print(f'Task {j+1} test loss: {test_loss}')
                print(f'Task {j+1} test accuracy: {test_accuracy}')

    return task_test_losses, task_test_accuracies, epoch_wise_classification_matrices
