import torch
from typing import Callable, Optional
from tqdm import tqdm
import wandb
import numpy as np
import pdb
import pickle
import sys
import copy

from replay_buffers.replay import Replay

from metrics.vog import VoG
from metrics.learning_speed import calculate_learning_speed
from metrics.mc_dropout import mc_dropout_inference

def training_loop(
    train_tasks: list[torch.utils.data.DataLoader],
    test_tasks: list[torch.utils.data.DataLoader],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
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
    replay_buffer: Optional[Replay],
    epochs_per_task: int,
    num_checkpoints: int,
    is_classification: bool,
    store_checkpoint: bool = False,
    use_checkpoint: bool = False,
    initial_lr: float = 5e-4,
    lr_decay: float = 0.1, # this parameter defines how much the lr decayse after training on the first task
    seed: int = 42,
) -> list[float]:
    """
    The function trains the model on each of the different tasks sequentially using continual learning and uses a replay buffer to store the data from the previous tasks.
    """

    task_test_losses = []
    task_test_accuracies = []

    epoch_wise_classification_matrices = []
    epoch_wise_classification_matrices_test = []
    frozen = torch.zeros(len(train_tasks), dtype=torch.bool)

    for task in train_tasks:
        task_test_losses.append([])
        task_test_accuracies.append([])

        task_classification_matrix = torch.zeros((len(task.dataset), len(train_tasks), epochs_per_task))
        epoch_wise_classification_matrices.append(task_classification_matrix)

    for task in test_tasks:
        task_classification_matrix = torch.zeros((len(task.dataset), len(train_tasks), epochs_per_task))
        epoch_wise_classification_matrices_test.append(task_classification_matrix.clone())

    for task_id, task in enumerate(train_tasks):
        # load model and metrics if training on task 0 has already been done and metrics have been computed (use only for classification) -> task 0 is skipped in this case
        if task_id == 0 and use_checkpoint:
            print("WARNING: loading checkpoint from previous run")
            checkpoint = torch.load(f'checkpoints/checkpoint_seed{seed}.pth', weights_only=True)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch_wise_classification_matrices = checkpoint['epoch_wise_classification_matrices']
            epoch_wise_classification_matrices_test = checkpoint['epoch_wise_classification_matrices_test']
            with open(f"checkpoints/seed_{seed}_training_metrics.pkl", "rb") as f:
                metrics = pickle.load(f)

            metrics = {col.lower(): torch.tensor(metrics[col].to_numpy(), dtype=torch.float32) for col in metrics.columns}
            replay_buffer.strategy(model, task, task_id, metrics)
            vog_train = VoG(
                task, epochs_per_task, num_checkpoints, task_id, is_classification
            )
            vog_test = VoG(
                test_tasks[task_id],
                epochs_per_task,
                num_checkpoints,
                task_id,
                is_classification,
            )
            continue
        # print(f"State before training on task {task_id}:\nmodel: {model.state_dict}\noptimizer: {optimizer.state_dict}\nmetrics: {metrics}")
        # start training
        print(f"Training on task {task_id + 1}")

        # Initialize variance_of_gradients classes for each task except for the last
        if task_id < len(train_tasks) - 1:
            vog_train = VoG(
                task, epochs_per_task, num_checkpoints, task_id, is_classification
            )
            vog_test = VoG(
                test_tasks[task_id],
                epochs_per_task,
                num_checkpoints,
                task_id,
                is_classification,
            )

        if frozen[task_id]:
            for param in model.task_classifiers[task_id].parameters():
                param.requires_grad = True  # unfreeze current classification head
            frozen[task_id] = False

        
        pbar = tqdm(range(epochs_per_task), file=sys.stderr)

        for epoch in pbar:
            replay_buffer.reset()
            model.train()

            for inputs, labels, _ in task:

                inputs = inputs.to(device)
                labels = labels.to(device)

                replay_inputs = replay_buffer.sample()
                replay_inputs[task_id] = (inputs, labels)

                optimizer.zero_grad()

                loss = 0

                for i, (inp, lab) in enumerate(replay_inputs):
                    if inp is None or len(inp) == 0:
                        if not frozen[i]:
                            # freeze classification head for inactive task
                            for param in model.task_classifiers[i].parameters():
                                param.requires_grad = False
                            frozen[i] = True
                        continue

                    inp = inp.to(device)
                    lab = lab.to(device)

                    outputs = model(inp, i)

                    loss += criterion(outputs, lab)

                loss.backward()

                optimizer.step()

            wandb.log({f"train-loss_task-{task_id}": loss})

            vog_train.update(model, task_id, epoch)
            vog_test.update(model, task_id, epoch)

            accuracy_summary = ''

            for j, task_test in enumerate(test_tasks):
                _, test_accuracy, sample_wise_accuracy = evaluate(model, task_test, criterion, device, metric, j)

                epoch_wise_classification_matrices_test[j][
                    :, task_id, epoch
                ] = sample_wise_accuracy

                _, train_accuracy, sample_wise_accuracy = evaluate(model, train_tasks[j], criterion, device, metric, j)

                epoch_wise_classification_matrices[j][
                    :, task_id, epoch
                ] = sample_wise_accuracy

                accuracy_summary += f"Task {j}: Test: {test_accuracy:.4f} Train: {train_accuracy:.4f}, "

            pbar.set_description(accuracy_summary)

        if task_id < len(train_tasks) - 1 and (
            replay_buffer.strategy is not None or store_checkpoint
        ):

            # calculate training metrics
            vog_results_train_early = vog_train.finalise(early=True).cpu().numpy()
            vog_results_train_late = vog_train.finalise(late=False).cpu().numpy()
            vog_results_train = vog_train.finalise().cpu().numpy()
            # vog_train.visualise()
            training_metrics_df = mc_dropout_inference(
                model,
                task,
                task_id,
                device,
                replay_buffer.weights,
                num_samples=100,
                classification=is_classification,
                store_checkpoint=store_checkpoint,
            )
            training_metrics_df = training_metrics_df.set_index("Index")
            training_metrics_df["Variance_of_Gradients_Early"] = vog_results_train_early
            training_metrics_df["Variance_of_Gradients_Late"] = vog_results_train_late
            training_metrics_df["vog"] = vog_results_train

            if is_classification:
                training_metrics_df["Learning_Speed"] = calculate_learning_speed(
                    epoch_wise_classification_matrices
                )[task_id]

            if replay_buffer.strategy is not None:
                # replay buffer expects a dictionary
                dummy = np.zeros_like(vog_results_train_late)

                if is_classification:
                    metrics = {
                        "vog": vog_results_train_late,
                        "learning_speed": training_metrics_df[
                            "Learning_Speed"
                        ].to_numpy(),
                        "predictive_entropy": training_metrics_df[
                            "Predictive_Entropy"
                        ].to_numpy(),
                        "mutual_information": training_metrics_df[
                            "Mutual_Information"
                        ].to_numpy(),
                        "variation_ratio": training_metrics_df[
                            "Variation_Ratio"
                        ].to_numpy(),
                        "mean_std_deviation": training_metrics_df[
                            "Mean_Std_Deviation"
                        ].to_numpy(),
                        "mc_variance": dummy,
                    }
                else:
                    metrics = {
                        "vog": vog_results_train_late,
                        "learning_speed": dummy,
                        "predictive_entropy": dummy,
                        "mutual_information": dummy,
                        "variation_ratio": dummy,
                        "mean_std_deviation": dummy,
                        "mc_variance": training_metrics_df["MC_Variance"].to_numpy(),
                    }
                replay_buffer.strategy(
                    model,
                    task,
                    task_id,
                    metrics,
                )

            if store_checkpoint:
                checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch_wise_classification_matrices": epoch_wise_classification_matrices,
                    "epoch_wise_classification_matrices_test": epoch_wise_classification_matrices_test,
                }
                torch.save(checkpoint, f"checkpoints/checkpoint_seed{seed}.pth")

                # calculate test metrics
                vog_results_test_early = vog_test.finalise(early=True).cpu().numpy()
                vog_results_test_late = vog_test.finalise(late=True).cpu().numpy()
                vog_results_test = vog_test.finalise().cpu().numpy()
                # vog_test.visualise()
                test_metrics_df = mc_dropout_inference(
                    model,
                    test_tasks[task_id],
                    task_id,
                    device,
                    replay_buffer.weights,
                    num_samples=100,
                    classification=is_classification,
                    store_checkpoint=store_checkpoint,
                )
                test_metrics_df = test_metrics_df.set_index("Index")
                test_metrics_df["Variance_of_Gradients_Early"] = vog_results_test_early
                test_metrics_df["Variance_of_Gradients_Late"] = vog_results_test_late
                test_metrics_df["vog"] = vog_results_test

                if is_classification:
                    test_metrics_df["Learning_Speed"] = calculate_learning_speed(
                        epoch_wise_classification_matrices_test
                    )[task_id]

                # store both training and test metrics
                filename = f"checkpoints/seed_{seed}_"
                training_metrics_df.to_pickle(filename + "training_metrics.pkl")
                test_metrics_df.to_pickle(filename + "test_metrics.pkl")

                print("stored metrics and training checkpoint")

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

        if task_id == 0:
            new_lr = initial_lr * lr_decay
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr

    return (
        task_test_losses,
        task_test_accuracies,
        epoch_wise_classification_matrices,
        epoch_wise_classification_matrices_test,
    )
