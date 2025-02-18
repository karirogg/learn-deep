import torch
from typing import Callable, Optional
from tqdm import tqdm
import wandb
import numpy as np
import pickle
import sys
import pandas as pd

from replay_buffers.replay import Replay

from metrics.vog import VoG
from metrics.learning_speed import calculate_learning_speed
from metrics.mc_dropout import mc_dropout_inference

def update_learning_rate(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

def training_loop(
    train_tasks: list[torch.utils.data.DataLoader], # this parameter defines the tasks to train on
    test_tasks: list[torch.utils.data.DataLoader], # this parameter defines the tasks to test on
    model: torch.nn.Module, # this parameter defines the model to train
    optimizer: torch.optim.Optimizer, # this parameter defines the optimizer to use
    criterion: torch.nn.Module, # this parameter defines the loss function to use
    device: torch.device, # this parameter defines the device to train the model on
    metric: Callable[[float], float], # this parameter defines the metric to evaluate the model with
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
    ], # this parameter defines the function to evaluate the model
    replay_buffer: Optional[Replay], # this parameter defines the replay buffer to use
    epochs_per_task: int, # this parameter defines the number of epochs to train on each task
    num_checkpoints: int, # this parameter defines the number of checkpoints to store in the VoG class
    is_classification: bool, # this parameter defines whether the task is a classification task
    store_checkpoint: bool = False, # this parameter defines whether to store a checkpoint after training on each task
    use_checkpoint: bool = False, # this parameter defines whether to load a checkpoint from a previous run
    initial_lr: float = 5e-4, # this parameter defines the initial learning rate
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

    # initialie accuracy matrices
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

        # when training on the subsequent tasks, the learning rate is half of the initial learning rate due to previous instability in training
        if task_id != 0:
            update_learning_rate(optimizer, initial_lr / 2)

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

        # training loop
        for epoch in pbar:
            # reshuffle the replay buffer at the start of each epoch
            replay_buffer.reset()
            model.train()

            # during the second half of the training on the task after task 1, the learning rate is decayed by a factor specified by lr_decay
            if task_id != 0 and epoch == epochs_per_task // 2:
                update_learning_rate(optimizer, initial_lr * lr_decay)

            for inputs, labels, _ in task:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # sample from the replay buffer
                # the sample function outputs a list of tuples, where each tuple contains the inputs and labels for a task
                replay_inputs = replay_buffer.sample()

                # store the inputs and labels for the current task in the inputs sampled from the replay buffer
                replay_inputs[task_id] = (inputs, labels)

                optimizer.zero_grad()

                loss = 0

                # pass the inputs for each task through the model and calculate the loss
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

            # evaluate model after each epoch on both training and test tasks
            for j, task_test in enumerate(test_tasks):
                _, test_accuracy, sample_wise_accuracy = evaluate(model, task_test, criterion, device, metric, j)

                epoch_wise_classification_matrices_test[j][
                    :, task_id, epoch
                ] = sample_wise_accuracy

                _, train_accuracy, sample_wise_accuracy = evaluate(model, train_tasks[j], criterion, device, metric, j)

                epoch_wise_classification_matrices[j][
                    :, task_id, epoch
                ] = sample_wise_accuracy

                accuracy_summary += f"Task {j + 1}: Test: {test_accuracy:.4f} Train: {train_accuracy:.4f}, "

            pbar.set_description(accuracy_summary)

        if task_id < len(train_tasks) - 1 and (
            replay_buffer.strategy is not None or store_checkpoint
        ):

            # calculate training metrics and store them in a dataframe
            vog_results_train_early = vog_train.finalise(early=True).cpu().numpy()
            vog_results_train_late = vog_train.finalise(late=False).cpu().numpy()
            vog_results_train = vog_train.finalise().cpu().numpy()

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

            if isinstance(training_metrics_df, pd.DataFrame):
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
                        "vog": vog_results_train,
                        "learning_speed": dummy,
                        "predictive_entropy": dummy,
                        "mutual_information": dummy,
                        "variation_ratio": dummy,
                        "mean_std_deviation": dummy,
                        "mc_variance": training_metrics_df["MC_Variance"].to_numpy() if isinstance(training_metrics_df, pd.DataFrame) else dummy,
                    }
                replay_buffer.strategy(
                    model,
                    task,
                    task_id,
                    metrics,
                )

            # save metrics if store_checkpoint is True
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

        # finally, evaluate model on all test tasks after training on each task
        with torch.no_grad():
            for j, task_test in enumerate(test_tasks):
                test_loss, test_accuracy, _ = evaluate(model, task_test, criterion, device, metric, j)

                task_test_losses[j].append(test_loss)
                task_test_accuracies[j].append(test_accuracy)
                # wandb.log({f"test-loss_task-{task_id}": test_loss})
                wandb.log({f"test-accuracy_task-{j}": test_accuracy})

                print(f'Task {j+1} test loss: {test_loss}')
                print(f'Task {j+1} test accuracy: {test_accuracy}')

    return (
        task_test_losses,
        task_test_accuracies,
        epoch_wise_classification_matrices,
        epoch_wise_classification_matrices_test,
    )
