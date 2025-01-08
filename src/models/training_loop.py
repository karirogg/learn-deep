import torch
from typing import Callable, Optional
from tqdm import tqdm
import wandb
import numpy as np
import pdb
import pickle
import sys

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
    max_replay_buffer_size: int,
    epochs_per_task: int,
    num_checkpoints: int,
    is_classification: bool,
    store_checkpoint: bool = False,
    use_checkpoint: bool = False,
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
            checkpoint = torch.load('checkpoints/checkpoint.pth')
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            with open("checkpoints/metrics.pkl", "rb") as f:
                metrics = pickle.load(f)
            metrics = {col.lower(): torch.tensor(metrics[col].to_numpy(), dtype=torch.float32) for col in metrics.columns}
            replay_buffer.strategy(model, task, task_id, metrics, max_replay_buffer_size / (len(train_tasks) - 1))
            continue
        # print(f"State before training on task {task_id}:\nmodel: {model.state_dict}\noptimizer: {optimizer.state_dict}\nmetrics: {metrics}")
        # start training
        print(f"Training on task {task_id + 1}")
        vog = VoG(task, epochs_per_task, num_checkpoints, task_id, is_classification)

        if frozen[task_id]:
            for param in model.task_classifiers[task_id].parameters():
                param.requires_grad = True  # unfreeze current classification head
            frozen[task_id] = False

        for epoch in tqdm(range(epochs_per_task), file=sys.stderr):
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
                
            vog.update(model, epoch)
                
            for j, task_test in enumerate(test_tasks):
                _, _, sample_wise_accuracy = evaluate(model, task_test, criterion, device, metric, j)

                epoch_wise_classification_matrices_test[j][
                    :, task_id, epoch
                ] = sample_wise_accuracy

            for j, task_train in enumerate(train_tasks):
                _, _, sample_wise_accuracy = evaluate(model, task_train, criterion, device, metric, j)

                epoch_wise_classification_matrices[j][
                    :, task_id, epoch
                ] = sample_wise_accuracy

        if task_id < len(train_tasks)-1:
            vog_results = vog.finalise()
            # vog.visualise()
            if is_classification:
                metrics_df = mc_dropout_inference(
                    model,
                    task,
                    task_id,
                    device,
                    replay_buffer.weights,
                    num_samples=100,
                    classification=is_classification,
                    store_checkpoint=store_checkpoint,
                )
                metrics_df = metrics_df.set_index("Index")
                metrics_df["Learning_Speed"] = calculate_learning_speed(
                    epoch_wise_classification_matrices
                )[task_id]
                metrics_df["vog"] = vog_results.cpu()

            if replay_buffer.strategy is not None:
                dummy = np.zeros_like(vog_results.cpu())
                if is_classification:
                    metrics = {
                        "vog": metrics_df["vog"].to_numpy(),
                        "learning_speed": metrics_df["Learning_Speed"].to_numpy(),
                        "predictive_entropy": metrics_df["Predictive_Entropy"].to_numpy(),
                        "mutual_information": metrics_df[
                            "Mutual_Information"
                        ].to_numpy(),
                        "variation_ratio": metrics_df["Variation_Ratio"].to_numpy(),
                        "mean_std_deviation": metrics_df["Mean_Std_Deviation"].to_numpy(),
                        "mc_variance": dummy,
                    }
                else:

                    metrics = {
                        "vog": metrics_df["vog"].to_numpy(),
                        "learning_speed": dummy,
                        "predictive_entropy": dummy,
                        "mutual_information": dummy,
                        "variation_ratio": dummy,
                        "mean_std_deviation": dummy,
                        "mc_variance": mc_dropout_inference(
                            model,
                            task,
                            task_id,
                            device,
                            replay_buffer.weights,
                            num_samples=100,
                            classification=is_classification,
                            store_checkpoint=store_checkpoint,
                        )[1],
                    }

                replay_buffer.strategy(model, task, task_id, metrics, max_replay_buffer_size / (len(train_tasks) - 1))

                if task_id == 0 and store_checkpoint:
                    checkpoint = {
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()
                    }
                    torch.save(checkpoint, 'checkpoints/checkpoint.pth')
                    filename = f"checkpoints/metrics.pkl"
                    if is_classification:
                        metrics_df.to_pickle(filename)
                    else:
                        with open(filename, "wb") as f:
                            pickle.dump(metrics, f)
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

    return task_test_losses, task_test_accuracies, epoch_wise_classification_matrices_test
