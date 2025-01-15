import argparse
import matplotlib.pyplot as plt
import os
import torch
import wandb
import numpy as np
import json
import pickle

from replay_buffers.replay import Replay

from models.cifar.accuracy import accuracy
from models.cifar.evaluate import evaluate
from models.cifar.task_preprocessing import preprocess_cifar
from models.training_loop import training_loop
from models.cifar.TIL_squeezenet import Task_IL_SqueezeNet

from utils.fix_seed import fix_seed

if __name__ == "__main__":
    # define flags for script
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", action="store", type=int, default=2, help="Number of tasks")
    parser.add_argument("--epochs", action="store", type=int, default=10, help="Number of epochs")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--classes", action="store", type=int, default=100, help="Number of classes")
    parser.add_argument("--replay-buffer", action="store", type=str, default=None, help="Replay buffer strategy")
    parser.add_argument("--replay-weights", type=str, default="{}") # example: --replay-weights '{"vog": 1.0, "learning_speed": 1.0, "predictive_entropy": 1.0, "mutual_information": 1.0, "variation_ratio": 1.0, "mean_std_deviation": 1.0, "mc_variance": 0.0}' NOTE: mc_variance should have zero weight for classification
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--store-checkpoint", action="store_true")
    parser.add_argument("--use-checkpoint", action="store_true")
    parser.add_argument("--buffer-size", action="store", type=int, default=10, help="Size of replay buffer (percentage of training set)")
    parser.add_argument("--cutoff-lower", action="store", type=int, default=20, help="Percentage of lower cutoff")
    parser.add_argument("--cutoff-upper", action="store", type=int, default=20, help="Percentage of upper cutoff")

    args = parser.parse_args()

    # set seed
    fix_seed(args.seed)

    # hyperparameters for task-incremental learning
    n = args.n
    epochs_per_task = args.epochs
    num_classes = args.classes

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    model_config = {"num_classes" : num_classes}

    # initialize model
    model = Task_IL_SqueezeNet(num_classes_per_task = int(num_classes / n), num_tasks=n)
    model.to(device)

    wandb.init(project="learn-deep", config=model_config, mode="online" if args.wandb else "disabled")

    # set up hyperparamters, optimizer and loss function
    initial_lr = 5e-4
    lr_decay = 0.25
    batch_size = 128
    replay_batch_size = 8

    optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr, weight_decay=5e-5)

    criterion = torch.nn.CrossEntropyLoss()

    # always want to store some checkpoint if we are storing them
    num_checkpoints = min(10, epochs_per_task)

    if args.store_checkpoint:
        if not os.path.exists("checkpoints"):
            os.mkdir("checkpoints")

    # preprocess data
    train_tasks, test_tasks = preprocess_cifar(num_classes, n, batch_size, device)

    # set up replay byffer
    replay_params = {"remove_lower_percent" : args.cutoff_lower, "remove_upper_percent" : args.cutoff_upper}
    replay_weights = json.loads(args.replay_weights)

    # specify the number of samples to add in each task
    # this value is calculated from the buffer-size argument which is a percentage of the training set
    num_samples_to_add = len(train_tasks[0].dataset) * args.buffer_size // 100
    replay_buffer = Replay(replay_params, strategy=args.replay_buffer, batch_size=replay_batch_size, samples_to_add=num_samples_to_add, num_tasks=n, weights=replay_weights)

    if args.replay_buffer:
        print("running with replay strategy:", args.replay_buffer)
    else:
        print("WARNING: no valid replay strategy provided - running without")

    (
        task_test_losses,
        task_test_accuracies,
        epoch_wise_classification_matrices,
        epoch_wise_classification_matrices_test,
    ) = training_loop(
        train_tasks=train_tasks,
        test_tasks=test_tasks,
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        metric=accuracy,
        evaluate=evaluate,
        replay_buffer=replay_buffer,
        epochs_per_task=epochs_per_task,
        num_checkpoints=num_checkpoints,
        is_classification=True,
        store_checkpoint=args.store_checkpoint,
        use_checkpoint=args.use_checkpoint,
        initial_lr=initial_lr,
        lr_decay=lr_decay,
        seed=args.seed,
    )

    wandb.finish()

    replay_buffer_details = args.replay_buffer

    if args.replay_buffer == 'simple_sorted':
        for key, value in replay_weights.items():
            if value > 0:
                replay_buffer_details += f'_{key}_{value}'

    # plots
    print("creating plots...")
    task_name = f'cifar_{num_classes}_n_{n}_epochs_{epochs_per_task}_replay_{replay_buffer_details}_seed_{args.seed}_lower_{args.cutoff_lower}_upper_{args.cutoff_upper}'

    if not os.path.exists('../img'):
        os.mkdir('../img')

    if not os.path.exists('../img/cifar'):
        os.mkdir('../img/cifar')

    if not os.path.exists('../img/cifar/task_progression'):
        os.mkdir('../img/cifar/task_progression')
        os.mkdir('../img/cifar/heatmaps')

    for i, task in enumerate(test_tasks):
        task_progression = []
        for j in range(len(test_tasks)):
            task_progression.append(
                torch.mean(epoch_wise_classification_matrices_test[i][:, j, :], dim=0)
            )

        plt.plot(
            np.arange(len(test_tasks) * epochs_per_task),
            np.concatenate(task_progression, axis=0),
            label=f"Task {i+1}",
        )

    plt.legend()

    plt.xlabel('Epoch')
    plt.ylabel('Classification Accuracy')

    plt.xlim(0, len(train_tasks) * epochs_per_task)
    plt.ylim(0, 1)

    plt.savefig(f"../img/cifar/task_progression/{task_name}_test.pdf")
    plt.close()

    for i, task in enumerate(train_tasks):
        task_progression = []
        for j in range(len(train_tasks)):
            task_progression.append(
                torch.mean(epoch_wise_classification_matrices[i][:, j, :], dim=0)
            )

        plt.plot(
            np.arange(len(train_tasks) * epochs_per_task),
            np.concatenate(task_progression, axis=0),
            label=f"Task {i+1}",
        )

    plt.legend()

    plt.xlabel('Epoch')
    plt.ylabel('Classification Accuracy')

    plt.xlim(0, len(train_tasks) * epochs_per_task)
    plt.ylim(0, 1)

    plt.savefig(f"../img/cifar/task_progression/{task_name}_train.pdf")
    plt.close()

    for i, task in enumerate(train_tasks):
        # plot heatmap of classification accuracy per sample

        order = torch.argsort(
            torch.mean(epoch_wise_classification_matrices[i][:, i, :], axis=1),
            descending=False,
        )
        concat_task_progression = torch.cat([epoch_wise_classification_matrices[i][:, j, :] for j in range(n)], dim=1)[order]

        plt.figure(figsize=(5 * n, 5))
        plt.imshow(concat_task_progression.cpu().numpy(), cmap='cividis', interpolation='nearest', aspect='auto')
        plt.xlabel("Epoch")
        plt.ylabel("Example number")
        plt.savefig(f"../img/cifar/heatmaps/{task_name}_task_{i}_train.pdf")
        plt.close()

        order = torch.argsort(
            torch.mean(epoch_wise_classification_matrices_test[i][:, i, :], axis=1),
            descending=False,
        )
        concat_task_progression = torch.cat(
            [epoch_wise_classification_matrices_test[i][:, j, :] for j in range(n)],
            dim=1,
        )[order]

        plt.figure(figsize=(5 * n, 5))
        plt.imshow(
            concat_task_progression.cpu().numpy(),
            cmap="cividis",
            interpolation="nearest",
            aspect="auto",
        )
        plt.xlabel("Epoch")
        plt.ylabel("Example number")
        plt.savefig(f"../img/cifar/heatmaps/{task_name}_task_{i}_test.pdf")
        plt.close()

    print("done")

    if args.store_checkpoint and replay_buffer.strategy == None:
        # Save epoch_wise classification matrices
        matrices_train = torch.stack(epoch_wise_classification_matrices, dim=0).detach()
        matrices_test = torch.stack(
            epoch_wise_classification_matrices_test, dim=0
        ).detach()

        with open(f"checkpoints/matrices_train_seed{args.seed}", "wb") as f:
            pickle.dump(matrices_train, f)

        with open(f"checkpoints/matrices_test_seed{args.seed}", "wb") as f:
            pickle.dump(matrices_test, f)

        print("Saved classification_matrices")

    for i, (losses, accuracies) in enumerate(
        zip(task_test_losses, task_test_accuracies)
    ):
        print(f"Task {i+1} test loss: {losses}")
        print(f"Task {i+1} test accuracy: {accuracies}")
