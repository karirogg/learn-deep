import argparse
import matplotlib.pyplot as plt
import os
import torch
import wandb
import numpy as np
import json
import pickle

from replay_buffers.replay import Replay
from torch.utils.data import DataLoader, TensorDataset

from models.ewf.mse import mse
from models.cifar.evaluate import evaluate
from models.ewf.til_nn import TaskILNN

from models.training_loop import training_loop

from utils.fix_seed import fix_seed


if __name__ == "__main__":
    # define flags for script
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", action="store", type=int, default=10, help="Number of epochs")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--replay-buffer", action="store", type=str, default=None, help="Replay buffer strategy")
    parser.add_argument("--replay-weights", type=str, default="{}") # example: --replay_weights '{"vog": 1.0, "learning_speed": 1.0, "predictive_entropy": 0.0, "mutual_information": 0.0, "variation_ratio": 0.0, "mean_std_deviation": 0.0, "mc_variance": 1.0}' NOTE: all mc_weights except mc_variance should have zero weight for regression
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
    n = 4
    epochs_per_task = args.epochs

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    model = TaskILNN(num_tasks=n)
    model.to(device)

    model_config = {"dataset" : 'EuropeWindFarm'}

    wandb.init(project="learn-deep", config=model_config, mode="online" if args.wandb else "disabled")

    # hyperparameters and optimizer set up
    initial_lr = 1e-5
    lr_decay = 1
    batch_size = 128
    replay_batch_size = 8
    num_checkpoints = 5

    optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)
    criterion = torch.nn.MSELoss()

    # load the data in memory (no further preprocessing is needed)
    train_tasks = []
    test_tasks = []

    for i in range(1, n+1):
        with open(f'../data/ewf/train/task_{i}', 'rb') as f:
            task_X, task_y = pickle.load(f)

            task_train_tensor = torch.tensor(task_X, dtype=torch.float32, device=device)
            task_labels = torch.tensor(task_y, dtype=torch.float32, device=device)

            train_tasks.append(
                DataLoader(
                    TensorDataset(
                        task_train_tensor,
                        task_labels,
                        torch.arange(len(task_labels), device=device), # this is to keep track of the order of the samples
                    ),
                    batch_size=batch_size,
                    shuffle=True,
                )
            )

        with open(f'../data/ewf/test/task_{i}', 'rb') as f:
            task_X, task_y = pickle.load(f)

            task_test_tensor = torch.tensor(task_X, dtype=torch.float32, device=device)
            task_test_labels = torch.tensor(task_y, dtype=torch.float32, device=device)

            test_tasks.append(
                DataLoader(
                    TensorDataset(
                        task_test_tensor,
                        task_test_labels,
                        torch.arange(len(task_test_labels), device=device),
                    ),
                    batch_size=batch_size,
                    shuffle=False,
                )
            )

    # setup replay buffer
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

    task_test_losses, task_test_accuracies, epoch_wise_classification_matrices, epoch_wise_classification_matrices_test = (
        training_loop(
            train_tasks=train_tasks,
            test_tasks=test_tasks,
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            metric=mse,
            evaluate=evaluate,
            replay_buffer=replay_buffer,
            epochs_per_task=epochs_per_task,
            num_checkpoints=num_checkpoints,
            is_classification=False,
            store_checkpoint=args.store_checkpoint,
            use_checkpoint=args.use_checkpoint,
            seed=args.seed,
            initial_lr=initial_lr,
            lr_decay=lr_decay,
        )
    )

    wandb.finish()

    replay_buffer_details = args.replay_buffer

    if args.replay_buffer == 'simple_sorted':
        for key, value in replay_weights.items():
            if value > 0:
                replay_buffer_details += f'_{key}_{round(value, 2)}'

    print("creating plots...")
    task_name = f'ewf_epochs_{epochs_per_task}_replay_{replay_buffer_details}_seed_{args.seed}_lower_{args.cutoff_lower}_upper_{args.cutoff_upper}'

    if not os.path.exists('../img/'):
        os.mkdir('../img')

    if not os.path.exists('../img/ewf'):
        os.mkdir('../img/ewf')

    if not os.path.exists('../img/ewf/task_progression'):
        os.mkdir('../img/ewf/task_progression')
        os.mkdir('../img/ewf/heatmaps')

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
    plt.ylabel('MSE')

    plt.xlim(0, len(train_tasks) * epochs_per_task)
    plt.ylim(0, None)

    plt.savefig(f"../img/ewf/task_progression/{task_name}_test.pdf")
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
            label=f"Task {i+1} (train)",
        )

    plt.legend()

    plt.xlabel('Epoch')
    plt.ylabel('MSE')

    plt.xlim(0, len(train_tasks) * epochs_per_task)
    plt.ylim(0, None)

    plt.savefig(f"../img/ewf/task_progression/{task_name}_train.pdf")
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
        plt.savefig(f"../img/ewf/heatmaps/{task_name}_task_{i}_train.pdf")
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
        plt.savefig(f"../img/ewf/heatmaps/{task_name}_task_{i}_test.pdf")
        plt.close()

    print("done")
    if args.store_checkpoint:
        # Save final example wise accuracies of task 1 examples
        final_acc_train = epoch_wise_classification_matrices[0][:, -1, -1].cpu().numpy()
        final_acc_test = (
            epoch_wise_classification_matrices_test[0][:, -1, -1].cpu().numpy()
        )

        np.save(f"checkpoints/ewf_final_acc_seed_{args.seed}_train.npy", final_acc_train)
        np.save(f"checkpoints/ewf_final_acc_seed_{args.seed}_test.npy", final_acc_test)

    for i, (losses, accuracies) in enumerate(
        zip(task_test_losses, task_test_accuracies)
    ):
        print(f"Task {i+1} test loss: {losses}")
        print(f"Task {i+1} test accuracy: {accuracies}")