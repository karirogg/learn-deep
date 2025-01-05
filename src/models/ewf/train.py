import argparse
import matplotlib.pyplot as plt
import os
import torch
import wandb
import numpy as np
import pdb
import json
import pickle

from replay_buffers.replay import Replay
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

from models.ewf.mse import mse
from models.cifar.evaluate import evaluate
from models.ewf.til_nn import TaskILNN

from models.training_loop import training_loop

# from metrics.vog import compute_VoG, visualize_VoG

from utils.fix_seed import fix_seed


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", action="store", type=int, default=10, help="Number of epochs")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--replay-buffer", action="store", type=str, default=None, help="Replay buffer strategy")
    parser.add_argument("--replay_weights", type=str, default="{}") # example: --replay_weights '{"vog": 1.0, "learning_speed": 1.0, "mc_entropy": 0.0, "mc_mutual_information": 0.0, "mc_variation_ratio": 0.0, "mc_mean_std": 0.0, "mc_variance": 1.0}' NOTE: all mc_weights except mc_variance should have zero weight for regression
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    fix_seed(args.seed)
    n = 4
    epochs_per_task = args.epochs

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    model = TaskILNN(num_tasks=n)
    model.to(device)

    model_config = {"dataset" : 'EuropeWindFarm'}

    wandb.init(project="learn-deep", config=model_config, mode="online" if args.wandb else "disabled")

    # TODO: Possibly use lower learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # TODO: After we have verified that task incremental learning works well, we will want to use SGD with momentum and a scheduler
    # optimizer = torch.optim.SGD(
    #    model.parameters(), lr=0.02, momentum=0.9, weight_decay=4e-4
    # )
    # scheduler = CosineAnnealingLR(optimizer, T_max=epochs_per_task * n)
    
    criterion = torch.nn.MSELoss()

    batch_size = 128
    replay_batch_size = 8
    num_checkpoints = 5

    train_tasks = []
    test_tasks = []

    for i in range(1, n+1):
        with open(f'../data/ewf/train/task_{i}', 'rb') as f:
            task_X, task_y = pickle.load(f)

            task_train_tensor = torch.tensor(task_X, dtype=torch.float32)
            task_labels = torch.tensor(task_y, dtype=torch.float32)

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

            task_test_tensor = torch.tensor(task_X, dtype=torch.float32)
            task_test_labels = torch.tensor(task_y, dtype=torch.float32)

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


    replay_params = {"remove_lower_percent" : 20, "remove_upper_percent" : 20}
    replay_weights = json.loads(args.replay_weights)
    replay_buffer = Replay(replay_params, strategy=args.replay_buffer, batch_size=replay_batch_size, num_tasks=n, weights=replay_weights)
    
    if args.replay_buffer:
        print("running with replay strategy:", args.replay_buffer)
    else:
        print("WARNING: no valid replay strategy provided - running without")

    task_test_losses, task_test_accuracies, epoch_wise_classification_matrices = (
        training_loop(
            train_tasks=train_tasks,
            test_tasks=test_tasks,
            model=model,
            optimizer=optimizer,
            # scheduler=scheduler,
            criterion=criterion,
            device=device,
            metric=mse,
            evaluate=evaluate,
            replay_buffer=replay_buffer,
            max_replay_buffer_size=5000,
            epochs_per_task=epochs_per_task,
            num_checkpoints=num_checkpoints,
            is_classification=False
        )
    )

    wandb.finish()

    print("creating plots...")
    task_name = f'ewf_epochs_{epochs_per_task}_replay_{args.replay_buffer}'

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
    plt.ylabel('MSE')

    plt.xlim(0, len(train_tasks) * epochs_per_task)

    plt.savefig(f'../img/task_progression/{task_name}.png')
    plt.close()

    for i, task in enumerate(train_tasks):
        # plot heatmap of classification accuracy per sample

        order = torch.argsort(torch.mean(epoch_wise_classification_matrices[i][:,i,:], axis=1), descending=False)

        concat_task_progression = torch.cat([epoch_wise_classification_matrices[i][:, j, :] for j in range(n)], dim=1)[order]

        plt.figure(figsize=(5 * n, 5))
        plt.imshow(concat_task_progression.cpu().numpy(), cmap='cividis', interpolation='nearest', aspect='auto')

        plt.savefig(f'../img/heatmaps/{task_name}.png')

    print("done")

    for i, (losses, accuracies) in enumerate(
        zip(task_test_losses, task_test_accuracies)
    ):
        print(f"Task {i+1} test loss: {losses}")
        print(f"Task {i+1} test accuracy: {accuracies}")
