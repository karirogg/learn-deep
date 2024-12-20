import argparse
import matplotlib.pyplot as plt
import os
import torch
import wandb
import numpy as np
import pdb

from replay_buffers.replay import Replay

from models.cifar.accuracy import accuracy
from models.cifar.evaluate import evaluate
from models.cifar.task_preprocessing import preprocess_cifar
from models.training_loop import training_loop
from models.custom_cnn import CIFAR_CNN
from torch.optim.lr_scheduler import CosineAnnealingLR

# from metrics.vog import compute_VoG, visualize_VoG

from utils.fix_seed import fix_seed


if __name__ == "__main__":
    fix_seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument("--n", action="store", type=int, default=2, help="Number of tasks")
    parser.add_argument("--epochs", action="store", type=int, default=10, help="Number of epochs")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--classes", action="store", type=int, default=10, help="Number of classes")
    parser.add_argument("--replay-buffer", action="store", type=str, default=None, help="Replay buffer strategy")

    args = parser.parse_args()
    n = args.n
    epochs_per_task = args.epochs
    num_classes = args.classes

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    model_config = {"num_classes" : num_classes}

    model = torch.hub.load("pytorch/vision:v0.10.0", "squeezenet1_1", pretrained=False)
    model.features[0] = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
    model.features[2] = torch.nn.MaxPool2d(kernel_size=2, stride=1, ceil_mode=True)
    model.features[5] = torch.nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
    model.features[8] = torch.nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
    model.classifier[1] = torch.nn.Conv2d(512, num_classes, kernel_size=1)
    model = model.to(device)
    wandb.init(project="learn-deep", config=model_config, mode="online" if args.wandb else "disabled")

    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.02, momentum=0.9, weight_decay=4e-4
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs_per_task)
    criterion = torch.nn.CrossEntropyLoss()

    batch_size = 128
    num_checkpoints = 5

    train_tasks, test_tasks, unique_labels = preprocess_cifar(num_classes, n, batch_size, device)

    replay_buffer_strategy = None

    replay_params = {"remove_lower_percent" : 20, "remove_upper_percent" : 20}
    replay = Replay(replay_params)
    replay_strategies = {"uniform" : replay.uniform, "simple_sorted" : replay.simple_sorted}
    replay_buffer_strategy = replay_strategies.get(args.replay_buffer, None)
    if replay_buffer_strategy:
        print(args.replay_buffer)
    else:
        print("WARNING: no valid replay strategy provided - running without")

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
        replay_buffer_strategy=replay_buffer_strategy,
        max_replay_buffer_size=5000,
        epochs_per_task=epochs_per_task,
        num_checkpoints=num_checkpoints
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
