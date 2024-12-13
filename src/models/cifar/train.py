import argparse
import matplotlib.pyplot as plt
import os
import torch
import wandb
import numpy as np

from replay_buffers.uniform_replay_buffer import uniform_replay_buffer

from models.cifar.accuracy import accuracy
from models.cifar.evaluate import evaluate
from models.cifar.task_preprocessing import preprocess_cifar
from models.training_loop import training_loop
from models.custom_cnn import CIFAR_CNN

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
    
    model_config = {"num_classes" : 10}
    model = CIFAR_CNN(cfg=model_config)
    model = model.to(device)
    wandb.init(project="learn-deep", config=model_config, mode="online" if args.wandb else "disabled")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    criterion = torch.nn.CrossEntropyLoss()

    batch_size = 64
    num_checkpoints = 5

    train_tasks, test_tasks, unique_labels = preprocess_cifar(num_classes, n, batch_size, device)

    replay_buffer_strategy = None

    if args.replay_buffer == "uniform":
        replay_buffer_strategy = uniform_replay_buffer

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
