import argparse
import matplotlib.pyplot as plt
import os
import torch
import wandb
import numpy as np
import pdb
import json

from replay_buffers.replay import Replay
from torch.optim.lr_scheduler import CosineAnnealingLR

from models.cifar.accuracy import accuracy
from models.cifar.evaluate import evaluate
from models.cifar.task_preprocessing import preprocess_cifar
from models.training_loop import training_loop
from models.cifar.TIL_squeezenet import Task_IL_SqueezeNet

# from metrics.vog import compute_VoG, visualize_VoG

from utils.fix_seed import fix_seed

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--n", action="store", type=int, default=2, help="Number of tasks")
    parser.add_argument("--epochs", action="store", type=int, default=10, help="Number of epochs")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--classes", action="store", type=int, default=10, help="Number of classes")
    parser.add_argument("--replay-buffer", action="store", type=str, default=None, help="Replay buffer strategy")
    parser.add_argument("--replay_weights", type=str, default="{}") # example: --replay_weights '{"vog": 1.0, "learning_speed": 1.0, "predictive_entropy": 1.0, "mutual_information": 1.0, "variation_ratio": 1.0, "mean_std_deviation": 1.0, "mc_variance": 0.0}' NOTE: mc_variance should have zero weight for classification
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--store_checkpoint", action="store_true")
    parser.add_argument("--use_checkpoint", action="store_true")

    args = parser.parse_args()
    fix_seed(args.seed)
    n = args.n
    epochs_per_task = args.epochs
    num_classes = args.classes

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    model_config = {"num_classes" : num_classes}

    model = Task_IL_SqueezeNet(num_classes_per_task = int(num_classes / n), num_tasks=n)
    model.to(device)

    wandb.init(project="learn-deep", config=model_config, mode="online" if args.wandb else "disabled")

    # TODO: Possibly optimize further
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=5e-5)

    criterion = torch.nn.CrossEntropyLoss()

    batch_size = 128
    replay_batch_size = 8
    num_checkpoints = 10

    train_tasks, test_tasks = preprocess_cifar(num_classes, n, batch_size, device)

    replay_params = {"remove_lower_percent" : 20, "remove_upper_percent" : 20}
    replay_weights = json.loads(args.replay_weights)
    replay_buffer = Replay(replay_params, strategy=args.replay_buffer, batch_size=replay_batch_size, num_tasks=n, weights=replay_weights)

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
        max_replay_buffer_size=5000,
        epochs_per_task=epochs_per_task,
        num_checkpoints=num_checkpoints,
        is_classification=True,
        store_checkpoint=args.store_checkpoint,
        use_checkpoint=args.use_checkpoint,
        seed=args.seed,
    )

    wandb.finish()

    print("creating plots...")
    task_name = f'cifar_{num_classes}_n_{n}_epochs_{epochs_per_task}_replay_{args.replay_buffer}_seed_{args.seed}'

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

    if not os.path.exists('../img/'):
        os.mkdir('../img')

    if not os.path.exists('../img/task_progression'):
        os.mkdir('../img/task_progression')
        os.mkdir('../img/heatmaps')

    plt.xlabel('Epoch')
    plt.ylabel('Classification Accuracy')

    plt.xlim(0, len(train_tasks) * epochs_per_task)
    plt.ylim(0, 1)

    plt.savefig(f"../img/task_progression/{task_name}_test.pdf")
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
        plt.savefig(f"../img/heatmaps/{task_name}_task{i}_train.pdf")
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
        plt.savefig(f"../img/heatmaps/{task_name}_task{i}_test.pdf")
        plt.close()

    print("done")
    if args.store_checkpoint:
        # Save final example wise accuracies of task 1 examples
        final_acc_train = epoch_wise_classification_matrices[0][:, -1, -1].cpu().numpy()
        final_acc_test = (
            epoch_wise_classification_matrices_test[0][:, -1, -1].cpu().numpy()
        )

        np.save(f"checkpoints/final_acc_seed_{args.seed}_train.npy", final_acc_train)
        np.save(f"checkpoints/final_acc_seed_{args.seed}_test.npy", final_acc_test)

    for i, (losses, accuracies) in enumerate(
        zip(task_test_losses, task_test_accuracies)
    ):
        print(f"Task {i+1} test loss: {losses}")
        print(f"Task {i+1} test accuracy: {accuracies}")
