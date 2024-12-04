import torch
from typing import Callable
import pickle
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from torch import nn
import argparse
import pdb
import numpy as np


def training_loop(
        train_tasks: list[torch.utils.data.DataLoader], 
        test_tasks: list[torch.utils.data.DataLoader], 
        model: torch.nn.Module, 
        optimizer: torch.optim.Optimizer, 
        criterion: torch.nn.Module, 
        metric: Callable[[float], float],
        evaluate: Callable[[torch.nn.Module, torch.utils.data.DataLoader, torch.nn.Module, Callable[[float], float]], tuple[float, float]],
        epochs_per_task: int
    ) -> list[float]:
    """
    The function trains the model on each of the different tasks sequentially using continual learning and uses a replay buffer to store the data from the previous tasks.
    """

    # Ignore replay buffer for now
    # replay_buffer = []

    task_test_losses = []
    task_test_accuracies = []

    for _ in range(len(train_tasks)):
        task_test_losses.append([])
        task_test_accuracies.append([])

    for i, task in enumerate(train_tasks):
        print(f'Training on task {i + 1}')
        for _ in tqdm(range(epochs_per_task)):
            for inputs, labels in task:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                # pdb.set_trace()

        print(f'Results after training on task {i + 1}')

        for i, task_test in enumerate(test_tasks):
            test_loss, test_accuracy = evaluate(model, task_test, criterion, metric)

            task_test_losses[i].append(test_loss)
            task_test_accuracies[i].append(test_accuracy)

            print(f'Task {i+1} test loss: {test_loss}')
            print(f'Task {i+1} test accuracy: {test_accuracy}')


    return task_test_losses, task_test_accuracies

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
    
num_classes = 10

model = nn.Sequential(torch.hub.load('pytorch/vision:v0.10.0', 'squeezenet1_0', pretrained=True), nn.LazyLinear(num_classes)).to(device)

# Define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

# Define the loss function
criterion = torch.nn.CrossEntropyLoss()

# Define the evaluation metric
def accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    return (output.argmax(1) == target).float().mean().item()

# Define the evaluation function
def evaluate(model: torch.nn.Module, test_loader: torch.utils.data.DataLoader, criterion: torch.nn.Module, metric: Callable[[float], float]) -> tuple[float, float]:
    model.eval()
    test_loss = 0
    test_accuracy = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            test_loss += criterion(outputs, labels).item()
            test_accuracy += metric(outputs, labels)

    test_loss /= len(test_loader)
    test_accuracy /= len(test_loader)

    return test_loss, test_accuracy

# Define the number of epochs per task
epochs_per_task = 10

# Define the training and testing tasks
train_tasks = []
test_tasks = []

parser = argparse.ArgumentParser()
parser.add_argument(
    "--n", action="store", type=int, default=5, help="Number of tasks"
)
args = parser.parse_args()

n = args.n

print(n)

for i in range(1, n+1):  
    with open(f'../data/cifar-10-{n}/train/task_{i}', 'rb') as f:
        task_train, task_labels = pickle.load(f)

    with open(f'../data/cifar-10-{n}/test/task_{i}', 'rb') as f:
        task_test, task_test_labels = pickle.load(f)

    train_tasks.append(DataLoader(TensorDataset(torch.tensor(task_train, dtype=torch.float32, device=device), torch.tensor(task_labels, dtype=torch.long, device=device)), batch_size=64, shuffle=True))
    test_tasks.append(DataLoader(TensorDataset(torch.tensor(task_test, dtype=torch.float32, device=device), torch.tensor(task_test_labels, dtype=torch.long, device=device)), batch_size=64, shuffle=False))

task_test_losses, task_test_accuracies = training_loop(train_tasks, test_tasks, model, optimizer, criterion, accuracy, evaluate, epochs_per_task)

for i, (losses, accuracies) in enumerate(zip(task_test_losses, task_test_accuracies)):
    print(f'Task {i+1} test loss: {losses}')
    print(f'Task {i+1} test accuracy: {accuracies}')