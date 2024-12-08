import torch
from typing import Callable
import pickle
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from torch import nn
import argparse
import pdb
import numpy as np
from PIL import Image


def training_loop(
        train_tasks: list[torch.utils.data.DataLoader], 
        test_tasks: list[torch.utils.data.DataLoader], 
        model: torch.nn.Module, 
        optimizer: torch.optim.Optimizer, 
        criterion: torch.nn.Module, 
        metric: Callable[[float], float],
        evaluate: Callable[[torch.nn.Module, torch.utils.data.DataLoader, torch.nn.Module, Callable[[float], float]], tuple[float, float]],
        epochs_per_task: int,
        num_checkpoints : int
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
        grad_matrices = []
        checkpoints = np.linspace(0, epochs_per_task, num_checkpoints, endpoint=False, dtype=np.int32)
        # pdb.set_trace()
        # input_images = torch.cat([img for img, _ in task], dim=0)
        # epoch_labels = torch.cat([labels for _, labels in task], dim=0)
        input_images, epoch_labels = map(torch.cat, zip(*[(img, labels) for img, labels in task]))

        for epoch in tqdm(range(epochs_per_task)):
            grad_matrices_epoch = []
            for inputs, labels in task:
                optimizer.zero_grad()
                inputs.retain_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                if epoch in checkpoints:
                    pixel_grads = inputs.grad.mean(axis=1)
                    grad_matrices_epoch.append(pixel_grads.clone())
                optimizer.step()
            if epoch in checkpoints:
                grad_matrices.append(torch.concat(grad_matrices_epoch, axis=0))
        grad_variances = compute_VoG(grad_matrices, epoch_labels, checkpoints)
        visualize_VoG(grad_variances, input_images, epoch_labels)
        
        print(f'Results after training on task {i + 1}')

        for i, task_test in enumerate(test_tasks):
            test_loss, test_accuracy = evaluate(model, task_test, criterion, metric)

            task_test_losses[i].append(test_loss)
            task_test_accuracies[i].append(test_accuracy)

            print(f'Task {i+1} test loss: {test_loss}')
            print(f'Task {i+1} test accuracy: {test_accuracy}')


    return task_test_losses, task_test_accuracies

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

def compute_VoG(grad_matrices, epoch_labels, checkpoints):
    # calculate VoG
    grad_matrices = torch.stack(grad_matrices, axis=0)
    grad_means = torch.mean(grad_matrices, axis=0)
    grad_variances = np.sqrt(1 / len(checkpoints)) * torch.sum(torch.pow(grad_matrices - grad_means.unsqueeze(0), 2), axis=0)
    grad_variances = grad_variances.mean(axis=[1, 2]) # average over pixels    
    # normalise per class
    normalised_grad_variances = []
    for l in epoch_labels.unique():
        class_grad_variances = grad_variances[epoch_labels == l]
        normalized_class_values = (class_grad_variances - class_grad_variances.mean()).abs() / class_grad_variances.std()
        normalised_grad_variances.append(normalized_class_values)
    return normalised_grad_variances

def visualize_VoG(grad_variances, input_images, labels, num_imgs=10):
    for i, l in enumerate(labels.unique()):
        _, top_idcs = torch.topk(grad_variances[i], k=num_imgs, largest=True, sorted=False)
        _, bottom_idcs = torch.topk(grad_variances[i], k=num_imgs, largest=False, sorted=False)
        top_imgs = input_images[labels == l][top_idcs, :, :, :]
        bottom_imgs = input_images[labels == l][bottom_idcs, :, :, :]
        for j, (t_img, b_img) in enumerate(zip(top_imgs, bottom_imgs)):
            Image.fromarray(t_img.permute(1, 2, 0).byte().cpu().detach().numpy()).save(f"../visus/vog/top_picks/class_{l}_pick_{j}.png")
            Image.fromarray(b_img.permute(1, 2, 0).byte().cpu().detach().numpy()).save(f"../visus/vog/bottom_picks/class_{l}_pick_{j}.png")
    return

def mc_dropout_inference(model, dataloader, num_samples=50, device, probs=True):
    """
    Perform MC Dropout inference over a dataloader to compute the variance of predictions.
    """

    model.train()   # Enable dropout during inference
    mean_predictions = []
    variances = []
    
    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            batch_predictions = []
            
            for _ in range(num_samples):
                if probs:
                    outputs = model(inputs)     # [batch_size, num_classes]
                    outputs = torch.nn.functional.softmax(outputs, dim=1)
                else:
                    outputs = model(inputs)

                batch_predictions.append(outputs)
            
            batch_predictions = torch.stack(batch_predictions)  # [num_samples, batch_size, num_classes]
            batch_mean = batch_predictions.mean(dim=0)  # [batch_size, num_classes]
            batch_variance = batch_predictions.var(dim=0)  
            
            mean_predictions.append(batch_mean)
            variances.append(batch_variance)
    
    mean_predictions = torch.cat(mean_predictions, dim=0)   # [num_examples, num_classes]
    variances = torch.cat(variances, dim=0)

    if probs:
        predictive_entropy = -torch.sum((mean_predictions + 1e-9) * torch.log(mean_predictions + 1e-9), dim=1)  # [num_examples]
        weighted_variances = torch.sum(mean_predictions * variances, dim=1)

        return mean_predictions, variances, predictive_entropy, weighted_variances
    
    return mean_predictions, variances


seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
    
num_classes = 10

squeeze_net = torch.hub.load('pytorch/vision:v0.10.0', 'squeezenet1_1', pretrained=True)

model = squeeze_net
model.classifier = nn.Sequential( # modify classifier layer to return correct number of classes
    nn.Dropout(p=0.5, inplace=False),
    nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1)),
    nn.ReLU(inplace=True),
    nn.AdaptiveAvgPool2d(output_size=(1, 1))
)
model = model.to(device)

# Define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

# Define the loss function
criterion = torch.nn.CrossEntropyLoss()

# Define the number of epochs per task
epochs_per_task = 10
batch_size = 100
num_checkpoints = 5

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

    train_tasks.append(DataLoader(TensorDataset(torch.tensor(task_train, dtype=torch.float32, device=device, requires_grad=True), torch.tensor(task_labels, dtype=torch.long, device=device)), batch_size=batch_size, shuffle=True))
    test_tasks.append(DataLoader(TensorDataset(torch.tensor(task_test, dtype=torch.float32, device=device, requires_grad=True), torch.tensor(task_test_labels, dtype=torch.long, device=device)), batch_size=batch_size, shuffle=False))

task_test_losses, task_test_accuracies = training_loop(train_tasks, test_tasks, model, optimizer, criterion, accuracy, evaluate, epochs_per_task, num_checkpoints)

for i, (losses, accuracies) in enumerate(zip(task_test_losses, task_test_accuracies)):
    print(f'Task {i+1} test loss: {losses}')
    print(f'Task {i+1} test accuracy: {accuracies}')