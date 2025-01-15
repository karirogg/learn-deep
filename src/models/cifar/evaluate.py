import torch
from typing import Callable

def evaluate(model: torch.nn.Module, evaluation_loader: torch.utils.data.DataLoader, criterion: torch.nn.Module, device: torch.device, metric: Callable[[float], float], task_id: int) -> tuple[float, float]:
    '''
    Evaluate the model on an evaluation dataset, considering both aggregated and sample-wise metrics.
    '''
    model.eval()
    test_loss = 0
    test_accuracy = 0

    sample_wise_accuracy = torch.zeros(len(evaluation_loader.dataset)).to(device)

    with torch.no_grad():
        for inputs, labels, indices in evaluation_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs, task_id)
            test_loss += criterion(outputs, labels).item() * len(outputs)

            batch_metric_agg, sample_wise_metric = metric(outputs, labels)

            test_accuracy += batch_metric_agg

            sample_wise_accuracy[indices] = sample_wise_metric

    test_loss /= len(evaluation_loader.dataset)
    test_accuracy /= len(evaluation_loader.dataset)

    return test_loss, test_accuracy, sample_wise_accuracy
