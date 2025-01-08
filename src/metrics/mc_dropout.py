import torch
import pandas as pd
import pdb
from tqdm import tqdm
import time
import numpy as np
import sys


def mc_dropout_inference(
    model, dataloader, task_id, device, weights, num_samples=100, classification=True, store_checkpoint=False
):
    """
    Perform MC Dropout inference over a dataloader to compute mean, variances and uncertainty measures of predictions.
    """
    dataset_len = len(dataloader.dataset)
    num_classes = (
        model(dataloader.dataset[0][0].unsqueeze(0).to(device), task_id).shape[1]
        if classification
        else 1
    )

    if not store_checkpoint: # skip computation if metrics won't be used
        if classification and (weights == {} or weights["mc_entropy"] + weights["mc_mutual_information"] + weights["mc_variation_ratio"] + weights["mc_mean_std"] == 0):
            df = pd.DataFrame(
                {
                    "Predicted_Class": np.zeros(dataset_len),
                    "Predictive_Entropy": np.zeros(dataset_len),
                    "Mutual_Information": np.zeros(dataset_len),
                    "Variation_Ratio": np.zeros(dataset_len),
                    "Mean_Std_Deviation": np.zeros(dataset_len),
                }
            )
            return df
        if not classification and (weights == {} or weights["mc_variance"] == 0):
            return np.zeros(dataset_len), np.zeros(dataset_len)

    model.train()   # Enable dropout during inference

    # [num_samples, dataset_length, num_classes/output_dim]
    all_predictions = torch.zeros(num_samples, dataset_len, num_classes, device=device)
    all_labels = torch.zeros(dataset_len, device=device, dtype=torch.long)

    with torch.no_grad():
        for inputs, labels, indices in tqdm(
            dataloader, desc="collecting mc dropout metric", file=sys.stderr
        ):
            inputs, labels, indices = (
                inputs.to(device),
                labels.to(device),
                indices.to(device),
            )

            for i in range(num_samples):
                outputs = model(inputs, task_id)  # [batch_size, num_classes/output_dim]
                if classification:
                    outputs = torch.nn.functional.softmax(outputs, dim=1)
                all_predictions[i, indices] = outputs

            all_labels[indices] = labels

    # [dataset_length, num_classes]
    mean_predictions = all_predictions.mean(dim=0)
    variances = all_predictions.var(dim=0)

    if classification:
        predictive_entropy = -torch.sum(
            (mean_predictions + 1e-9) * torch.log(mean_predictions + 1e-9), dim=1
        )

        per_sample_entropies = -torch.sum(
            (all_predictions + 1e-9) * torch.log(all_predictions + 1e-9), dim=2
        )  # [num_samples, dataset_length]
        expected_entropy = per_sample_entropies.mean(dim=0)
        mutual_information = predictive_entropy - expected_entropy

        mean_std_deviation = torch.mean(torch.sqrt(variances), dim=1)

        max_probability, predicted_class = mean_predictions.max(dim=1)
        variation_ratio = 1 - max_probability

        df = pd.DataFrame(
            {
                "Index": np.arange(0, dataset_len, 1),
                "Predicted_Class": predicted_class.cpu().numpy(),
                "Predictive_Entropy": predictive_entropy.cpu().numpy(),
                "Mutual_Information": mutual_information.cpu().numpy(),
                "Variation_Ratio": variation_ratio.cpu().numpy(),
                "Mean_Std_Deviation": mean_std_deviation.cpu().numpy(),
            }
        )
        return df

    return mean_predictions.cpu().numpy().squeeze(), variances.cpu().numpy().squeeze()
