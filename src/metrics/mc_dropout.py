import torch
import pandas as pd
import pdb
from tqdm import tqdm
import time


def mc_dropout_inference(
    model, dataloader, task_id, device, weights, num_samples=100, classification=True
):
    """
    Perform MC Dropout inference over a dataloader to compute mean, variances and uncertainty measures of predictions.
    """
    if weights == {} or weights["mc_entropy"] + weights["mc_mutual_information"] + weights["mc_variation_ratio"] + weights["mc_mean_std"] == 0:
        df = pd.DataFrame(
            {
                "Predicted_Class": torch.zeros(len(dataloader.dataset)),
                "Predictive_Entropy": torch.zeros(len(dataloader.dataset)),
                "Mutual_Information": torch.zeros(len(dataloader.dataset)),
                "Variation_Ratio": torch.zeros(len(dataloader.dataset)),
                "Mean_Std_Deviation": torch.zeros(len(dataloader.dataset)),
            }
        )
        return df

    model.train()   # Enable dropout during inference
    all_predictions = []

    with torch.no_grad():
        for inputs, _, _ in tqdm(dataloader, desc="collecting mc dropout metric"):
            inputs = inputs.to(device)
            batch_predictions = []

            for _ in range(num_samples):
                outputs = model(inputs, task_id)  # [batch_size, num_classes/output_dim]
                if classification:
                    outputs = torch.nn.functional.softmax(outputs, dim=1)
                batch_predictions.append(outputs)

            batch_predictions = torch.stack(batch_predictions, dim=0)
            all_predictions.append(batch_predictions)

    # [num_samples, num_examples, num_classes]
    all_predictions = torch.cat(all_predictions, dim=1)

    mean_predictions = all_predictions.mean(dim=0)
    variances = all_predictions.var(dim=0)

    if classification:
        predictive_entropy = -torch.sum(
            (mean_predictions + 1e-9) * torch.log(mean_predictions + 1e-9), dim=1
        )

        per_sample_entropies = -torch.sum(
            (all_predictions + 1e-9) * torch.log(all_predictions + 1e-9), dim=2
        )  # [num_samples, num_examples]
        expected_entropy = per_sample_entropies.mean(dim=0)
        mutual_information = predictive_entropy - expected_entropy

        mean_std_deviation = torch.mean(torch.sqrt(variances), dim=1)

        max_probability, predicted_class = mean_predictions.max(dim=1)
        variation_ratio = 1 - max_probability

        df = pd.DataFrame(
            {
                "Predicted_Class": predicted_class,
                "Predictive_Entropy": predictive_entropy,
                "Mutual_Information": mutual_information,
                "Variation_Ratio": variation_ratio,
                "Mean_Std_Deviation": mean_std_deviation,
            }
        )
        return df

    return mean_predictions.cpu().numpy().squeeze(), variances.cpu().numpy().squeeze()
