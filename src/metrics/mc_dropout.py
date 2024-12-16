import torch
import pandas as pd


def mc_dropout_inference(
    model, dataloader, device, num_samples=100, classification=True
):
    """
    Perform MC Dropout inference over a dataloader to compute mean, variances and uncertainty measures of predictions.
    """

    model.train()   # Enable dropout during inference
    all_predictions = []

    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            batch_predictions = []

            for _ in range(num_samples):
                outputs = model(inputs)  # [batch_size, num_classes/output_dim]
                if classification:
                    outputs = torch.nn.functional.softmax(outputs, dim=1)
                batch_predictions.append(outputs)

            batch_predictions = torch.stack(batch_predictions)
            all_predictions.append(batch_predictions)

    all_predictions = torch.stack(
        all_predictions, dim=1
    )  # [num_samples, num_examples, num_classes]

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
                "Predicted_Class": predicted_class.cpu().numpy(),
                "Predictive_Entropy": predictive_entropy.cpu().numpy(),
                "Mutual_Information": mutual_information.cpu().numpy(),
                "Variation_Ratio": variation_ratio.cpu().numpy(),
                "Mean_Std_Deviation": mean_std_deviation.cpu().numpy(),
            }
        )
        return df

    return mean_predictions.cpu().numpy().squeeze(), variances.cpu().numpy().squeeze()
