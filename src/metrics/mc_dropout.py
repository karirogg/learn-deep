import torch


def mc_dropout_inference(
    model, dataloader, device, num_samples=50, classification=True
):
    """
    Perform MC Dropout inference over a dataloader to compute mean and variance of predictions.
    """

    model.train()   # Enable dropout during inference
    mean_predictions = []
    variances = []

    if classification:
        expected_entropy = []

    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            batch_predictions = []

            for _ in range(num_samples):
                if classification:
                    outputs = model(inputs)     # [batch_size, num_classes]
                    outputs = torch.nn.functional.softmax(outputs, dim=1)
                else:
                    outputs = model(inputs)

                batch_predictions.append(outputs)

            batch_predictions = torch.stack(batch_predictions)  # [num_samples, batch_size, num_classes]
            batch_mean = batch_predictions.mean(dim=0)  # [batch_size, num_classes]
            batch_variance = batch_predictions.var(dim=0)

            if classification:
                per_sample_entropy = -torch.sum(
                    (batch_predictions + 1e-9) * torch.log(batch_predictions + 1e-9),
                    dim=2,
                )  # Shape: [num_samples, batch_size]
                expected_entropy.append(
                    per_sample_entropy.mean(dim=0)
                )  # Shape: [batch_size]

            mean_predictions.append(batch_mean)
            variances.append(batch_variance)

    mean_predictions = torch.cat(mean_predictions, dim=0)   # [num_examples, num_classes]
    variances = torch.cat(variances, dim=0)

    if classification:
        expected_entropy = torch.cat(expected_entropy, dim=0)  # [num_examples]
        predictive_entropy = -torch.sum(
            (mean_predictions + 1e-9) * torch.log(mean_predictions + 1e-9), dim=1
        )
        mutual_information = predictive_entropy - expected_entropy
        mean_std_deviation = torch.mean(torch.sqrt(variances), dim=1)

        return (
            mean_predictions,
            variances,
            predictive_entropy,
            mutual_information,
            mean_std_deviation,
        )

    return mean_predictions, variances
