import torch

def mc_dropout_inference(model, dataloader, device, num_samples=50, probs=True):
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