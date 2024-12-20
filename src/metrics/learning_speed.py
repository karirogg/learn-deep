import torch


def calculate_learning_speed(epoch_wise_classification_matrices):
    """
    Calculate the learning speed of examples from epoch wise classification matrices each of shape [num_examples_per_task, num_tasks, epochs_per_task].

    Output: List of length num_tasks, each entry a vector of size [num_examples_per_task]
    """

    learning_speeds = []

    for i, class_matrix in enumerate(epoch_wise_classification_matrices):
        class_matrix = class_matrix[:, i, :]
        learning_speeds.append(class_matrix.mean(dim=1).cpu().numpy())

    return learning_speeds
