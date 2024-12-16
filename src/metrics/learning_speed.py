import torch


def calculate_learning_speed(epoch_wise_classification_matrices):

    # Assuming a list of epoch wise classification matrices of size [num_examples, E]

    learning_speeds = []

    for class_matrix in epoch_wise_classification_matrices:

        learning_speeds.append(class_matrix.mean(dim=1).cpu().numpy())

    return learning_speeds
