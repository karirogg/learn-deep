import numpy as np

# From Goldilocks paper: Unless otherwise specified, classes are divided into tasks according to the original order they appeared in the dataset (often alphabetically).
def cifar_n(data, labels, n):
    unique_labels = np.unique(labels)

    k = len(unique_labels) // n

    tasks = []

    for i in range(n):
        task_data = []
        task_labels = []
        for j in range(i*k, (i+1)*k):
            task_data.append(data[labels == unique_labels[j]])
            task_labels.append(labels[labels == unique_labels[j]])
        
        task_data = np.concatenate(task_data)
        task_labels = np.concatenate(task_labels)

        tasks.append((task_data, task_labels, np.unique(task_labels)))

    return tasks

