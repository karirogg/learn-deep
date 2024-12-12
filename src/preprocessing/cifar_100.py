import pickle
import numpy as np

from utils.cifar_n import cifar_n

with open(f'../raw_data/cifar-100-python/train', 'rb') as fo:
    batch = pickle.load(fo, encoding='bytes')

    data = batch[b'data'].reshape(-1, 3, 32, 32)
    labels = np.array(batch[b'fine_labels']).reshape(-1)

with open(f'../raw_data/cifar-100-python/test', 'rb') as fo:
    batch = pickle.load(fo, encoding='bytes')

    test_data = batch[b'data'].reshape(-1, 3, 32, 32)
    test_labels = np.array(batch[b'fine_labels']).reshape(-1)

tasks = cifar_n(data, labels, n = 2)
test_tasks = cifar_n(test_data, test_labels, n = 2)

for i, task in enumerate(tasks):
    with open(f'../data/cifar-100-2/train/task_{i+1}', 'wb') as f:
        pickle.dump(task, f)

for i, task in enumerate(test_tasks):
    with open(f'../data/cifar-100-2/test/task_{i+1}', 'wb') as f:
        pickle.dump(task, f)
