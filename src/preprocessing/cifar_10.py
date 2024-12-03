import pickle
import numpy as np
import argparse
import os

from utils.cifar_n import cifar_n

parser = argparse.ArgumentParser()
parser.add_argument(
    "--n", action="store", type=int, default=5, help="Number of tasks"
)
args = parser.parse_args()

n = args.n

data = []
labels = []

for i in range(1,6):
    with open(f'../data/cifar-10-batches-py/data_batch_{i}', 'rb') as fo:
        batch = pickle.load(fo, encoding='bytes')

        data.append(batch[b'data'])
        labels.append(batch[b'labels'])

data = np.concatenate(data).reshape(-1, 3, 32, 32)
labels = np.concatenate(labels).reshape(-1)

with open(f'../data/cifar-10-batches-py/test_batch', 'rb') as fo:
    batch = pickle.load(fo, encoding='bytes')

    test_data = batch[b'data'].reshape(-1, 3, 32, 32)
    test_labels = np.array(batch[b'labels']).reshape(-1)

tasks = cifar_n(data, labels, n = n)
test_tasks = cifar_n(test_data, test_labels, n = n)

os.mkdir(f'../data/cifar-10-{n}')
os.mkdir(f'../data/cifar-10-{n}/train')
os.mkdir(f'../data/cifar-10-{n}/test')

for i, task in enumerate(tasks):
    with open(f'../data/cifar-10-{n}/train/task_{i+1}', 'wb') as f:
        pickle.dump(task, f)

for i, task in enumerate(test_tasks):
    with open(f'../data/cifar-10-{n}/test/task_{i+1}', 'wb') as f:
        pickle.dump(task, f)


