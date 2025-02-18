import pickle
import numpy as np
import os
import argparse

from utils.cifar_n import cifar_n

parser = argparse.ArgumentParser()
parser.add_argument(
    "--n", action="store", type=int, default=5, help="Number of tasks"
)
args = parser.parse_args()

n = args.n

with open(f'../raw_data/cifar-100-python/train', 'rb') as fo:
    batch = pickle.load(fo, encoding='bytes')

    data = batch[b'data'].reshape(-1, 3, 32, 32)
    labels = np.array(batch[b'fine_labels']).reshape(-1)

with open(f'../raw_data/cifar-100-python/test', 'rb') as fo:
    batch = pickle.load(fo, encoding='bytes')

    test_data = batch[b'data'].reshape(-1, 3, 32, 32)
    test_labels = np.array(batch[b'fine_labels']).reshape(-1)

tasks = cifar_n(data, labels, n)
test_tasks = cifar_n(test_data, test_labels, n)

if not os.path.exists('../data'):
    os.mkdir('../data')

if not os.path.exists(f'../data/cifar-100-{n}'):
    os.mkdir(f'../data/cifar-100-{n}')
    os.mkdir(f'../data/cifar-100-{n}/train')
    os.mkdir(f'../data/cifar-100-{n}/test')

for i, task in enumerate(tasks):
    with open(f'../data/cifar-100-2/train/task_{i+1}', 'wb') as f:
        pickle.dump(task, f)

for i, task in enumerate(test_tasks):
    with open(f'../data/cifar-100-2/test/task_{i+1}', 'wb') as f:
        pickle.dump(task, f)

print('Preprocessing done!')
