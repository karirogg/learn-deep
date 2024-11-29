import pickle

with open(f'../data/cifar-10-5/train/task_1', 'rb') as fo:
    data, labels = pickle.load(fo, encoding='bytes')

print(data.shape)
print(labels.shape)