import pandas as pd
import os 
import pickle
import torch
import numpy as np

# ewf_data = []

# # loop over all csv files as pandas df in ../data/ewf-raw and append them to ewf_data
# for file in os.listdir('../data/ewf-raw'):
#     if file.endswith('.csv'):
#         with open(f'../data/ewf-raw/{file}', 'r') as f:
#             ewf_data.append(pd.read_csv(f))

# # concat data
# ewf_data = pd.concat(ewf_data)

ewf_data_list = []

for i in range(1,12):
    if i == 2:
        continue

    ewf_data_list.append(pd.read_csv(f'../ewf-raw/wf{i}.csv'))

ewf_data = pd.concat(ewf_data_list)

ewf_test_data = pd.read_csv('../ewf-raw/wf2.csv')

# only include the first year in the data
ewf_data = ewf_data[ewf_data['Time'].str[:4] == '0000']
ewf_test_data = ewf_test_data[ewf_test_data['Time'].str[:4] == '0000']

ewf_data.sort_values(by='Time', inplace=True)
ewf_test_data.sort_values(by='Time', inplace=True)

# split data evenly into 4 tasks
task_size = len(ewf_data) // 4

# create tasks
tasks = []
split_dates = []
for i in range(4):
    relevant_data = ewf_data.iloc[i * task_size : (i + 1) * task_size]

    split_dates.append(relevant_data['Time'].iloc[0])

    X = relevant_data.drop(columns=['PowerGeneration', 'Time'])
    y = relevant_data['PowerGeneration']

    tasks.append((np.array(X), np.array(y).reshape(-1)))

split_dates.append(ewf_data['Time'].iloc[-1])

print(split_dates)

test_tasks = []

for i in range(4):
    relevant_data = ewf_test_data[(ewf_test_data['Time'] >= split_dates[i]) & (ewf_test_data['Time'] < split_dates[i+1])]

    X = relevant_data.drop(columns=['PowerGeneration', 'Time'])
    y = relevant_data['PowerGeneration']

    test_tasks.append((np.array(X), np.array(y).reshape(-1)))

if not os.path.exists('../data'):
    os.mkdir('../data')

if not os.path.exists(f'../data/ewf'):
    os.mkdir(f'../data/ewf')
    os.mkdir(f'../data/ewf/train')
    os.mkdir(f'../data/ewf/test')

for i, task in enumerate(tasks):
    with open(f'../data/ewf/train/task_{i+1}', 'wb') as f:
        pickle.dump(task, f)

for i, task in enumerate(test_tasks):
    with open(f'../data/ewf/test/task_{i+1}', 'wb') as f:
        pickle.dump(task, f)



