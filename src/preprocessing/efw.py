import sys
import os

import pandas as pd

efw_data = []

# loop over all csv files as pandas df in ../data/efw-raw and append them to efw_data
for file in os.listdir('../data/efw-raw'):
    if file.endswith('.csv'):
        with open(f'../data/efw-raw/{file}', 'r') as f:
            efw_data.append(pd.read_csv(f))

# concat data
efw_data = pd.concat(efw_data)


