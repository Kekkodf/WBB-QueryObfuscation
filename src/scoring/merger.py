import pandas as pd
import os

path_to_datasets = './data/'
path_to_results = './data/'

#read all csv files in path_to_datasets
list_of_datasets = [dataset for dataset in os.listdir(path_to_datasets) if dataset.endswith('.csv')]

#merge all datasets
df = pd.concat([pd.read_csv(path_to_datasets + dataset, sep=",") for dataset in list_of_datasets])
#order by query_id
df = df.sort_values(by=['query_id'])
#reset index
df = df.reset_index(drop=True)

#define obfuscation parameters
k = 3
n = 8
distribution = ('gamma', (1, 2)) #(name, param_1, ..., param_n)
collection = 'msmarco-passage'

dataset_name = 'dataset-{k}-{n}_{distribution}_{collection}.csv'.format(k=k, n=n, distribution=distribution, collection=collection)
#save merged dataset
df.to_csv(path_to_results + dataset_name, index=False, header=True, sep=',')