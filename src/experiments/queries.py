import pandas as pd

path_data = 'data/msmarco-test2019-queries.tsv'

def get_data():
    return pd.read_csv(path_data, sep='\t', header=None, names=['id', 'query'])