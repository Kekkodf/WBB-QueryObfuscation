import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import multiprocessing as mp
tqdm.pandas()

def calculate_cosine_similarity(model, df):
    # Calculate the cosine similarity using progress_apply for all the rows in the dataframe
    cosine_similarities = df.progress_apply(lambda x: cosine_similarity(model.encode([x['original_query']]), model.encode([x['obfuscated_query_angle']]))[0][0], axis = 1)
    return cosine_similarities

path_to_datasets = './data/final/msmarco-dl19/'

if __name__ == '__main__':
    epsilons = [0.001, 0.001, 0.1, 1, 10, 100, 1000]
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    k = 4
    n = 250

    print(f'Reading datasets for k = {k} and n = {n}...')
    dfs = [pd.read_csv(f'./data/final/msmarco-dl19/dataset-{k}-{n}-{epsilon}_msmarco-dl19.csv', sep = ',') for epsilon in epsilons]

    for index, df in enumerate(dfs):
        dfs[index] = df.groupby('query_id').head(20)

    for index, e in enumerate(epsilons):
        print(f'Starting epsilon: {e}')
        with mp.Pool(30) as pool:
            results = pool.starmap_async(calculate_cosine_similarity, [(model, dfs[index])])
            results = results.get()
            dfs[index]['cosine_similarity'] = results[0]
            dfs[index].to_csv(f'./data/final/msmarco-dl19/cosine_similarity_WBB-{k}-{n}_{e}.csv', index = False)
            print(f'Mean cosine similarity for {e}: {dfs[index]["cosine_similarity"].mean()}')
            print(f'Finished {e}')