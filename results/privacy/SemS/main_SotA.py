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
    cosine_similarities = df.progress_apply(lambda x: cosine_similarity(model.encode([x['text']]), model.encode([x['obfuscatedText']]))[0][0], axis = 1)
    return cosine_similarities

# Set the paths
path_to_datasets = './data/final/msmarco-dl19/'
path_to_results = './plots/'

if __name__ == '__main__':
    epsilons = [1, 5, 10, 12.5, 15, 17.5, 20, 50]
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    mechansim = 'Mhl'
    dfs = [pd.read_csv(f'./data/final/SotA/{mechansim}/obfuscated_Mahalanobis_{epsilon}.csv', sep = ',') for epsilon in epsilons]
    #mechansim = 'CMP'
    #dfs = [pd.read_csv(f'./data/final/SotA/{mechansim}/obfuscated_{mechansim}_{epsilon}.csv', sep = ',') for epsilon in epsilons]
    
    for index, e in enumerate(epsilons):
        print(f'Starting epsilon: {e}')

        with mp.Pool(40) as pool:
            results = pool.starmap_async(calculate_cosine_similarity, [(model, dfs[index])])
            results = results.get()
    #save the results in a csv file for each epsilon
            dfs[index]['cosine_similarity'] = results[0]
            dfs[index].to_csv(f'./data/final/SotA/{mechansim}/cosine_similarity_{mechansim}_{e}.csv', index = False)
            print(f'Mean cosine similarity for {e}: {dfs[index]["cosine_similarity"].mean()}')
            print(f'Finished {e}')


