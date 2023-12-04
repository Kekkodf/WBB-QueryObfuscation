import numpy as np
import pandas as pd
import os
import logging
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
tqdm.pandas()

dataset_path = './data/final/'

k = 4
n = 6
distribution = ('gamma', (1, 2)) #(name, param_1, ..., param_n)
collection = 'msmarco-passage'

dataset_name = 'dataset-{k}-{n}_{distribution}_{collection}.csv'.format(k=k, n=n, distribution=distribution, collection=collection)

#read the dataset
query_dataset = pd.read_csv(dataset_path + dataset_name, sep=",")

#encoding with model
model = SentenceTransformer('facebook/contriever-msmarco')

def encode_query(query):
    return model.encode(query)

#compute jaccard similarity
def jaccard_similarity(query, obfuscated_query):
    intersection = set(query).intersection(set(obfuscated_query))
    union = set(query).union(set(obfuscated_query))
    return len(intersection)/len(union)

def cosine_distance_matrix(x, y):
    x_expanded = x[:, np.newaxis, :]
    y_expanded = y[np.newaxis, :, :]
    return 1 - np.sum(x_expanded * y_expanded, axis=2) / (np.linalg.norm(x_expanded, axis=2) * np.linalg.norm(y_expanded, axis=2))

def main():
    #define list of possible obfuscation: original_query,obfuscated_query_distance,obfuscated_query_angle,obfuscated_query_product
    list_of_types_of_obfuscation = ['obfuscated_query_distance', 'obfuscated_query_angle', 'obfuscated_query_product']
    og_query = lambda row: encode_query(row['original_query'])
    for type_of_query in list_of_types_of_obfuscation:
        #compute jaccard similarity
        query_dataset['jaccard_similarity'] = query_dataset.progress_apply(lambda row: jaccard_similarity(row['original_query'], row[type_of_query]), axis=1)
        #compute mean jaccard similarity
        mean_jaccard_similarity = query_dataset['jaccard_similarity'].mean()
        print('Mean jaccard similarity for dataset {}, {}: {:.2f}'.format(dataset_name, type_of_query, mean_jaccard_similarity))
        #compute cosine similarity
        query_dataset['cosine_similarity'] = query_dataset.progress_apply(lambda row: cosine(og_query(row), encode_query(row[type_of_query])), axis=1)
        #compute mean cosine similarity
        mean_cosine_similarity = query_dataset['cosine_similarity'].mean()
        print('Mean cosine similarity for dataset {}, {}: {:.2f}'.format(dataset_name, type_of_query, mean_cosine_similarity))

if __name__ == '__main__':
    main()