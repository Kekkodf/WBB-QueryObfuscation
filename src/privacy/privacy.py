import argparse
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
tqdm.pandas()
from sentence_transformers import SentenceTransformer
from  nltk import word_tokenize, pos_tag
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
sns.set_context('paper')
sns.set_palette('colorblind')
dataset_path = './data/final/'
dataset_queries_path = './data/queries/'

k = 4
n = 6
distribution = ('gamma', (1, 2)) #(name, param_1, ..., param_n)
collection = 'msmarco-passage'

dataset_name = 'dataset-{k}-{n}_{distribution}_{collection}.csv'.format(k=k, n=n, distribution=distribution, collection=collection)

#read the dataset
query_dataset = pd.read_csv(dataset_path + dataset_name, sep=",")

models = {"tasb": 'sentence-transformers/msmarco-distilbert-base-tas-b',
          "contriever": "facebook/contriever-msmarco",
          "ance": 'sentence-transformers/msmarco-roberta-base-ance-firstp'}
#encoding with model
model = SentenceTransformer(models['tasb'])

def encode_query(query):
    return model.encode(query)

def jaccard_similarity(str1, str2):
    # Tokenize the phrases into sets of words
    set1 = set(word_tokenize(str1.lower()))
    set2 = set(word_tokenize(str2.lower()))

    # Calculate Jaccard similarity
    intersection_size = len(set1.intersection(set2))
    union_size = len(set1.union(set2))
    similarity = intersection_size / union_size if union_size != 0 else 0

    return similarity

def main():
    # Get unique values from the 'ID' column
    query_dataset = pd.read_csv(dataset_path + dataset_name, sep=",")
    unique_ids = query_dataset['query_id'].unique()

    # Iterate over unique IDs and save each corresponding group to a separate CSV file
    for unique_id in unique_ids:
        group_data = query_dataset[query_dataset['query_id'] == unique_id]
        file_name = f'group_{unique_id}.csv'
        group_data.to_csv(dataset_queries_path + file_name, index=False)
        #print(f'Group {unique_id} saved to {file_name}')

    list_of_types_of_obfuscation = ['obfuscated_query_distance', 'obfuscated_query_angle', 'obfuscated_query_product']
    #og_query = lambda row: encode_query(row['original_query'])
    for type_of_obfuscation in list_of_types_of_obfuscation:
        #read a dataset for a specific query_id
        mins = []
        maxs = []
        averages = []
        df = pd.DataFrame()
        for unique_id in unique_ids:
            file_name = f'group_{unique_id}.csv'
            query_dataset = pd.read_csv(dataset_queries_path + file_name, sep=",")
            #fusion with df
            df = pd.concat([df, query_dataset])
        #compute Jaccard similarity gruping by query_id
        df['jaccard_similarity'] = df.progress_apply(lambda row: jaccard_similarity(row['original_query'], row[type_of_obfuscation]), axis=1)

        #plot a boxplot of Jaccard similarity for each query_id
        plt.figure(figsize=(10, 8))
        sns.boxplot(x='query_id', y='jaccard_similarity', data=df)
        plt.title(f'Jaccard similarity for {type_of_obfuscation}')
        plt.xticks([])
        plt.savefig(f'./plots/{type_of_obfuscation}.png')
        plt.show()
        plt.close()            
        
if __name__ == "__main__":
    main()