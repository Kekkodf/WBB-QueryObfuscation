import numpy as np
import pandas as pd
import os
import logging
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
from nltk import word_tokenize, pos_tag, ngrams
tqdm.pandas()

dataset_path = './data/final/'

k = 4
n = 6
distribution = ('gamma', (1, 2)) #(name, param_1, ..., param_n)
collection = 'msmarco-passage'

dataset_name = 'dataset-counterFitting-{k}-{n}_{distribution}_{collection}.csv'.format(k=k, n=n, distribution=distribution, collection=collection)

#read the dataset
query_dataset = pd.read_csv(dataset_path + dataset_name, sep=",")

models = {"tasb": 'sentence-transformers/msmarco-distilbert-base-tas-b',
          "contriever": "facebook/contriever-msmarco",
          "ance": 'sentence-transformers/msmarco-roberta-base-ance-firstp'}
#encoding with model
model = SentenceTransformer(models['tasb'])

def encode_query(query):
    return model.encode(query)

#compute jaccard similarity
def jaccard_similarity_general(query, obfuscated_query):
    intersection = set(query).intersection(set(obfuscated_query))
    union = set(query).union(set(obfuscated_query))
    try:
        return len(intersection)/len(union)
    except:
        return 0

def jaccard_similarity(query, obfuscated_query):
    #get words to be masked
    query_words = word_tokenize(query)
    query_pos_tags = pos_tag(query_words)
    query_words = [word[0] for word in query_pos_tags if word[1] in ['NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJS']]
    obfuscated_query_words = word_tokenize(obfuscated_query)
    obfuscated_query_pos_tags = pos_tag(obfuscated_query_words)
    obfuscated_query_words = [word[0] for word in obfuscated_query_pos_tags if word[1] in ['NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJS']]
    return jaccard_similarity_general(query_words, obfuscated_query_words)

def bigram_similarity(query, obfuscated_query, n=3):
    #original query
    query_words = word_tokenize(query)
    query_pos_tags = pos_tag(query_words)
    query_words = [word[0] for word in query_pos_tags if word[1] in ['NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJS']]
    n_grams = [ngrams(word, n) for word in query_words]
    #obfuscated query
    obfuscated_query_words = word_tokenize(obfuscated_query)
    obfuscated_query_pos_tags = pos_tag(obfuscated_query_words)
    obfuscated_query_words = [word[0] for word in obfuscated_query_pos_tags if word[1] in ['NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJS']]
    n_grams_obfuscated = [ngrams(word, n) for word in obfuscated_query_words]
    #compute similarity
    similarity = sum(1 for n_gram in n_grams 
                     for n_gram_obfuscated in n_grams_obfuscated 
                     for g in n_gram for g_o in n_gram_obfuscated 
                     if g == g_o)
    return similarity       

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
        #query_dataset['jaccard_similarity_general'] = query_dataset.progress_apply(lambda row: jaccard_similarity_general(row['original_query'], row[type_of_query]), axis=1)
        ##compute mean jaccard similarity
        #mean_jaccard_similarity = query_dataset['jaccard_similarity_general'].mean()
        #print('Mean jaccard similarity for **{type_of_query}**: {mean_jaccard_similarity}'.format(type_of_query=type_of_query, mean_jaccard_similarity=mean_jaccard_similarity))
        #query_dataset['jaccard_similarity_specific'] = query_dataset.progress_apply(lambda row: jaccard_similarity(row['original_query'], row[type_of_query]), axis=1)
        ##compute mean jaccard similarity
        #mean_jaccard_similarity = query_dataset['jaccard_similarity_specific'].mean()
        #print('Mean jaccard similarity for **{type_of_query}**: {mean_jaccard_similarity}'.format(type_of_query=type_of_query, mean_jaccard_similarity=mean_jaccard_similarity))
        ##compute cosine similarity
        #query_dataset['cosine_similarity'] = query_dataset.progress_apply(lambda row: 1 - cosine(og_query(row), encode_query(row[type_of_query])), axis=1)
        ##compute mean cosine similarity
        #mean_cosine_similarity = query_dataset['cosine_similarity'].mean()
        #print('Mean cosine similarity for **{type_of_query}**: {mean_cosine_similarity}'.format(type_of_query=type_of_query, mean_cosine_similarity=mean_cosine_similarity))
        #compute bigram similarity
        query_dataset['bigram_similarity'] = query_dataset.progress_apply(lambda row: bigram_similarity(row['original_query'], row[type_of_query]), axis=1)
        print(query_dataset['bigram_similarity'])
        #compute mean bigram similarity
        mean_bigram_similarity = query_dataset['bigram_similarity'].mean()
        print('Mean bigram similarity for **{type_of_query}**: {mean_bigram_similarity}'.format(type_of_query=type_of_query, mean_bigram_similarity=mean_bigram_similarity))

if __name__ == '__main__':
    main()