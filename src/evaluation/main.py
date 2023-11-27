import pandas as pd
import numpy as np
import ir_datasets
from ir_measures import nDCG
from index_and_retrieve import sentence_embedding
import os
#import faiss
#from sentence_transformers import SentenceTransformer
from tqdm import tqdm
tqdm.pandas()
#import index_and_retrieve as ir


collections = {'robust04': 'irds:disks45/nocr/trec-robust-2004',
               'trec-covid': 'irds:beir/trec-covid',
               'msmarco-passage': 'irds:msmarco-passage/trec-dl-2019/judged'}

dataset_path = './results/pipeline/'

def main():
    #get list of dataset available
    list_of_dataset_queries = [dataset for dataset in os.listdir(dataset_path) if dataset.endswith('.csv')]
    #read ith-dataset obfuscated queries with
    #to read dataset use pd.read_csv(dataset_path + list_of_dataset[i])

    for dataset_q in list_of_dataset_queries:
        documents_retrieved = sentence_embedding(dataset_path + dataset_q)
        



if __name__ == '__main__':
    main()
