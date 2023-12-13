import pandas as pd
import numpy as np
import ir_datasets
import ir_measures
from sentence_transformers import SentenceTransformer
from multiprocessing import Pool
import faiss
from tqdm import tqdm
tqdm.pandas()

collections = {'robust04': 'disks45/nocr/trec-robust-2004',
                'trec-covid': 'beir/trec-covid',
                'msmarco-passage': 'msmarco-passage/trec-dl-2019/judged'}

m2hf = {"tasb": "sentence-transformers/msmarco-distilbert-base-tas-b",
        "contriever": "facebook/contriever-msmarco",
        "ance": "sentence-transformers/msmarco-roberta-base-ance-firstp"}
#define dataset

# Global variables
model = m2hf["contriever"]  # Your model retrieval
query_dataset_path = './data/dataset_obfuscation.csv'  # Your dataset of queries
document_collection = collections['msmarco-passage']  # Your collection of documents

# Load the dataset of documents
docs_dataset = ir_datasets.load(document_collection)

# Load the dataset of queries
query_dataset = pd.read_csv(query_dataset_path, sep=',')

#indexes and mapper
index_filename = '../../../../ssd/data/faggioli/24-ECIR-FF/data/indexes/msmarco-passage/faiss/contriever/contriever.faiss'
index = faiss.read_index(index_filename)
mapper = list(map(lambda x: x.strip(), open('../../../../ssd/data/faggioli/24-ECIR-FF/data/indexes/msmarco-passage/faiss/contriever/contriever.map', "r").readlines()))
    
#STEPS:
#1. get top k documents most similar to obf_queries
#2. compute dot product between top_k docs and original queries
#3. sort to get the most similar documents to the original query
#4. measure Precision@k, Recall@k, nDCG@k

#-------- Step#1: get top k documents most similar to obf_queries --------#
def top_k(docs, queries):
    '''
    Retrieve the top k most similar documents for a given obfuscated query.
    :param docs: vectorized documents
    :param queries: vectorized queries
    '''


def main():
    #get column with obfuscated queries distance
    obf_queries_distance = query_dataset['obfuscated_query_distance'].astype(str)
    #get column with obfuscated queries angle
    obf_queries_angle = query_dataset['obfuscated_query_angle'].astype(str)
    #get column with obfuscated queries product
    obf_queries_product = query_dataset['obfuscated_query_product'].astype(str)
    #get column with original queries
    original_queries = query_dataset['original_query'].astype(str)

    #vectorize the obfuscated queries
    obf_queries_distance_vector = model.encode(obf_queries_distance)
    obf_queries_angle_vector = model.encode(obf_queries_angle)
    obf_queries_product_vector = model.encode(obf_queries_product)
    original_queries_vector = model.encode(original_queries)

    
    exit()

    list_of_k = [10, 50, 100]

if __name__ == "__main__":
    with Pool(35) as pool:
        pool.apply(main())
        
