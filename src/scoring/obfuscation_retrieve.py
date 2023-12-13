import pandas as pd
import numpy as np
import ir_datasets
from sentence_transformers import SentenceTransformer
import faiss
import multiprocessing as mp
from tqdm import tqdm

###########################DEFINITION OF MODEL, DOCS AND QUERIES##############################################
collections = {'robust04': 'disks45/nocr/trec-robust-2004',
                'trec-covid': 'beir/trec-covid',
                'msmarco-passage': 'msmarco-passage/trec-dl-2019/judged'}

m2hf = {"tasb": "sentence-transformers/msmarco-distilbert-base-tas-b",
        "contriever": "facebook/contriever-msmarco",
        "ance": "sentence-transformers/msmarco-roberta-base-ance-firstp"}

model = SentenceTransformer(m2hf["contriever"])

#define dataset
dataset = ir_datasets.load(collections['msmarco-passage'])

vectorized_docs = model.encode(doc.text for doc in dataset.docs_iter())

#load the obfuscated queries
obf_queries = pd.read_csv('data/dataset_obfuscation.csv', sep=',')

###########################DEFINITION OF THE FUNCTIONS########################################################
def dot_products(x, y):
        x_expanded = x[:, np.newaxis, :]
        y_expanded = y[np.newaxis, :, :]
        return np.dot(x_expanded, y_expanded)

def obf_retrieval(obf_query, k):
        '''
        This function retrieves the top k most similar documents for a given obfuscated query.
        :param obf_query: obfuscated query
        :param k: number of retrieved documents
        ''' 
        #vectorize the obfuscated query
        obf_query_vector = model.encode(obf_query)

        index_filename = '../../../../ssd/data/faggioli/24-ECIR-FF/data/indexes/msmarco-passage/faiss/contriever/contriever.faiss'
        index = faiss.read_index(index_filename)

        mapper = list(map(lambda x: x.strip(), open('../../../../ssd/data/faggioli/24-ECIR-FF/data/indexes/msmarco-passage/faiss/contriever/contriever.map', "r").readlines()))
    
        #compute similarity between the obfuscated query and the documents
        innerproducts, indices = index.search(obf_query_vector, k)
        print('Search performed')
        out = []
        for i in tqdm(range(k)):
            

def main():
        #get column with obfuscated queries distance
        obf_queries_distance = obf_queries['obfuscated_query_distance']
        #get column with obfuscated queries angle
        obf_queries_angle = obf_queries['obfuscated_query_angle']
        #get column with obfuscated queries product
        obf_queries_product = obf_queries['obfuscated_query_product']



#if __name__ == '__main__':
#    with mp.Pool(35) as pool:
#        pool.apply(main)
#
