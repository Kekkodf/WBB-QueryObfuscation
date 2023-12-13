import pandas as pd
from sentence_transformers import SentenceTransformer
import ir_datasets
import faiss
import numpy as np
import multiprocessing as mp
from tqdm import tqdm   

def dot_product_matrix(x, y):
    x_expanded = x[:, np.newaxis, :]  # Shape: [n, 1, d]
    y_expanded = y[np.newaxis, :, :]  # Shape: [1, m, d]
    return np.matmul(x_expanded, y_expanded).squeeze()  # Shape: [n, m]
    

m2hf = {"tasb": 'sentence-transformers/msmarco-distilbert-base-tas-b',
        "contriever": "facebook/contriever-msmarco",
        "ance": 'sentence-transformers/msmarco-roberta-base-ance-firstp'}

model = SentenceTransformer(m2hf['contriever'])

collections = {'robust04': 'disks45/nocr/trec-robust-2004',
               'trec-covid': 'beir/trec-covid',
               'msmarco-passage': 'msmarco-passage/trec-dl-2019/judged'}

documents = ir_datasets.load(collections['msmarco-passage'])

#read the dataset
dataset = pd.read_csv('./data/to_rerank/df_to_rerank_obfuscated_query_product_10.csv', sep=",")
#convert query_id and doc_id to string
dataset.query_id = dataset.query_id.astype(str)
dataset.doc_id = dataset.doc_id.astype(str)
#drop duplicates of the rows
dataset.drop_duplicates(subset=['query_id', 'doc_id'], inplace=True)

#get original_queries
original_queries = pd.read_csv('./data/dataset_obfuscation.csv', sep=",")
original_queries = original_queries[['query_id', 'original_query']].drop_duplicates(subset=['query_id']).reset_index(drop=True)

def main():
    #create a set of query_id
    query_id_set = set(dataset['query_id'])
    #create a set of doc_id for each query_id
    dict_qid_docid = {}
    #get the doc_id for each query_id
    for qid in query_id_set:
        dict_qid_docid[qid] = set(dataset[dataset['query_id'] == qid]['doc_id'].astype(str))

    #encode documents that are in the dataset
    unique_doc_ids = set(dict_qid_docid[dataset['query_id'][0]])
    encoded_documents = []
    doc_ids = []
    for doc in tqdm(documents.docs_iter(), total=documents.docs_count()):
        if doc.doc_id in unique_doc_ids:
            doc_ids.append(doc.doc_id)
            encoded_documents.append(model.encode(doc.text))

    #encode documents as numpy array
    encoded_documents = np.array(encoded_documents)
    print(encoded_documents.shape)
    #encode original queries
    encoded_original_queries = model.encode(original_queries['original_query'])
    print(encoded_original_queries.shape)
    #compute top k documents for each query
    list_of_k = [10, 50, 100]

    for k in list_of_k:
        # Compute dot product between encoded documents and encoded queries
        dot_product = dot_product_matrix(encoded_original_queries, encoded_documents.T)
        print(dot_product.shape)
        # Compute top k documents for each query
        top_k_documents = np.argpartition(dot_product, -k)[:, -k:][:, ::-1]

        # Create a list of DataFrames for each query
        dfs = [
            pd.DataFrame({
                'query_id': str(qid),
                'doc_id': str(doc_ids[top_k_documents[i]]),
                'rank': np.arange(k)
            }) for i, qid in enumerate(query_id_set)
        ]

        # Concatenate DataFrames into a single DataFrame
        df_top_k_documents = pd.concat(dfs, ignore_index=True)

        # Convert query_id and doc_id to string
        df_top_k_documents['query_id'] = df_top_k_documents['query_id'].astype(str)
        df_top_k_documents['doc_id'] = df_top_k_documents['doc_id'].astype(str)

        # Save df_top_k_documents
        df_top_k_documents.to_csv('./data/to_evaluate/top_{k}_documents_reranked.csv'.format(k=k), index=False, header=True, sep=',')

if __name__ == '__main__':
    with mp.Pool(30) as pool:
        pool.apply(main)