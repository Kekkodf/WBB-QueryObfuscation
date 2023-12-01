import pandas as pd
import numpy as np
import logging
import ir_datasets
from index_and_retrieve import search_faiss
from ir_measures import AP, nDCG, P, R, iter_calc
from index_and_retrieve import search_faiss
import os
from tqdm import tqdm
tqdm.pandas()

#define possible collections
collections = {'robust04': 'disks45/nocr/trec-robust-2004',
               'trec-covid': 'beir/trec-covid',
               'msmarco-passage': 'msmarco-passage/trec-dl-2019/judged'}

#define dataset
dataset = ir_datasets.load(collections['msmarco-passage'])
#define scoring measures
measures = [AP]
#define path to dataset
dataset_path = './data/'

def main():
    #define dataset via parameters of obfuscation
    k = 4
    n = 8
    distribution = ('gamma', (1, 2)) #(name, param_1, ..., param_n)
    collection = 'msmarco-passage'
    model = 'contriever'
    #read the dataset
    dataset_name = 'dataset-{k}-{n}_{distribution}_{collection}.csv'.format(k=k, n=n, distribution=distribution, collection=collection)
    query_dataset = pd.read_csv(dataset_path + dataset_name, sep=",")
    #define list of possible obfuscation: original_query,obfuscated_query_distance,obfuscated_query_angle,obfuscated_query_product
    list_of_types_of_obfuscation = ['original_query', 'obfuscated_query_distance', 'obfuscated_query_angle', 'obfuscated_query_product']
    for type_of_query in list_of_types_of_obfuscation:
        #evaluation
        queries_retrieved = query_dataset[['query_id', type_of_query]]
        #use faiss indices to retrieve documents
        df_to_evaluate = search_faiss(queries_retrieved, model, type_of_query, k=1000)
        df_to_evaluate.query_id = df_to_evaluate.query_id.astype(str)
        df_to_evaluate.did = df_to_evaluate.did.astype(str)
        #rename did to doc_id
        df_to_evaluate.rename(columns={'did': 'doc_id'}, inplace=True)
        #get qrels
        qrels = dataset.qrels_iter()
        qrels = pd.DataFrame(qrels)
        #qrels.to_csv('qrels.csv', index=False, header=True, sep=',')
        qrels.columns = ['query_id', 'doc_id', 'relevance', 'type']
        qrels.query_id = qrels.query_id.astype(str)
        qrels.doc_id = qrels.doc_id.astype(str)
        #filter query_id and keep only the ones in qrels
        df_to_evaluate = df_to_evaluate[df_to_evaluate['query_id'].isin(qrels['query_id'])]

        #compute measure
        out = pd.DataFrame(iter_calc(measures, qrels, df_to_evaluate))
        out['measure'] = out['measure'].astype(str)
        out = out.pivot(index='query_id', columns='measure', values='value').reset_index()
        
        print('Mean {} for dataset {}, {}: {:.2f}%'.format(measures[0], dataset_name, type_of_query, out['{}'.format(measures[0])].mean()*100))

if __name__ == '__main__':
    main()
