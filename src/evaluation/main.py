import pandas as pd
import numpy as np
import ir_datasets
from index_and_retrieve import search_faiss
from ir_measures import AP, nDCG, P, R, iter_calc
import os
from tqdm import tqdm
tqdm.pandas()


collections = {'robust04': 'disks45/nocr/trec-robust-2004',
               'trec-covid': 'beir/trec-covid',
               'msmarco-passage': 'msmarco-passage/trec-dl-2019/judged'}


dataset = ir_datasets.load(collections['msmarco-passage'])
#for query in dataset.queries_iter():
#    print(query)
#    break
#for doc in dataset.docs_iter():
#    print(doc)
#    break
#exit()
measures = [P@10]
dataset_path = './data/final/'

k = 4
n = 10
distribution = ('gamma', (1, 2)) #(name, param_1, ..., param_n)
collection = 'msmarco-passage'

dataset_name = 'dataset-{k}-{n}_{distribution}_{collection}.csv'.format(k=k, n=n, distribution=distribution, collection=collection)

#read the dataset
query_dataset = pd.read_csv(dataset_path + dataset_name, sep=",")

def main():
    #get list of dataset available
    list_of_dataset_queries = [dataset for dataset in os.listdir(dataset_path) if dataset.endswith('.csv')]
    #read ith-dataset obfuscated queries with
    #to read dataset use pd.read_csv(dataset_path + list_of_dataset[i])
    list_of_k = [10, 50, 100]
    #lack collection name
    for i,k_i in zip(range(len(list_of_dataset_queries)), list_of_k):
        queries_retrieved = pd.read_csv(dataset_path + list_of_dataset_queries[i], sep=",")
        #get columns query_id and type of query: original_query,obfuscated_query_distance,obfuscated_query_angle,obfuscated_query_product
        type_of_query = 'obfuscated_query_product'
        queries_retrieved = queries_retrieved[['query_id', type_of_query]]
        df_to_evaluate = search_faiss(queries_retrieved, 'contriever', type_of_query, k=k_i, i=i)

        #df_to_evaluate.to_csv('df_containing_ranks_dataset_{}.csv'.format(list(list_of_dataset_queries)[i]), index=False, header=True, sep=',')

        df_to_evaluate.query_id = df_to_evaluate.query_id.astype(str)
        df_to_evaluate.did = df_to_evaluate.did.astype(str)
        #rename did to doc_id
        df_to_evaluate.rename(columns={'did': 'doc_id'}, inplace=True)
        #save df_to_evaluate
        #drop duplicates of the rows
        df_to_evaluate.drop_duplicates(subset=['query_id', 'doc_id'], inplace=True)
        df_to_evaluate.to_csv('./data/to_rerank/df_to_rerank_{type_of_query}_{k_i}.csv'.format(type_of_query=type_of_query, k_i=k_i), index=False, header=True, sep=',')
        

        #get qrels
        #qrels = dataset.qrels_iter()
        #qrels = pd.DataFrame(qrels)
        ##qrels.to_csv('qrels.csv', index=False, header=True, sep=',')
        #qrels.columns = ['query_id', 'doc_id', 'relevance', 'type']
        #qrels.query_id = qrels.query_id.astype(str)
        #qrels.doc_id = qrels.doc_id.astype(str)
#
        ##filter query_id and keep only the ones in qrels
        #df_to_evaluate = df_to_evaluate[df_to_evaluate['query_id'].isin(qrels['query_id'])]
        #
        ##compute measure
        #out = pd.DataFrame(iter_calc(measures, qrels, df_to_evaluate))
        #out['measure'] = out['measure'].astype(str)
        #out = out.pivot(index='query_id', columns='measure', values='value').reset_index()
        ##out.to_csv('nDCG@10_dataset_{}.csv'.format(list(list_of_dataset_queries)[i]), index=False, header=True, sep=',')
#
        #print('Mean nDCG@10 for dataset {}: {:.2f}%'.format(i, out['nDCG@10'].mean()*100))
#

if __name__ == '__main__':
    main()
