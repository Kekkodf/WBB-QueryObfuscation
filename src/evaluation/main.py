import pandas as pd
import numpy as np
import ir_datasets
from ir_measures import nDCG
from index_and_retrieve import search_faiss
from evaluate import compute_measure, iter_calc
import os
from tqdm import tqdm
tqdm.pandas()


collections = {'robust04': 'disks45/nocr/trec-robust-2004',
               'trec-covid': 'beir/trec-covid',
               'msmarco-passage': 'msmarco-passage/trec-dl-2019/judged'}


dataset = ir_datasets.load(collections['msmarco-passage'])
measures = [nDCG @ 10]
dataset_path = './results/pipeline/'

def main():
    #get list of dataset available
    list_of_dataset_queries = [dataset for dataset in os.listdir(dataset_path) if dataset.endswith('.csv')]
    #read ith-dataset obfuscated queries with
    #to read dataset use pd.read_csv(dataset_path + list_of_dataset[i])

    #lack collection name
    for i in range(len(list_of_dataset_queries)):
        queries_retrieved = pd.read_csv(dataset_path + list_of_dataset_queries[i], sep=",")
        #get columns query_id and type of query: original_query,obfuscated_query_distance,obfuscated_query_angle,obfuscated_query_product
        type_of_query = 'obfuscated_query_product'
        queries_retrieved = queries_retrieved[['query_id', type_of_query]]
        df_to_evaluate = search_faiss(queries_retrieved, 'contriever', type_of_query, k=1000, i=i)

        #df_to_evaluate.to_csv('df_containing_ranks_dataset_{}.csv'.format(list(list_of_dataset_queries)[i]), index=False, header=True, sep=',')

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
        #out.to_csv('nDCG@10_dataset_{}.csv'.format(list(list_of_dataset_queries)[i]), index=False, header=True, sep=',')

        print('Mean nDCG@10 for dataset {}: {:.2f}%'.format(i, out['nDCG@10'].mean() * 100))


if __name__ == '__main__':
    main()
