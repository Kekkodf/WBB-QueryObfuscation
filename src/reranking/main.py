import pandas as pd
import numpy as np
import ir_datasets
from index_and_retrieve import search_faiss
import os
from tqdm import tqdm
tqdm.pandas()


collections = { #'robust04': '/ssd/data/faggioli/EXPERIMENTAL_COLLECTIONS/CORPORA/IR_DATASETS_CORPORA/disks45',
                'robust-04': 'disks45/nocr/trec-robust-2004',
               'msmarco-dl19': 'msmarco-passage/trec-dl-2019/judged'}

#collection = 'msmarco-dl19'
collection = 'robust-04'

dataset = ir_datasets.load(collections[collection])
#for query in dataset.queries_iter():
#    print(query)
#    break
#for doc in dataset.docs_iter():
#    print(doc)
#    break
#exit()

dataset_path = f'./data/final/{collection}/'

retr = 'contriever'

def main():
    k = [2,4]
    n = [15,20]
    epsilons = [1, 5, 10, 12.5, 15, 17.5, 20, 50]

    for epsilon in epsilons:
        for n_i in n:
            for k_i in k:
                dataset_name = f'dataset-{k_i}-{n_i}-{epsilon}_{collection}' + '.csv'
                #read the dataset
                query_dataset = pd.read_csv(dataset_path + dataset_name, sep=",")
                #get dataset queries
                df_queries = pd.read_csv(dataset_path + dataset_name, sep=",", header=0)

                types_of_obf = ['obfuscated_query_angle']
                #get the type of query
                for type_of_obf in types_of_obf:

                    df_to_evaluate = search_faiss(df_queries, retr, type_of_obf)

                    df_to_evaluate.query_id = df_to_evaluate.query_id.astype(str)
                    df_to_evaluate.did = df_to_evaluate.did.astype(str)
                        #rename did to doc_id
                    df_to_evaluate.rename(columns={'did': 'doc_id'}, inplace=True)
                        #save df_to_evaluate
                    df_to_evaluate.to_csv(f'./data/to_rerank/df_to_rerank_{retr}_{type_of_obf}_100_{k_i}_{n_i}_{epsilon}.csv', index=False, header=True, sep=',')

if __name__ == '__main__':
    main()
