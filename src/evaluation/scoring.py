from ir_measures import AP, nDCG, P, RR, iter_calc, R
import pandas as pd
import ir_datasets
import os
from tqdm import tqdm
tqdm.pandas()
import time

collections = {'robust04': 'disks45/nocr/trec-robust-2004',
               'trec-covid': 'beir/trec-covid',
               'msmarco-passage': 'msmarco-passage/trec-dl-2019/judged'}


dataset = ir_datasets.load(collections['robust04'])

measures = [nDCG@10]

model = 'tfidf'

def main():
    k = [2,4]
    n = [15, 20]
    epsilons = [1, 5, 10, 12.5, 15, 17.5, 20, 50]
    types_of_obf = ['obfuscated_query_angle']
    for eps in epsilons:
        print('---'*50)
        for k_i in k:
            for n_i in n:
                for type in types_of_obf:
                    #get dataset in data/to_evaluate
                    dataset_path = './data/to_evaluate/robust-04/lexical/'
                    dataset_name = f'rank_{type}_100_{model}-{k_i}-{n_i}-{eps}.csv'
                    df_to_evaluate = pd.read_csv(dataset_path + dataset_name, sep=",", usecols=['query_id', 'doc_id', 'score'], dtype={'query_id': str, 'doc_id': str, 'score': float})

                    #reorder by query_id and score
                    #df_to_evaluate = df_to_evaluate.sort_values(by=['query_id', 'score'], ascending=[True, False])
                    #group by query_id and keep only the first 100 documents
                    #df_to_evaluate = df_to_evaluate.groupby('query_id')

                    #get qrels
                    qrels = dataset.qrels_iter()
                    qrels = pd.DataFrame(qrels)
                    #qrels.to_csv('qrels.csv', index=False, header=True, sep=',')
                    qrels.columns = ['query_id', 'doc_id', 'relevance', 'iteration']
                    qrels.query_id = qrels.query_id.astype(str)
                    qrels.doc_id = qrels.doc_id.astype(str)
                    #compute measure
                    out = pd.DataFrame(iter_calc(measures, qrels, df_to_evaluate))
                    out['measure'] = out['measure'].astype(str)
                    #out should be a df with two columns: query_id and measure, rename measure to the name of the measure
                    out = out.rename(columns={'value': str(measures[0])})
                    #drop the column iteration
                    out = out.drop(columns=['measure'])
                    #out = out.pivot(index='query_id', columns='measure', values='value')
                    out.to_csv(f'./data/results_evaluation/{k_i}-{n_i}-{eps}-obfuscation_{type}'+str(measures[0])+'.csv', index=False, header=True, sep=',')
                    print(f'{k_i}-{n_i}-{eps}-obfuscation_{type}')
                    print(f'Model {model} - Obfuscation {type} - Mean ' + str(measures[0])+' for dataset: {:.3f}'.format(out[str(measures[0])].mean()))

if __name__ == '__main__':
    main()