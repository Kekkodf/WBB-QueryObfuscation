from ir_measures import AP, nDCG, P, R, iter_calc
import pandas as pd
import ir_datasets
import os
from tqdm import tqdm
tqdm.pandas()

collections = {'robust04': 'disks45/nocr/trec-robust-2004',
               'trec-covid': 'beir/trec-covid',
               'msmarco-passage': 'msmarco-passage/trec-dl-2019/judged'}


dataset = ir_datasets.load(collections['msmarco-passage'])

measures = [P@10]

def compute_measure(run, qrels, measures):
    out = pd.DataFrame(iter_calc(measures, qrels, run))
    out['measure'] = out['measure'].astype(str)
    return out

def main():
    #get dataset in data/to_evaluate
    dataset_path = './data/to_evaluate/'
    dataset_name = 'top_10_documents_reranked.csv'
    df_to_evaluate = pd.read_csv(dataset_path + dataset_name, sep=",")
    #insert Q0 column as second column
    df_to_evaluate.insert(1, 'Q0', 'Q0')
    #insert score column as last column
    df_to_evaluate.insert(4, 'score', 1.0)
    #insert iteration column as last column
    df_to_evaluate.insert(5, 'iteration', 'iteration')
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
    print('Mean P@10 for dataset: {:.2f}%'.format(out['P@10'].mean()*100))

if __name__ == '__main__':
    main()