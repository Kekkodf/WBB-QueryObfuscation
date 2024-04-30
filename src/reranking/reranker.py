import pandas as pd
from sentence_transformers import SentenceTransformer
import ir_datasets
import numpy as np
import multiprocessing as mp
from tqdm import tqdm   
from MemmapEncoding import MemmapQueriesEncoding, MemmapCorpusEncoding
import time

#change the path for different models
#MSMARCO-dl19
#memmapcorpus = MemmapCorpusEncoding("/ssd/data/faggioli/EXPERIMENTAL_COLLECTIONS/INDEXES/memmap/msmarco-passages/contriever/contriever.dat", "/ssd/data/faggioli/EXPERIMENTAL_COLLECTIONS/INDEXES/memmap/msmarco-passages/contriever/contriever_map.csv")
#ROBUST-04
memmapcorpus = MemmapCorpusEncoding("PATH_TO_MEMMAP_DAT", "PATH_TO_MEMMAP_MAP")

def dot_product_matrix(x, y):
    x_expanded = x[:, np.newaxis, :]  # Shape: [n, 1, d]
    y_expanded = y[np.newaxis, :, :]  # Shape: [1, m, d]
    product = np.sum(x_expanded * y_expanded, axis=-1)  # Shape: [n, m]
    return product
    

m2hf = {"tasb": 'sentence-transformers/msmarco-distilbert-base-tas-b',
        "contriever": "facebook/contriever-msmarco",
        "ance": 'sentence-transformers/msmarco-roberta-base-ance-firstp'}

model = SentenceTransformer(m2hf['contriever']) #rerank performed by contriever

collections = {'robust04': 'disks45/nocr/trec-robust-2004',
               'msmarco-passage': 'msmarco-passage/trec-dl-2019/judged'}

documents = ir_datasets.load(collections['robust04'])

retr = 'contriever' #model used as IRS
def main():
    k = [2,4]
    n = [15,20]
    
    epsilons = [1, 5, 10, 12.5, 15, 17.5, 20, 50]
    #get original_queries
    for eps in epsilons:
        for k_i in k:
            for n_i in n:
                #original_queries = pd.read_csv(f'./data/final/msmarco-dl19/dataset-{k_i}-{n_i}-{eps}_msmarco-dl19.csv', sep=",")
                original_queries = pd.read_csv(f'./data/final/robust-04/dataset-{k_i}-{n_i}-{eps}_robust-04.csv', sep=",")
                original_queries = original_queries[['query_id', 'original_query']].drop_duplicates(subset=['query_id']).reset_index(drop=True)
                #convert query_id to string
                original_queries.query_id = original_queries.query_id.astype(str)

                types = ['obfuscated_query_angle']
                for type in types:
                    #read the dataset
                    dataset = pd.read_csv(f'./data/to_rerank/robust-04/df_to_rerank_{retr}_{type}_100_{k_i}_{n_i}_{eps}.csv', sep=",")
                    #convert query_id and doc_id to string
                    dataset.query_id = dataset.query_id.astype(str)
                    dataset.doc_id = dataset.doc_id.astype(str)
                    #drop duplicates of the rows
                    dataset.drop_duplicates(subset=['query_id', 'doc_id'], inplace=True)

                    #create a set of query_id
                    query_id_set = set(dataset['query_id'])
                    #convert the set to a pd.Series
                    query_id_set = pd.Series(list(query_id_set))
                    #create a set of doc_id for each query_id
                    dict_qid_docid = {}
                    #get the doc_id for each query_id
                    for qid in query_id_set:
                        dict_qid_docid[qid] = set(dataset[dataset['query_id'] == qid]['doc_id'].astype(str))

                #see how many documents have been retrieved by each query
                #for qid in query_id_set:
                #    print('Number of documents for query {qid}: {n}'.format(qid=qid, n=len(dict_qid_docid[qid])))
                    rank = []
                    i = 1
                    for qid in query_id_set:
                        print('Reranking documents for query: {qid}, iteration {i}'.format(qid=qid, i=i))
                        #get the original query if qid == query_id in original_queries
                        if qid in original_queries['query_id'].values:
                            original_query = original_queries.iloc[original_queries[original_queries['query_id'] == qid].index[0]]['original_query']
                            #encode the original query
                            original_query_encoded = model.encode(original_query)
                            #get the documents for the query qid
                            document_matrix = memmapcorpus.get_encoding(list(dict_qid_docid[qid]))
                            scores = np.dot(document_matrix, original_query_encoded)
                            #use argpartition to get the top k documents
                            top_100 = np.argpartition(scores, -100)[-100:]
                            #get the top 100 documents  
                            top_100_docs = np.array(list(dict_qid_docid[qid]))[top_100]
                            #get the scores for the top 100 documents
                            top_100_scores = scores[top_100]

                            #create a dataframe with the rank
                            rank.append(pd.DataFrame(list(zip([qid] * len(top_100_docs), top_100_docs, top_100_scores)), columns=['query_id', 'doc_id', 'score']))
                            i += 1


                    #create a dataframe with the rank
                    rank = pd.concat(rank)
                    #sort the dataframe by query_id and score
                    rank.sort_values(['query_id', 'score'], ascending=[True, False], inplace=True)
                    #reset the index
                    rank.reset_index(drop=True, inplace=True)
                    #add the rank
                    rank['rank'] = rank.groupby('query_id').cumcount() + 1
                    #add the run
                    rank['run'] = retr
                    #add the Q0
                    rank['Q0'] = 'Q0'
                    #reorder the columns
                    rank = rank[['query_id', 'Q0', 'doc_id', 'rank', 'score', 'run']]
                    #save the dataframe
                    rank.to_csv(f'./data/to_evaluate/rank_{type}_100_{retr}-{k_i}-{n_i}-{eps}.csv', index=False, header=True, sep=',')                
if __name__ == '__main__':
    with mp.Pool(50) as pool:
        pool.apply(main)
