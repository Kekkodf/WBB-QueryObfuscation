import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm
tqdm.pandas()

# m2hf = {"tasb": 'sentence-transformers/msmarco-distilbert-base-tas-b'}
m2hf = {"tasb": 'sentence-transformers/msmarco-distilbert-base-tas-b',
        "contriever": "facebook/contriever-msmarco",
        "ance": 'sentence-transformers/msmarco-roberta-base-ance-firstp'}

def search_faiss(queries, model_name, type_of_obfuscation='original_query', k=100):
    '''
    queries = set of queries to be encoded and searched
    model_name = name of the model to be used
    k = number of documents to be retrieved
    '''

    #------- STEP 0: choose the model that you wish to use -------#
    model = SentenceTransformer(m2hf[model_name])   
    if i == 0:
        print('Model used: ', model_name) 

    #------- STEP 1: encode the documents -------#
    #------- STEP 2: encode the queries -------#
    #original_query,obfuscated_query_distance,obfuscated_query_angle,obfuscated_query_product
    enc_queries_real = model.encode(queries[type_of_obfuscation])
    if i == 0:
        print('Queries encoded')
        print('Shape of encoded queries: ', enc_queries_real.shape) #shape = (n_queries, 768)
    #----------- STEP 3: read faiss indeces for documents -----------#
    # Load the index from the file
    #index = faiss.IndexFlatIP(model[1].word_embedding_dimension)
    index_filename = '../../../../ssd/data/faggioli/24-ECIR-FF/data/indexes/msmarco-passage/faiss/contriever/contriever.faiss'
    index = faiss.read_index(index_filename)

    mapper = list(map(lambda x: x.strip(), open('../../../../ssd/data/faggioli/24-ECIR-FF/data/indexes/msmarco-passage/faiss/contriever/contriever.map', "r").readlines()))
    #print('Index loaded')
    if i == 0:
        print('Number of documents in the index: ', len(mapper))
        print('Number of documents to be retrieved: ', k)

    #--------------------- STEP 4: search ---------------------#
    
    innerproducts, indices = index.search(enc_queries_real, k) #error assert d == model[1].word_embedding_dimension

    #print('Search performed')
    #print('Shape of innerproducts: ', innerproducts.shape) #shape = (n_queries, k)
    #print('Shape of indices: ', indices.shape) #shape = (n_queries, k)
    nqueries = len(innerproducts)
    if i == 0:
        print('Number of queries: ', nqueries)
    out = []
    for i in tqdm(range(nqueries)):
        run = pd.DataFrame(list(zip([queries.iloc[i]['query_id']] * len(innerproducts[i]), indices[i], innerproducts[i])), columns=["query_id", "did", "score"])
        run.sort_values("score", ascending=False, inplace=True)
        run['did'] = run['did'].apply(lambda x: mapper[x])
        run['rank'] = np.arange(len(innerproducts[i]))
        out.append(run)
    out = pd.concat(out)
    out["Q0"] = "Q0"
    out["run"] = model_name.replace('_', '-')
    out = out[["query_id", "Q0", "did", "rank", "score", "run"]]

    return out