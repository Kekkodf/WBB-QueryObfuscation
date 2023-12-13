import pandas as pd
import numpy as np
import ir_datasets
from sentence_transformers import SentenceTransformer
from multiprocessing import Pool

collections = {'robust04': 'disks45/nocr/trec-robust-2004',
                'trec-covid': 'beir/trec-covid',
                'msmarco-passage': 'msmarco-passage/trec-dl-2019/judged'}

m2hf = {"tasb": "sentence-transformers/msmarco-distilbert-base-tas-b",
        "contriever": "facebook/contriever-msmarco",
        "ance": "sentence-transformers/msmarco-roberta-base-ance-firstp"}
#define dataset

# Global variables
model = m2hf["contriever"]  # Your model for similarity computation
query_dataset_path = './data/dataset_obfuscation.csv'  # Your dataset of queries
document_collection = collections['msmarco-passage']  # Your collection of documents

def compute_similarity(query, documents):
    # Your code to compute dot product or other similarity measure
    return np.dot(query, documents)

def process_chunk(chunk):
    return chunk.apply(lambda row: compute_similarity(row['query'], row['documents']), axis=1)

def main():
    global model, query_dataset, document_collection

    k = 100
    # Initialize your model and load data
    model = SentenceTransformer(model)  # Implement this function to initialize your model
    query_dataset = pd.read_csv(query_dataset_path, sep = ',', header=0) # Implement this function to load your query dataset
    query_dataset['query'] = query_dataset['obfuscated_query_distance'].apply(model.encode)  # Implement this function to vectorize your queries

    document_collection = ir_datasets.load(document_collection)  # Implement this function to load your document collection
    document_collection = pd.DataFrame(list(document_collection.docs_iter()), columns=['common_key_column', 'documents'])  # Implement this function to convert your document collection into a DataFrame

    # Vectorize documents
    document_collection['documents'] = document_collection['documents'].apply(model.encode)

    # Combine queries and documents into a single DataFrame
    data = pd.merge(query_dataset, document_collection, on='common_key_column')

    # Split data into chunks
    num_processes = 35
    chunks = np.array_split(data, num_processes)

    # Use multiprocessing to parallelize the computation
    with Pool(num_processes) as pool:
        results = pool.map(process_chunk, chunks)

    # Combine results into a single DataFrame
    final_result = pd.concat(results)

    # Get the top most similar documents for each query
    top_similar_documents = final_result.groupby('query_index').apply(lambda group: group.nlargest(k, 'similarity'))

    # Save the results
    top_similar_documents.to_csv('./data/top_1000_documents.csv', index=False)

if __name__ == "__main__":
    main()
