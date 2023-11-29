import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# m2hf = {"tasb": 'sentence-transformers/msmarco-distilbert-base-tas-b'}
m2hf = {"tasb": 'sentence-transformers/msmarco-distilbert-base-tas-b',
        "contriever": "facebook/contriever-msmarco",
        "glove": 'sentence-transformers/average_word_embeddings_glove.6B.300d'}


def search_faiss(queries, collection, model_name, k=1000):
    model = SentenceTransformer(m2hf[model_name])
    enc_queries = model.encode(queries['query'])

    # Specify the file path of the saved index
    index_filename = f"../../data/indexes/{collection}/faiss/{model_name}/{model_name}.faiss"
    # Load the index from the file
    index = faiss.read_index(index_filename)

    mapper = list(map(lambda x: x.strip(), open(f"../../data/indexes/{collection}/faiss/{model_name}/{model_name}.map", "r").readlines()))

    innerproducts, indices = index.search(enc_queries, k)
    nqueries = len(innerproducts)
    out = []
    for i in range(nqueries):
        run = pd.DataFrame(list(zip([queries.iloc[i]['qid']] * len(innerproducts[i]), indices[i], innerproducts[i])), columns=["qid", "did", "score"])
        run.sort_values("score", ascending=False, inplace=True)
        run['did'] = run['did'].apply(lambda x: mapper[x])
        run['rank'] = np.arange(len(innerproducts[i]))
        out.append(run)
    out = pd.concat(out)
    out["Q0"] = "Q0"
    out["run"] = model_name.replace('_', '-')
    out = out[["qid", "Q0", "did", "rank", "score", "run"]]

    return out
