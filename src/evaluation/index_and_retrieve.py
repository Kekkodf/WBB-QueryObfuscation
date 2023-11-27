import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

def sentence_embedding(doc, queries, model_name = "facebook/contriever-msmarco"):

    #------- STEP 0: choose the model that you wish to use -------#

    #eg: contriever
    #use a handler of an hugging face model
    model = SentenceTransformer(model_name)

    #--------------- STEP 1: encode the documents ---------------#
    docs = pd.read_csv(doc)
    encoded_docs =  model.encode(docs["body"])

    #----------- STEP 2: put them into a faiss index -----------#
    # When searching, faiss will return the number of rows: you will need a map to keep track
    # of which document correspond to each row
    mapper_path = '/home/kdf/mnt_grace/ssd/data/faggioli/24-ECIR-FF/data/indexes/msmarco-passage/faiss/glove/glove.map'
    mapper = list(map(lambda x: x.strip(), open(mapper_path, "r").readlines()))

    '''
    # if you have a faiss index already saved on disk, you can read it
    # Specify the file path of the saved index
    index_filename = "path/to/your/faiss/index.faiss"
    # Load the index from the file
    faiss_index = faiss.read_index(index_filename)
    '''
    index_filename = "/home/kdf/mnt_grace/ssd/data/faggioli/24-ECIR-FF/data/indexes/msmarco-passage/faiss/glove/glove.faiss"
    faiss_index = faiss.read_index(index_filename)

    faiss_index = faiss.IndexFlatIP(model[1].word_embedding_dimension)
    faiss_index.add(encoded_docs)

    # you can also save the model for future use
    #faiss.write_index(faiss_index, f"path/to/your/faiss/index.faiss")

    #--------------- STEP 3: encode the queries ---------------#
    queries = pd.read_csv(queries)
    encoded_queries_real = model.encode(queries["original_query"])
    encoded_queries_obf_distance = model.encode(queries["obfuscated_query_distance"])
    encoded_queries_obf_angle = model.encode(queries["obfuscated_query_angle"])
    encoded_queries_obf_product = model.encode(queries["obfuscated_query_product"])

    #--------------------- STEP 4: search ---------------------#  
    k = 10 # number of docs retrieved per query
    dots, idxs = faiss_index.search(encoded_queries_real, k)
    dots_obf_distance, idxs_obf_distance = faiss_index.search(encoded_queries_obf_distance, k)
    dots_obf_angle, idxs_obf_angle = faiss_index.search(encoded_queries_obf_angle, k)
    dots_obf_product, idxs_obf_product = faiss_index.search(encoded_queries_obf_product, k)

    #dots is a list of lists (maximum dimension #queries x k), where the ij-th cell corresponds to the dot
    #product of the i-th query representation with the j-th most similar document
    #indxs is a similar list of lists, where there are the indexes of the most similar documents

    run = []
    for i in range(len(dots)):
        retrieved_docs = pd.DataFrame({"did": idxs[i], "score_real":dots[i], "score_obf_distance":dots_obf_distance[i], "score_obf_angle":dots_obf_angle[i], "score_obf_product":dots_obf_product[i]})
        retrieved_docs['qid'] = queries.iloc[i]['qid']
        retrieved_docs['did'] = retrieved_docs['did'].apply(lambda x: mapper[x])
        run.append(retrieved_docs)

    run = pd.concat(run)

    print(run)