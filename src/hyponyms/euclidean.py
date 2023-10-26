import numpy as np
import embeddings as emb
import scipy.spatial.distance as dist

embeddings_dict = emb.embeddings_dict

def euclidean_distance(word1, word2):
    """Returns euclidean distance between two words"""
    if word2 in embeddings_dict.keys():
        return dist.euclidean(embeddings_dict[word1], embeddings_dict[word2])
