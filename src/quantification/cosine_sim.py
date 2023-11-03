import numpy as np
import embeddings as emb
import scipy.spatial.distance as dist
import math

embeddings_dict = emb.embeddings_dict

def cosine_sim(word1, word2, cos_alpha):
    """Returns cosine similarity between two words"""
    if word2 in embeddings_dict.keys():
        return 1-dist.cosine(embeddings_dict[word1], embeddings_dict[word2])