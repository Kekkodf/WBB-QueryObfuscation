import numpy as np
import pandas as pd
import distance
import math

def get_safe_box(word, word_embeddings_queries, model, k=3, feature='distance'):
    if feature == 'distance':
        rank = distance.get_rank(word, word_embeddings_queries, model, feature)
        secure_box = rank[:k]
        return rank, secure_box
    elif feature == 'angle':
        rank = distance.get_rank(word, word_embeddings_queries, model, feature)
        secure_box = rank[:k]
        return rank, secure_box
    else:
        raise ValueError('Feature not supported')