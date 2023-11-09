import numpy as np
import distance
import random

def get_safe_box(word, word_embeddings_queries, model, k=3, feature='distance'):
    if feature == 'distance':
        rank = distance.get_rank(word, word_embeddings_queries, model, feature)
        candidates = rank[k:]
        secure_words = rank[:k]
        return rank, candidates, secure_words
    elif feature == 'angle':
        rank = distance.get_rank(word, word_embeddings_queries, model, feature)
        candidates = rank[k:]
        secure_words = rank[:k]
        return rank, candidates, secure_words
    else:
        raise ValueError('Feature not supported')
    
def get_weights(distribution, size, scale, shape = 2):
    if distribution == 'laplace':
        weights = np.random.laplace(loc=0, scale=scale, size=size)
        weights = np.abs(weights)
        weights = weights/np.sum(weights)
        return sorted(weights)
    elif distribution == 'uniform':
        weights = np.random.uniform(low=0.0, high=1.0, size=size)
        return weights
    elif distribution == 'normal':
        weights = np.random.normal(loc=0.0, scale=scale, size=size)
        weights = weights/np.sum(weights)
        return sorted(weights)
    elif distribution == 'gamma':
        weights = np.random.gamma(shape, scale, size)
        weights = weights/np.sum(weights)
        return sorted(weights)
    else:
        raise ValueError('Distribution not supported')

def get_new_word(secure_box, weights):
    new_word = random.choices(secure_box, weights)[0][0]
    return new_word