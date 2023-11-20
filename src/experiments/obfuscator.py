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
        return sorted(weights, reverse=True)
    elif distribution == 'uniform':
        weights = np.random.uniform(low=0.0, high=1.0, size=size)
        return sorted(weights, reverse=True)
    elif distribution == 'normal':
        weights = np.random.normal(loc=0.0, scale=scale, size=size)
        weights = weights/np.sum(weights)
        return sorted(weights, reverse=True)
    elif distribution == 'gamma':
        weights = np.random.gamma(shape, scale, size)
        weights = weights/np.sum(weights)
        return sorted(weights, reverse=True)
    else:
        raise ValueError('Distribution not supported')

def get_new_word(rank, weights):
    new_word = random.choices(rank, weights)[0][0]
    return new_word
    
def get_obfuscated_query(tokenized_query, feature, k, distribution, scale_param, shape, model, vocab_embeddings_queries, N):
    new_query_distance_based = []
    new_query_angle_based = []
    for word, tag in tokenized_query:
        if tag == 'NN' or tag == 'NNS' or tag == 'JJ' or tag == 'NNP' or tag == 'NNPS':
            for i in range(N):
                word_i = []
                #get boxes
                rank_distance, candidates_distance, safe_box_distance = get_safe_box(word, vocab_embeddings_queries, model, k, 'distance')
                rank_angle, candidates_angle, safe_box_distance = get_safe_box(word, vocab_embeddings_queries, model, k, 'angle')
                #get weights
                weights = get_weights(distribution, len(candidates_distance), scale_param, shape) 
                #get new query
                word_i.append(get_new_word(candidates_distance, weights))
                word_i.append(get_new_word(candidates_angle, weights))
                #construct a voting system that counts the number of times a word is selected
                #the most selected word is the one that will be used
                word_i = np.array(word_i)
                unique, counts = np.unique(word_i, return_counts=True)
                word_i = dict(zip(unique, counts))
                word_i = max(word_i, key=word_i.get)
            new_query_distance_based.append(word_i)
            new_query_angle_based.append(word_i)
        else:
            new_word_distance_based = word
            new_word_angle_based = word
            new_query_distance_based.append(new_word_distance_based)
            new_query_angle_based.append(new_word_angle_based)
    if feature == 'distance':
        return new_query_distance_based
    elif feature == 'angle':
        return new_query_angle_based
    else:    
        raise ValueError('Feature not supported')