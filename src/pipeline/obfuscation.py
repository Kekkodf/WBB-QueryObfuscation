import numpy as np
from scipy.spatial.distance import euclidean, cosine
import random
import pandas as pd
from tqdm import tqdm
tqdm.pandas()

'''
Obfusctaion phases:
    - Rank computation
    - Safe box extraction
    - Candidate extraction
    - Candidate selection
'''

'''
Parameters:
    - N:= Number of extractions
    - k:= Size of safe_box
    - distribution:= Distribution for candidates selection
'''
def obfuscate(row, vocab, model, k, n, distribution, feature, num_queries):
    '''
    The method obfuscates a query using the distance based obfuscation technique.
    '''
    new_query = row.progress_apply(compute_query, args=(vocab, model, k, n, distribution, feature, num_queries))
    print(new_query)
    #print(type(new_query))
    #raise ValueError('Stop')
    return new_query
    

def compute_query(row, vocab, model, k, n, distribution, feature, num_queries):
    '''
    The method computes the rank of of most similar words in the vocabulary from a selected word in the query.
    The rank is a list of tuples (word, score) sorted by score in descending order.
    '''
    list_of_queries = []
    
    for word in row:
        query = []
        if word[1] == 'NN' or word[1] == 'NNS' or word[1] == 'NNP' or word[1] == 'NNPS' or word[1] == 'JJ':
            try:
                word_embedding = model[word[0]]
                rank = []
                for wrd in vocab:
                    wrd_embedding = model[wrd]
                    if feature == 'distance':
                        r_word = euclidean(word_embedding, wrd_embedding)
                        rank.append((wrd, r_word))
                    elif feature == 'angle':
                        r_word = 1 - cosine(word_embedding, wrd_embedding)
                        rank.append((wrd, r_word))
                    elif feature == 'product':
                        r_word = euclidean(word_embedding, wrd_embedding)*(1-cosine(word_embedding, wrd_embedding))
                        rank.append((wrd, r_word))
                    else:
                        raise ValueError('Feature not supported!')
                if feature == 'distance':
                    rank = sorted(rank, key = lambda x: x[1], reverse=False)
                    safe_box, candidates_box = partitions(rank, k, n)
                    new_word = candidate_extraction(candidates_box, distribution)
                    query.append(new_word)
                elif feature == 'angle':
                    rank = sorted(rank, key = lambda x: x[1], reverse=True)
                    safe_box, candidates_box = partitions(rank, k, n)
                    new_word = candidate_extraction(candidates_box, distribution)
                    query.append(new_word)
                elif feature == 'product':
                    rank = sorted(rank[1:], key = lambda x: x[1], reverse=False)
                    safe_box, candidates_box = partitions(rank, k, n)
                    new_word = candidate_extraction(candidates_box, distribution)
                    query.append(new_word)
            except KeyError:
                pass
        else:
            query.append(word[0])
        query = ' '.join(query)
        list_of_queries.append(query)
        
    return list_of_queries

def partitions(rank, k, n):
    '''
    rank: list of tuples (word, score)
    k: size of safe_box

    The methods partitions the rank in two parts:
    - safe_box: the first k elements of the rank (safe words, words not used in the obfuscation)
    - candidates_box: the remaining elements of the rank (possible candidates)

    return: safe_box, candidates_box
    '''
    if rank is None:
        return None, None
    elif k == 0:
        return [rank[0]], [rank[0]]
    else:
        safe_box = rank[:k]
        candidates_box = rank[k:n*k]
        return safe_box, candidates_box
    
def get_probabilities_of_extraction(candidates_box, distribution):
    if distribution[0] == 'laplace':
            probabilities = np.random.laplace(loc = distribution[1][0], scale = distribution[1][1], size = len(candidates_box))
            probabilities = np.abs(probabilities)
            probabilities = probabilities/np.sum(probabilities)
            #sort probabilities
            probabilities = sorted(probabilities, reverse=True)
            return probabilities
    elif distribution[0] == 'uniform':
        probabilities = np.random.uniform(low = distribution[1][0], high = distribution[1][1], size = len(candidates_box))
        probabilities = sorted(probabilities, reverse=True)
        return probabilities
    elif distribution[0] == 'normal':
        probabilities = np.random.normal(loc = distribution[1][0], scale = distribution[1][1], size = len(candidates_box))
        probabilities = probabilities/np.sum(probabilities)
        #sort probabilities
        probabilities = sorted(probabilities, reverse=True)
        return probabilities
    elif distribution[0] == 'gamma':
        probabilities = np.random.gamma(distribution[1][0], distribution[1][1], size = len(candidates_box))
        probabilities = probabilities/np.sum(probabilities)
        #sort probabilities
        probabilities = sorted(probabilities, reverse=True)
        return probabilities
    else:
        raise ValueError('Distribution not supported')


    
def candidate_extraction(candidates_box, distribution):
    probabilities = get_probabilities_of_extraction(candidates_box, distribution)
    new_word = random.choices(candidates_box, probabilities)[0][0]
    return new_word