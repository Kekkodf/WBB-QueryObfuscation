import numpy as np
from scipy.spatial.distance import euclidean, cosine
import random

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

def compute_rank(word, vocab, model):
    '''
    The method computes the rank of of most similar words in the vocabulary from a selected word in the query.
    The rank is a list of tuples (word, score) sorted by score in descending order.
    '''
    try:
        word_embedding = model[word]
        rank_distance = []
        rank_angle = []
        rank_product = []
        for wrd in vocab:
            wrd_embedding = model[wrd]
            r_word_distance = euclidean(word_embedding, wrd_embedding)
            rank_distance.append((wrd, r_word_distance))
            r_word_angle = 1 - cosine(word_embedding, wrd_embedding)
            rank_angle.append((wrd, r_word_angle))
            r_word_product = r_word_distance*r_word_angle
            rank_product.append((wrd, r_word_product))
            
            rank_distance = sorted(rank_distance, key = lambda x: x[1], reverse=False)  
            rank_angle = sorted(rank_angle, key = lambda x: x[1], reverse=True) 
            rank_product = sorted(rank_product[1:], key = lambda x: x[1], reverse=False)
            
        return rank_distance, rank_angle, rank_product
    except KeyError:
        print('Word not in vocabulary')
        pass
        

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
    #print(probabilities[:5])
    #raise ValueError('Stop')
    new_word = random.choices(candidates_box, probabilities)[0][0]
    return new_word

        

            

#def candidate_selection():
#    
#