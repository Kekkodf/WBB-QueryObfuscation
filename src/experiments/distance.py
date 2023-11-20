from scipy.spatial.distance import euclidean, cosine
import math
import nltk
#import Part of Speech (POS) tagger
from nltk import pos_tag

def get_distance(v1, v2, distance):
    '''
    get_distance computes the distance between two vectors v1 and v2
    '''
    try:
        if distance == 'cosine':
            return 1-cosine(v1, v2)
        elif distance == 'euclidean':
            return euclidean(v1, v2)
        else:
            raise ValueError('Distance not supported')
    except:
        pass

def get_rank(word, vocab_embeddings_queries, model, feature = 'distance'):
    '''
    get_rank returns rank of words based on a feature

    the rank is from the nearest to the farthest word from the word of interest
    '''
    word_embedding = model[word]
    if feature == 'distance':
        distances = []
        for wrd in vocab_embeddings_queries.keys():
            embedding = model[wrd]
            distance = get_distance(word_embedding, embedding, 'euclidean')
            distances.append((wrd, distance))
        distances = sorted(distances, key = lambda x: x[1], reverse=False)
        return distances
    elif feature == 'angle':
        angles = []
        for wrd in vocab_embeddings_queries.keys():
            embedding = model[wrd]
            angle = get_distance(word_embedding, embedding, 'cosine')
            angles.append((wrd, angle))
        angles = sorted(angles, key = lambda x: x[1], reverse=True)
        return angles