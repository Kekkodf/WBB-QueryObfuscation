from scipy.spatial.distance import euclidean, cosine
import math
import nltk
#import Part of Speech (POS) tagger
from nltk import pos_tag

def get_distance(v1, v2, distance):
    try:
        if distance == 'cosine':
            return math.degrees(math.acos(1-cosine(v1, v2)))
        elif distance == 'euclidean':
            return euclidean(v1, v2)
        else:
            raise ValueError('Distance not supported')
    except:
        pass

def get_rank(word, vocab_embeddings_queries, model, feature = 'distance'):
    word_embedding = vocab_embeddings_queries[word]
    if feature == 'distance':
        distances = []
        for wrd in vocab_embeddings_queries.keys():
            if wrd == word:
                distance = 0
                distances.append((wrd, distance))
            else:
                embedding = model[wrd]
                distance = get_distance(word_embedding, embedding, 'euclidean')
                distances.append((wrd, distance))
        distances = sorted(distances, key = lambda x: x[1])
        return distances
    elif feature == 'angle':
        angles = []
        for wrd in vocab_embeddings_queries.keys():
            if wrd == word:
                angle = 0
                angles.append((wrd, angle))
            else:
                embedding = model[wrd]
                angle = get_distance(word_embedding, embedding, 'cosine')
                angles.append((wrd, angle))
        angles = sorted(angles, key = lambda x: x[1])
        return angles