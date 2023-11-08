from scipy.spatial.distance import euclidean, cosine
import math

def get_distance(word1, word2, model, distance):
    try:
        if distance == 'cosine':
            return math.acos(1-cosine(model[word1], model[word2]))
        elif distance == 'euclidean':
            return euclidean(model[word1], model[word2])
        else:
            raise ValueError('Distance not supported')
    except:
        pass

def get_most_angular_distant(word, alpha, vocab_embeddings, model):
    angles = []
    for wrd in vocab_embeddings.keys():
        if wrd == word:
            continue
        else:
            angle = get_distance(word, wrd, model, 'cosine')
            angles.append((wrd, angle))
    angles = sorted(angles, key=lambda x: x[1])
    answer = []
    for angle in angles:
        if angle[1] < alpha:
            angle = (angle[0], math.degrees(angle[1]))
            answer.append(angle)
    return answer

def get_most_euclidean_distant(word, d, vocab_embeddings, model):
    distances = []
    for wrd in vocab_embeddings.keys():
        if wrd == word:
            continue
        else:
            distance = get_distance(word, wrd, model, 'euclidean')
            distances.append((wrd, distance))
    distances = sorted(distances, key=lambda x: x[1])
    answer = []
    for distance in distances:
        if distance[1] < d:
            answer.append(distance)
    return answer