import numpy as np
from scipy import spatial
import embeddings

embeddings_dict = embeddings.embeddings_dict

def find_closest_embeddings_euclidean(embedding):
    #use euclidean distance for finding vectors
    return sorted(embeddings_dict.keys(), key=lambda word: spatial.distance.euclidean(embeddings_dict[word], embedding))


def find_closest_embeddings_cosine(embedding):
    #use cosine distance for finding vectors
    return sorted(embeddings_dict.keys(), key=lambda word: spatial.distance.cosine(embeddings_dict[word], embedding))

def count_similar_words_euclidean(embedding, k):
    #count similar words using euclidean distance
    number_of_words = []
    for word in embeddings_dict.keys():
        if spatial.distance.euclidean(embeddings_dict[word], embedding) <= k:
            number_of_words.append(word)
    return number_of_words

#dists = np.sqrt(np.sum((dots[:, np.newaxis] - dots) ** 2, axis=2))
# cosine distance is 1-cosine similarity
def count_similar_words_cosine(embedding, cos_alpha):
    #count similar words using cosine distance
    number_of_words = []
    for word in embeddings_dict.keys():
        if 1-spatial.distance.cosine(embeddings_dict[word], embedding) >= cos_alpha:
            number_of_words.append(word)
    return number_of_words

def count_similar_words_mahalanobis(embedding, k):
    #count similar words using cosine distance
    number_of_words = []
    for word in embeddings_dict.keys():
        iv = np.linalg.inv(np.cov(embeddings_dict[word], embedding, rowvar=False))
        if spatial.distance.mahalanobis(embeddings_dict[word], embedding, iv) <= k:
            number_of_words.append(word)
    return number_of_words