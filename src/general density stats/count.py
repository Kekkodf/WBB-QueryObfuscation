import numpy as np
import embeddings
import distances


def count_words_euclidean(word, k):
    results = []
    x = np.linspace(0, k)
    for i in x:
        distance = distances.count_similar_words_euclidean(embeddings.embeddings_dict[word], i)
        results.append(distance)
    counts = []
    for element in results:
        counts.append(len(element))
    return x, counts

def count_words_cosine(word, cos_alpha):
    results = []
    x = np.linspace(cos_alpha, 1)
    for i in x:
        distance = distances.count_similar_words_cosine(embeddings.embeddings_dict[word], i)
        results.append(distance)
    counts = []
    for element in results:
        counts.append(len(element))
    return x, counts

def count_words_mahalanobis(word, k):
    results = []
    x = np.linspace(0, k)
    for i in x:
        distance = distances.count_similar_words_mahalanobis(embeddings.embeddings_dict[word], i)
        results.append(distance)
    counts = []
    for element in results:
        counts.append(len(element))
    return x, counts