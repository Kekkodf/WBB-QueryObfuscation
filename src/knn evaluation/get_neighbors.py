import numpy as np
import embeddings as emb

embeddings_dict = emb.embeddings_dict

def get_neighbors(word, nbrs):
    """Returns k nearest neighbors of a word"""
    word = word.lower()
    try:
        if word in embeddings_dict.keys():
            distances, indices = nbrs.kneighbors([embeddings_dict[word]])
            return distances
    except:
        pass

def count_neighbours(syn_distances, hyp_distances, distance):
    counts_syn = []
    count = 0
    for syn in range(syn_distances.shape[0]):
        for i in range(syn_distances.shape[1]):
            if syn_distances[syn][i] > distance:
                count += 1
                break
        counts_syn.append(count)
                
    counts_hyp = []
    count = 0
    for hyp in range(hyp_distances.shape[0]):
        for i in range(hyp_distances.shape[1]):
            if hyp_distances[hyp][i] > distance:
                count += 1
                break
        counts_hyp.append(count)

    return max(counts_syn), max(counts_hyp)