import get_synonyms
import get_hyponyms
import get_neighbors
import numpy as np
import plot
import pandas as pd
import embeddings as emb
from sklearn.neighbors import NearestNeighbors
import seaborn as sns
import matplotlib.pyplot as plt



embeddings_dict = emb.embeddings_dict

def main():
    #define variables
    word = 'car'
    k = 50
    distance = 8
    synonyms = get_synonyms.get_synonyms(word)
    hyponyms = get_hyponyms.get_hyponyms(word)
    
    #define knn model
    X = np.array(list(embeddings_dict.values()))
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(X)

    #save distances
    syn_distances_per_syn = []
    hyp_distances_per_hyp = []
    for syn in synonyms:
        var = get_neighbors.get_neighbors(syn, nbrs)
        if var is not None:
            syn_distances_per_syn.append(var)
    for hyp in hyponyms:
        var = get_neighbors.get_neighbors(hyp, nbrs)
        if var is not None:
            hyp_distances_per_hyp.append(var)
        
    #flatten list
    syn_distances = [item for sublist in syn_distances_per_syn for item in sublist]
    hyp_distances = [item for sublist in hyp_distances_per_hyp for item in sublist]

    #create a metrix stacking all distances
    syn_distances = np.vstack(syn_distances)
    hyp_distances = np.vstack(hyp_distances)
    #save in df
    print(syn_distances.shape)
    print(hyp_distances.shape)
    #print(get_neighbors.count_neighbours(syn_distances, hyp_distances, distance))
    results_syn = []
    results_hyp = []
    x = np.linspace(0, distance, 1000)
    for distances in x:
        counts_syn, counts_hyp = get_neighbors.count_neighbours(syn_distances, hyp_distances, distances)
        results_syn.append(counts_syn)
        results_hyp.append(counts_hyp)

    #plot
    #plot as x = range 1 to distance, y = results_syn
    #plot as x = range 1 to distance, y = results_hyp
    plot.plot_graph(x, results_syn, results_hyp, k, word)


    
if __name__ == "__main__":
    main()