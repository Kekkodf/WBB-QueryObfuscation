import get_hyponyms as gh
import get_synonyms as gs
import embeddings as emb
#import knn
import plot
import numpy as np
import NearestNeighbours as nn

def main():
    word = 'car'
    embeddings_dict = emb.embeddings_dict
    word_embedding = embeddings_dict[word]
    print('-----------------------------------')
    print('Word: %s'%word)
    print('-----------------------------------')
    print('Embedding: %s'%word_embedding)
    print('-----------------------------------')
    #get synonyms and hyponyms
    synonyms = gs.get_synonyms(word)
    hyponyms = gh.get_hyponyms(word)
    embeddings_syn = gs.get_synonyms_embedding(embeddings_dict, synonyms, word)
    embeddings_hyp = gh.get_hyponyms_embedding(embeddings_dict, hyponyms, word)
    #print number of synonyms and hyponyms
    print('Number of synonyms: %s'%len(synonyms))
    print('Number of hyponyms: %s'%len(hyponyms))
    print('-----------------------------------')
    
    #get the distances and indices of the 5 nearest neighbors of the word
    k = 10
    distances_form_word = []
    indices_from_word = []
    #call nn
    nn_word = nn.NearestNeighbours(word_embedding, k)
    distances_form_word.append(nn_word.get_distances())
    indices_from_word.append(nn_word.get_indices())
    #call nn for synonyms
    distances_form_syn = []
    indices_from_syn = []
    for syn in embeddings_syn.values():
        nn_syn = nn.NearestNeighbours(syn, k)
        distances_form_syn.append(nn_syn.get_distances())
        indices_from_syn.append(nn_syn.get_indices())
    #call nn for hyponyms
    distances_form_hyp = []
    indices_from_hyp = []
    for hyp in embeddings_hyp.values():
        nn_hyp = nn.NearestNeighbours(hyp, k)
        distances_form_hyp.append(nn_hyp.get_distances())
        indices_from_hyp.append(nn_hyp.get_indices())
    #plot the density of the distances of the k nearest neighbors
    plot.plot_knn_density_values(distances_form_word, word, k)
    plot.plot_knn_density_values(distances_form_syn, 'synonyms', k)
    plot.plot_knn_density_values(distances_form_hyp, 'hyponyms', k)
        
    
if __name__ == '__main__':
    main()