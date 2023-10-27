import get_hyponyms as gh
import get_synonyms as gs
import cosine_sim as cs
import euclidean as eu
import embeddings as emb
import plot
import numpy as np

def main():
    word = 'death'
    embeddings_dict = emb.embeddings_dict
    print('-----------------------------------')
    print('Word: %s'%word)
    print('-----------------------------------')
    #get synonyms and hyponyms
    synonyms = gs.get_synonyms(word)
    hyponyms = gh.get_hyponyms(word)
    embeddings_syn = gs.get_synonyms_embedding(embeddings_dict, synonyms, word)
    embeddings_hyp = gh.get_hyponyms_embedding(embeddings_dict, hyponyms, word)
    #get cosine similarity
    cos_similarity_synonyms = {syn : cs.cosine_sim(word, syn, -1) for syn in embeddings_syn.keys()}
    cos_similarity_hyponyms = {hyp : cs.cosine_sim(word, hyp, -1) for hyp in embeddings_hyp.keys()}
    #get euclidean distance
    euclidean_distance_synonyms = {syn : eu.euclidean_distance(word, syn) for syn in embeddings_syn.keys()}
    euclidean_distance_hyponyms = {hyp : eu.euclidean_distance(word, hyp) for hyp in embeddings_hyp.keys()}

    #sample from embeddings_dict
    sample = np.random.choice(list(embeddings_dict.keys()), 10000)
    #get cosine similarity
    cos_similarity_sample = {wrd : cs.cosine_sim(word, wrd, -1) for wrd in sample}
    #get euclidean distance
    euclidean_distance_sample = {wrd : eu.euclidean_distance(word, wrd) for wrd in sample}
    #get maximum euclidean distance
    max_euclidean_distance = max(euclidean_distance_sample.values())

    #plot a scatterplot in polar coordinates
    plot.polar_plot_distribution(sample, embeddings_syn, embeddings_hyp, 
                                 cos_similarity_sample, euclidean_distance_sample,  
                                 cos_similarity_synonyms, euclidean_distance_synonyms,
                                 cos_similarity_hyponyms, euclidean_distance_hyponyms, 
                                 word)
    
if __name__ == '__main__':
    main()