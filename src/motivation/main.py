import get_hyponyms as gh
import get_synonyms as gs
import cosine_sim as cs
import euclidean as eu
import embeddings as emb
import plot
import numpy as np
import pandas as pd

def main():
    word = 'medicine'
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
    sample = embeddings_dict.keys()
    #get cosine similarity
    cos_similarity_sample = {wrd : cs.cosine_sim(word, wrd, -1) for wrd in sample}
    #get euclidean distance
    euclidean_distance_sample = {wrd : eu.euclidean_distance(word, wrd) for wrd in sample}

    #create a df with colums: distance, type
    data = pd.DataFrame(columns=['distance', 'type'])
    data['distance'] = euclidean_distance_sample.values()
    data['type'] = 'random words'
    data = data.append(pd.DataFrame({'distance': euclidean_distance_synonyms.values(), 'type': 'synonyms/hyponyms'}))
    data = data.append(pd.DataFrame({'distance': euclidean_distance_hyponyms.values(), 'type': 'synonyms/hyponyms'}))
    #plot a density plot
    plot.plot_graph(data, word, 'Euclidean')

    data = pd.DataFrame(columns=['cosine', 'type'])
    data['cosine'] = cos_similarity_sample.values()
    data['type'] = 'random words'
    data = data.append(pd.DataFrame({'cosine': cos_similarity_synonyms.values(), 'type': 'synonyms/hyponyms'}))
    data = data.append(pd.DataFrame({'cosine': cos_similarity_hyponyms.values(), 'type': 'synonyms/hyponyms'}))
    #plot a density plot
    plot.plot_graph(data, word, 'Cosine')

     
if __name__ == '__main__':
    main()