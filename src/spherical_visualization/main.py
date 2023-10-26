import get_hyponyms as gh
import get_synonyms as gs
import cosine_sim as cs
import euclidean as eu
import embeddings as emb
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
sns.set_context("paper")
sns.set(font_scale=1.5)

def main():
    word = 'car'
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
    sample = np.random.choice(list(embeddings_dict.keys()), 1000)
    #get cosine similarity
    cos_similarity_sample = {wrd : cs.cosine_sim(word, wrd, -1) for wrd in sample}
    #get euclidean distance
    euclidean_distance_sample = {wrd : eu.euclidean_distance(word, wrd) for wrd in sample}
    #get maximum euclidean distance
    max_euclidean_distance = max(euclidean_distance_sample.values())

    #plot a scatterplot in polar coordinates
    
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111, projection='polar')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title('Word: %s'%word, fontsize=20)
    ax.set_rmax(max_euclidean_distance)
    ax.set_rticks([0.25, 0.5, 0.75, 1.0, 1.25, 1.5])
    ax.grid(True)
    
    for wrd in sample:
        ax.scatter(cos_similarity_sample[wrd], euclidean_distance_sample[wrd], marker='o', s=100, color='grey', alpha=0.2)
    for syn in embeddings_syn.keys():
        ax.scatter(cos_similarity_synonyms[syn], euclidean_distance_synonyms[syn], marker='x', s=100, label=syn)
    for hyp in embeddings_hyp.keys():
        ax.scatter(cos_similarity_hyponyms[hyp], euclidean_distance_hyponyms[hyp], marker='s', s=100, label=hyp)
    #place thenlegend outside the plot
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()
    
if __name__ == '__main__':
    main()