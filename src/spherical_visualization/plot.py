import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
sns.set_context("paper")
sns.set(font_scale=1.5)

def polar_plot_distribution(sample, embeddings_syn, embeddings_hyp, 
                                 cos_similarity_sample, euclidean_distance_sample,  
                                 cos_similarity_synonyms, euclidean_distance_synonyms,
                                 cos_similarity_hyponyms, euclidean_distance_hyponyms, 
                                 word):
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111, projection='polar')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title('Word: %s'%word, fontsize=20)
    ax.set_rmax(max(euclidean_distance_sample.values()))
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
