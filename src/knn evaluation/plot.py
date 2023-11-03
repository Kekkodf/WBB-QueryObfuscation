import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
sns.set_style("whitegrid")
sns.set_context("paper")
sns.set(font_scale=1.5)

def plot_graph(x, count_syn, count_hyp, k, word):
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=x, y=count_syn, label='synonyms')
    sns.lineplot(x=x, y=count_hyp, label='hyponyms')
    plt.xlabel('Distance from a Neighbour')
    plt.ylabel('Number of Synonyms/Hyponyms')
    plt.title('Clustering done by unsupervised {} nearest neighbours - Word: {}'.format(k, word))
    plt.legend(title='Type of word', loc='best')
    plt.show()
