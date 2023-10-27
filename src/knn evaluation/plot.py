import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
sns.set_style("whitegrid")
sns.set_context("paper")
sns.set(font_scale=1.5)

def plot_knn_density_values(distances_form_word, word, k):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title(f'Density of the distances of the {k} nearest neighbors of the word {word}')
    ax.set_xlabel('Distance')
    ax.set_ylabel('Density')
    ax = sns.kdeplot(distances_form_word, fill=True, alpha=0.2, legend=False)
    plt.show()
