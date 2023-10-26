import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_graph(x1, y1, x2, y2):
    plt.figure(figsize=(10, 6))
    plt.scatter(x1, y1, marker='o', color='blue', label='Synonyms')
    plt.scatter(x2, y2, marker='o', color='red', label='Hyponyms')
    plt.xlabel('Distance')
    plt.ylabel('Cosine Symilarity')
    plt.title('Cosine Similarity vs Distance of Synonyms vs Hyponyms')
    plt.legend(loc='best')

    plt.show()
    return 0