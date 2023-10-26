import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_graph(x,y, labels):
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, marker='o', color='blue')
    #set point label
    
    #for i, txt in enumerate(labels):
        #plt.annotate(txt, (x, y))
    plt.xlabel('Distance')
    plt.ylabel('Cosine Symilarity')
    plt.title('Cosine Similarity vs Distance')
    plt.show()
    return 0