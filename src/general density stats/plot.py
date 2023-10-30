import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_graph(x,y):
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, marker='o', color='blue')
    plt.plot(x, y, color='red')
    plt.xlabel('Euclidean distance')
    plt.ylabel('Number of words')
    plt.title('Number of words vs Distance')
    plt.show()
    return 0