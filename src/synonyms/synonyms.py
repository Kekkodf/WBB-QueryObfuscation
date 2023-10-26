import get_synonyms as gs
import numpy as np
import cosine_sim as cs
import euclidean as eu
import plot
import matplotlib.pyplot as plt

def main():
    word = 'neoplasm'
    synonyms = gs.get_synonyms(word)
    #k = 10
    cos_alpha = -1
    cosines = {syn : cs.cosine_sim(word, syn, cos_alpha) for syn in synonyms if syn!=word}
    distances =  {syn: eu.euclidean_distance(word, syn) for syn in synonyms if syn!=word}
    #print(synonyms)
    #print(len(synonyms))
    #save dictionaries in file
    with open('./results/synonyms/txt/'+word+'_cosine.txt', 'w') as f:
        for key in cosines.keys():
            f.write("%s %s\n" % (key, cosines[key]))
    with open('./results/synonyms/txt/'+word+'_euclidean.txt', 'w') as f:
        for key in distances.keys():
            f.write("%s %s\n" % (key, distances[key]))
    print('Cosines = ' + str(cosines))
    print('Distances = ' + str(distances))
    plot.plot_graph(distances.values(), cosines.values(), synonyms)

if __name__ == '__main__':
    main()