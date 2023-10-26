import get_hyponyms as gh
import numpy as np
import cosine_sim as cs
import euclidean as eu
import plot
import matplotlib.pyplot as plt

def main():
    word = 'man'
    hyponyms = gh.get_hyponyms(word)
    #k = 10
    cos_alpha = -1
    cosines = {hyp : cs.cosine_sim(word, hyp, cos_alpha) for hyp in hyponyms if hyp!=word}
    distances =  {hyp: eu.euclidean_distance(word, hyp) for hyp in hyponyms if hyp!=word}
    #print(hyponyms)
    #print(len(hyponyms))
    #save dictionaries in file
    with open('./results/hyponyms/txt/'+word+'_cosine.txt', 'w') as f:
        for key in cosines.keys():
            f.write("%s %s\n" % (key, cosines[key]))
    with open('./results/hyponyms/txt/'+word+'_euclidean.txt', 'w') as f:
        for key in distances.keys():
            f.write("%s %s\n" % (key, distances[key]))
    print('Cosines = ' + str(cosines))
    print('Distances = ' + str(distances))
    plot.plot_graph(distances.values(), cosines.values(), hyponyms)

if __name__ == '__main__':
    main()