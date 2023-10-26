import get_synonyms as gs
import numpy as np
import cosine_sim as cs
import euclidean as eu
import plot
import get_hyponyms as gh

def main():
    word = 'tumor'
    print('-----------------------------------')
    print('Word: %s'%word)
    print('-----------------------------------')
    print('Synonyms')
    print('-----------------------------------')
    synonyms = gs.get_synonyms(word)
    #k = 10
    cos_alpha = -1
    cosines_syn = {syn : cs.cosine_sim(word, syn, cos_alpha) for syn in synonyms if syn!=word}
    distances_syn =  {syn: eu.euclidean_distance(word, syn) for syn in synonyms if syn!=word}
    #print(synonyms)
    #print(len(synonyms))
    #save dictionaries in file
    print('Cosines = ' + str(cosines_syn))
    print('Distances = ' + str(distances_syn))
    with open('./results/synonyms/txt/'+word+'_cosine.txt', 'w') as f:
        for key in cosines_syn.keys():
            f.write("%s %s\n" % (key, cosines_syn[key]))
    with open('./results/synonyms/txt/'+word+'_euclidean.txt', 'w') as f:
        for key in distances_syn.keys():
            f.write("%s %s\n" % (key, distances_syn[key]))

    print('-----------------------------------')
    print('Hyponyms')
    print('-----------------------------------')
    hyponyms = gh.get_hyponyms(word)
    #k = 10
    cos_alpha = -1
    cosines_hyp = {hyp : cs.cosine_sim(word, hyp, cos_alpha) for hyp in hyponyms if hyp!=word}
    distances_hyp =  {hyp: eu.euclidean_distance(word, hyp) for hyp in hyponyms if hyp!=word}
    #print(hyponyms)
    #print(len(hyponyms))
    #save dictionaries in file
    with open('./results/hyponyms/txt/'+word+'_cosine.txt', 'w') as f:
        for key in cosines_hyp.keys():
            f.write("%s %s\n" % (key, cosines_hyp[key]))
    with open('./results/hyponyms/txt/'+word+'_euclidean.txt', 'w') as f:
        for key in distances_hyp.keys():
            f.write("%s %s\n" % (key, distances_hyp[key]))
    print('Cosines = ' + str(cosines_hyp))
    print('Distances = ' + str(distances_hyp))
    
    plot.plot_graph(distances_syn.values(), cosines_syn.values(), distances_hyp.values(), cosines_hyp.values())

if __name__ == '__main__':
    main()