import count
from plot import plot_graph
import time

def main():
    #set k maximum distance
    t_0 = time.time()
    k = 10
    cos_alpha = -1
    #set word
    word = 'tumor'
    #x, counts = count.count_words_euclidean(word, k)
    x, counts = count.count_words_cosine(word, cos_alpha)
    print(word)
    print(counts)
    print('Execution time: %.2f' % (time.time() - t_0))
    plot_graph(x, counts)

    

if __name__ == '__main__':
    main()