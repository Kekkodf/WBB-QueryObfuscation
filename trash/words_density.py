import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import time

## PREPARE DATA
#read word embedding file
def read_file(file_name):
    t_0 = time.time()
    with open(file_name, 'r') as f:
        lines = f.readlines()
    t_f = time.time()-t_0
    print('Time to read file: %.2f sec.' % t_f)
    return lines

#save word and vector
def save_word_vector(lines):
    t_0 = time.time()
    words = []
    vectors = []
    for line in lines:
        word_vector = line.split()
        word = word_vector[0]
        vector = word_vector[1:]
        words.append(word)
        vectors.append(vector)
    t_f = time.time()-t_0
    print('Words and Vectors saved successfully in %.2f sec.' % t_f)
    return words, vectors

#convert to dataframe
def convert_to_dataframe(words, vectors):
    t_0 = time.time()
    df = pd.DataFrame(vectors, index=words)
    df = df.astype(float)
    t_f = time.time()-t_0
    print('Time to convert in df: %.2f sec.' % t_f)
    return df

#filter stopwords
def filter_stopwords(df):
    t_0 = time.time()
    try:
        stop_words = set(stopwords.words('english'))
        for row in df.index:
            if row in stop_words:
                df = df.drop(row)
            #remove punctuation
            elif row in ['.', ',', '?', '!', ':', ';', '"', "'", '(', ')', '[', ']', '{', '}', '-', '\'\'', '``', '\'s', '--']:
                df = df.drop(row)
    except KeyError:
        pass
    t_f = time.time()-t_0
    print('Time to filter stopwords: %.2f sec.' % t_f)
    return df

## COMPUTE DENSITY

#define function epsilon
def function_epsilon(epsilon):
    name = '1/epsilon'
    return 1/epsilon #1/f(x) f(x) must be strictly positive and strictly increasing

def compute_density(df, df_sampled, epsilon):
    t_0 = time.time()
    density_results = {}
    #for each word in df.index
    for word in df_sampled.index:
        #count the number of words with a distance less than f_epsilon to the selected word
        count = 0
        for other_word in df.index:
            if word != other_word:
                #compute Euclidean distance using the columns of df as coordinates
                distance = euclidean_distance(df_sampled.loc[word], df.loc[other_word])
                if distance <= epsilon:
                    count += 1
        density_results[word] = count
    t_f = time.time() - t_0
    print('Time to compute density: %.2f sec.' % t_f)
    #save results in a txt file
    with open('results/density_results.txt', 'w') as f:
        for word, density in density_results.items():
            f.write('%s:%s\n' % (word, density))
    return density_results

#dists = np.sqrt(np.sum((dots[:, np.newaxis] - dots) ** 2, axis=2))
#formula for computing the Euclidean distance between two vectors

# Function to calculate Euclidean distance
def euclidean_distance(vector1, vector2):
    return np.linalg.norm(vector1 - vector2)

## PLOT RESULTS
def plot_density_results(density_results):
    words = list(density_results.keys())
    densities = list(density_results.values())
    plt.figure(figsize=(10, 6))
    plt.bar(words, densities)
    plt.xlabel('Words')
    plt.ylabel('Density')
    plt.title('Density of Words')
    plt.xticks(rotation=45) 
    plt.show()

def save_results(epsilon, density_results):
    with open ('results/density_results.txt', 'w') as f:
        f.write('Density of Words\n')
        f.write('----------------\n')
        f.write('Epsilon: %s \n' % epsilon)
        f.write('Function epsilon: %s \n' % function_epsilon(epsilon))
        f.write('----------------\n')
        f.write('Words\tDensity\n')
        for word, density in density_results.items():
            f.write('%s\t%s\n' % (word, density))

def count_words(df_sampled, df, distances):
    results = []
    for word in df_sampled.index:
        count = 0
        partial = []
        for other_word in df.index:
            for distance in distances:
                if euclidean_distance(df_sampled.loc[word], df.loc[other_word]) <= distance:
                   count += 1
                partial.append((word, count, distance))
        results.append(partial)
    return results

def main():
    #read file txt
    lines = read_file('data/glove_6B_50d.txt')
    #save word and vector
    words, vector = save_word_vector(lines)
    #create df of dictionary words and vectors
    df = filter_stopwords(convert_to_dataframe(words, vector))
    #save df
    df.to_csv('data/df.csv')
    # sample words
    df = df.sample(n=100)
    df_sampled = df.sample(n=5)
    distances = [1, 5, 7, 8, 10]
    results = count_words(df_sampled, df, distances)
    #save in a df
    print(results)


    

    
    #define set of epsilon (remove 0 from epsilons to avoid errors)
    #epsilon = 0.2
    #compute function epsilon
    #f_epsilon = function_epsilon(epsilon)
    #print('f_epsilon: %.2f' % f_epsilon)
    #compute density
    #density_results = compute_density(df, df_sampled, f_epsilon)
    #plot density
    #plot_density_results(density_results)
    #print(density_results)
    #save_results(epsilon, density_results)

if __name__ == '__main__':
    main()