#mechanism script
import preprocess as pre
import obfuscation as obf
#import evaluate as eva
import time
#usefull imports
import numpy as np
import pandas as pd
import ir_datasets
from multiprocessing import Pool
import WordEmbeddings as we
import os

#glove, vocab = pre.model(dimension = 300)

glove, vocab = pre.initialize_glove_optimized(optimization=False)

model = we.WordEmbeddings(glove, vocab)
#get query
dataset = ir_datasets.load("msmarco-passage/trec-dl-2019/judged")

def obfuscate_parallelize(i, k, n, epsilon):
    #t_0 = time.time()
    '''
     _______  _______  _______  _______  _______  _______  _______  _______  _______  _______ _________ _        _______ 
    (  ____ )(  ____ )(  ____ \(  ____ )(  ____ )(  ___  )(  ____ \(  ____ \(  ____ \(  ____ \\__   __/( (    /|(  ____ \
    | (    )|| (    )|| (    \/| (    )|| (    )|| (   ) || (    \/| (    \/| (    \/| (    \/   ) (   |  \  ( || (    \/
    | (____)|| (____)|| (__    | (____)|| (____)|| |   | || |      | (__    | (_____ | (_____    | |   |   \ | || |      
    |  _____)|     __)|  __)   |  _____)|     __)| |   | || |      |  __)   (_____  )(_____  )   | |   | (\ \) || | ____ 
    | (      | (\ (   | (      | (      | (\ (   | |   | || |      | (            ) |      ) |   | |   | | \   || | \_  )
    | )      | ) \ \__| (____/\| )      | ) \ \__| (___) || (____/\| (____/\/\____) |/\____) |___) (___| )  \  || (___) |
    |/       |/   \__/(_______/|/       |/   \__/(_______)(_______/(_______/\_______)\_______)\_______/|/    )_)(_______)
                                                                                                                     
    '''
    #define model
    query_df = pd.DataFrame(list(dataset.queries_iter()))
    query_id = query_df['query_id']
    query_df = query_df[['text']]

    #print('Finished preprocessing in {:.2f} seconds'.format(time.time() - t_0))

    '''
         _______  ______   _______           _______  _______  _______ __________________ _______  _       
        (  ___  )(  ___ \ (  ____ \|\     /|(  ____ \(  ____ \(  ___  )\__   __/\__   __/(  ___  )( (    /|
        | (   ) || (   ) )| (    \/| )   ( || (    \/| (    \/| (   ) |   ) (      ) (   | (   ) ||  \  ( |
        | |   | || (__/ / | (__    | |   | || (_____ | |      | (___) |   | |      | |   | |   | ||   \ | |
        | |   | ||  __ (  |  __)   | |   | |(_____  )| |      |  ___  |   | |      | |   | |   | || (\ \) |
        | |   | || (  \ \ | (      | |   | |      ) || |      | (   ) |   | |      | |   | |   | || | \   |
        | (___) || )___) )| )      | (___) |/\____) || (____/\| )   ( |   | |   ___) (___| (___) || )  \  |
        (_______)|/ \___/ |/       (_______)\_______)(_______/|/     \|   )_(   \_______/(_______)|/    )_)

    '''
    '''
    As optimization use Pool(number_of_workers for parallelization)
    3 workers = 6min faster for 10 datasets (on 100 obfuscation I get 1h)
    '''
    #define obfuscation parameters
    #parameters required for obfuscation
    #t_0 = time.time()
    df = pd.DataFrame(columns=['original_query'])
    #add original query to df under column 'original_query'
    df['original_query'] = query_df['text']
    #tokenize query
    query_df['text'] = query_df['text'].apply(lambda x: pre.tokenize_query(x))
    #get POS tags
    query_df['text'] = query_df['text'].apply(lambda x: pre.get_POS_tags(x))
    #query_df['text'] is a list of tuples (word, POS)
    #obfuscate query
    #print('Distance: Obfuscation started...')
    #t_0 = time.time()
    df['obfuscated_query_distance'] = query_df.apply(obf.obfuscate, args=(model, k, n, epsilon, 'distance'))
    print('Finished obfuscation distance based')
    #t_0 = time.time()
    df['obfuscated_query_angle'] = query_df.apply(obf.obfuscate, args=(model, k, n, epsilon, 'angle'))
    print('Finished obfuscation angle based')
    #t_0 = time.time()
    df['obfuscated_query_product'] = query_df.apply(obf.obfuscate, args=(model, k, n, epsilon, 'product'))
    df.insert(0, 'query_id', query_id)
    print('Finished obfuscation product based')
    #save df
    df.to_csv('./data/obfuscated-queries_{k}_{n}_{epsilon}_{i}.csv'.format(k=k, n=n, epsilon=epsilon, i=i), index=False, header=True)
    
    query_df = pd.DataFrame(list(dataset.queries_iter()))
    query_df = query_df[['text']]

def main(i):
    k = 2 #size of safe_box, default = 3
    n = 5 #size of candidates_box, default = 10
    epsilon = 17.5 #privacy budget, default = 1

    if i == 0:
        #OBFUSCATION PARAMS
        print('------------------------------------------')
        print('Parameters:')
        print('k: {}'.format(k))
        print('n: {}'.format(n))
        print('epsilon: {}'.format(epsilon))
        print('------------------------------------------')
        obfuscate_parallelize(i, k, n, epsilon)
    else:
        obfuscate_parallelize(i, k, n, epsilon)
    
if __name__ == '__main__':
    t_0 = time.time()
    with Pool(65) as p:
        p.map(main, [i for i in range(0, 100)])
    print('Finished in {:.2f} s.'.format(time.time()-t_0))
