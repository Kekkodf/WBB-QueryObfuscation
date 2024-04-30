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
dataset = ir_datasets.load("msmarco-passage/trec-dl-2019/judged") #msmarco
#dataset = ir_datasets.load("disks45/nocr/trec-robust-2004") #robust

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
    query_df = query_df[['text']] #msmarco
    #query_df = query_df[['title']] #robust
    #remove all \n from query
    query_df['text'] = query_df['text'].apply(lambda x: pre.remove_newline(x)) #msmarco
    #query_df['title'] = query_df['title'].apply(lambda x: pre.remove_newline(x)) #robust

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
    df['original_query'] = query_df['text']#msmarco
    #df['original_query'] = query_df['title']#robust
    #tokenize query
    query_df['text'] = query_df['text'].apply(lambda x: pre.tokenize_query(x)) #msmarco
    #query_df['title'] = query_df['title'].apply(lambda x: pre.tokenize_query(x)) #robust
    #get POS tags
    query_df['text'] = query_df['text'].apply(lambda x: pre.get_POS_tags(x)) #msmarco
    #query_df['title'] = query_df['title'].apply(lambda x: pre.get_POS_tags(x)) #robust
    #query_df['text'] is a list of tuples (word, POS)
    #obfuscate query
    #print('Distance: Obfuscation started...')
    #t_0 = time.time()
    #df['obfuscated_query_distance'] = query_df.apply(obf.obfuscate, args=(model, k, n, epsilon, 'distance'))
    #print('Finished obfuscation distance based')
    #t_0 = time.time()
    df['obfuscated_query_angle'] = query_df.apply(obf.obfuscate, args=(model, k, n, epsilon, 'angle'))
    print('Finished obfuscation angle based')
    #t_0 = time.time()
    #df['obfuscated_query_product'] = query_df.apply(obf.obfuscate, args=(model, k, n, epsilon, 'product'))
    df.insert(0, 'query_id', query_id)
    #print('Finished obfuscation product based')
    #save df
    df.to_csv('./data/obfuscated-queries_{k}_{n}_{epsilon}_{i}.csv'.format(k=k, n=n, epsilon=epsilon, i=i), index=False, header=True)
    
    query_df = pd.DataFrame(list(dataset.queries_iter()))
    query_df = query_df[['text']] #msmarco
    #query_df = query_df[['title']] #robust
    #remove all \n from query
    query_df['text'] = query_df['text'].apply(lambda x: pre.remove_newline(x)) #msmarco
    #query_df['title'] = query_df['title'].apply(lambda x: pre.remove_newline(x)) #robust

def main(args):
    i, eps = args
    k = 4 #size of safe_box                 2,4
    n = 1000 #size of candidates_box          15,20
    epsilon = eps #privacy budget            1,5,10,12.5,15,17.5,20,50

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
    epsilons = [0.001, 0.001, 0.1, 1, 10, 100, 1000]
    args = [(i,eps) for eps in epsilons for i in range(0,20)]
    
    t_0 = time.time()
    with Pool(50) as p:
        p.map(main, args)
    print('Finished in {:.2f} s.'.format(time.time()-t_0))
    path_to_datasets = './data/'
    path_to_results = './data/final/'
    for eps in epsilons:
        k = 4       #4, 4
        n = 1000     #5, 250
        epsilon = eps
        #read all csv files in path_to_datasets
        list_of_datasets = [dataset for dataset in os.listdir(path_to_datasets) if dataset.startswith(f'obfuscated-queries_{k}_{n}_{epsilon}')]
        #merge all datasets
        df = pd.concat([pd.read_csv(path_to_datasets + dataset, sep=",") for dataset in list_of_datasets])
        #order by query_id
        df = df.sort_values(by=['query_id'])
        #reset index
        df = df.reset_index(drop=True)
        #collection = 'robust-04'
        collection = 'msmarco-dl19'
        dataset_name = 'dataset-{k}-{n}-{epsilon}_{collection}.csv'.format(k=k, n=n, epsilon = epsilon, collection=collection)
        #save merged dataset
        df.to_csv(path_to_results+f'/{collection}/' + dataset_name, index=False, header=True, sep=',')