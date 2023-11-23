#mechanism script
import preprocess as pre
import obfuscation as obf
#import evaluate as eva
import time
#usefull imports
import numpy as np
import pandas as pd
import ir_datasets
from tqdm import tqdm
tqdm.pandas()


def main():
    t_0 = time.time()
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
    model = pre.model(dimension = 300)
    #load model
    model = pre.load(model)
    #preprocessing
    vocab = pre.get_vocab(model)
    #get query
    dataset = ir_datasets.load("msmarco-document/trec-dl-2019/judged")
    query_df = pd.DataFrame(list(dataset.queries_iter()))
    query_df = query_df[['text']]

    print('Finished preprocessing in {:.2f} seconds'.format(time.time() - t_0))

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
    #define obfuscation parameters
    #parameters required for obfuscation
    t_0 = time.time()
    k = 3 #size of safe_box, default = 3
    n = 5 #size of candidates_box, default = 10
    distribution = ('gamma', (1, 2)) #(name, param_1, ..., param_n)

    #OBFUSCATION DISTANCE BASED
    print('------------------------------------------')
    print('Parameters:')
    print('k: {}'.format(k))
    print('n: {}'.format(n))
    print('distribution: {}'.format(distribution))
    print('------------------------------------------')

    df = pd.DataFrame(columns=['original_query'])
    
    #add original query to df under column 'original_query'
    df['original_query'] = query_df['text']
    #tokenize query
    query_df['text'] = query_df['text'].apply(lambda x: pre.tokenize_query(x))
    #get POS tags
    query_df['text'] = query_df['text'].apply(lambda x: pre.get_POS_tags(x))
    #obfuscate query
    print('Distance: Obfuscation started...')
    df['obfuscated_query_distance'] = query_df.apply(obf.obfuscate, args=(vocab, model, k, n, distribution, 'distance'))
    print('Finished obfuscation distance based in {:.2f} s.'.format(time.time()-t_0))
    print(df.head())
    print('------------------------------------------')
    raise ValueError('Stop here')
    t_0 = time.time()
    df['obfuscated_query_angle'] = query_df.apply(obf.obfuscate, args=(vocab, model, k, n, distribution, 'angle'))
    print('Finished obfuscation angle based in {:.2f} s.'.format(time.time()-t_0))
    t_0 = time.time()
    df['obfuscated_query_product'] = query_df.apply(obf.obfuscate, args=(vocab, model, k, n, distribution, 'product'))
    print('Finished obfuscation product based in {:.2f} s.'.format(time.time()-t_0))
    df['original_query'] = query_df['text'] 
    #save df
    df.to_csv('results/pipeline/obfuscated_queries_{k}_{n}_{distribution}.csv'.format(k=k, n=n, distribution=distribution), index=False, header=True)
    print('Finished obfuscation distance based in {:.2f} s.'.format(time.time()-t_0))
    
if __name__ == '__main__':
    main()