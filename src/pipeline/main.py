#mechanism script
import preprocess as pre
import obfuscation as obf
#import evaluate as eva
import time
#usefull imports
import numpy as np
import pandas as pd
import ir_datasets


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
    vocab, _ = pre.get_vocab(model)
    #get query
    dataset = ir_datasets.load("msmarco-document/trec-dl-2019/judged")
    query_df = pd.DataFrame(dataset.queries_iter(), columns=['id', 'query'])
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
    k = 10 #size of safe_box, default = 3
    n = 3 #size of candidates_box, default = 10
    distribution = ('gamma', (1, 2)) #(name, param_1, ..., param_n)

    #OBFUSCATION DISTANCE BASED
    new_query = []
    print('------------------------------------------')
    print('Parameters:')
    print('k: {}'.format(k))
    print('n: {}'.format(n))
    print('distribution: {}'.format(distribution))
    print('------------------------------------------')
    print('Obfuscating...')

    df = pd.DataFrame(columns=['original_query', 
                               'obfuscated_query_distance', 
                               'obfuscated_query_angle', 
                               'obfuscated_query_ratio'])
    
    for query_og in query_df['query']:
        #tokenize query and get POS tags
        query = pre.get_POS_tags(pre.tokenize_query(query_og))
        #obfuscation variables
        new_query_distance = []
        new_query_angle = []
        new_query_ratio = []
        #iterate over all words in te query
        for word in query:
            if word[0] in vocab:
                if word[1] == 'NN' or word[1] == 'NNS' or word[1] == 'JJ' or word[1] == 'NNP' or word[1] == 'NNPS':
                    #get rank and partitions
                    rank_distance = obf.compute_rank(word[0], vocab, model, 'distance')
                    safe_box_distance, candidates_box_distance = obf.partitions(rank_distance, k, n)
                    rank_angle = obf.compute_rank(word[0], vocab, model, 'angle')
                    safe_box_angle, candidates_box_angle = obf.partitions(rank_angle, k, n)
                    rank_ratio = obf.compute_rank(word[0], vocab, model, 'ratio')
                    safe_box_ratio, candidates_box_ratio = obf.partitions(rank_ratio, k, n)
                    #start sampling
                    #obfuscation
                    substitution_distance = obf.candidate_extraction(candidates_box_distance, distribution)
                    substitution_angle = obf.candidate_extraction(candidates_box_angle, distribution)
                    substitution_ratio = obf.candidate_extraction(candidates_box_ratio, distribution)
                    #append new word
                    new_query_distance.append(substitution_distance)
                    new_query_angle.append(substitution_angle)
                    new_query_ratio.append(substitution_ratio)
                else:
                    new_query_distance.append(word[0])
                    new_query_angle.append(word[0])
                    new_query_ratio.append(word[0])
            else:
                new_query_distance.append(word[0])
                new_query_angle.append(word[0])
                new_query_ratio.append(word[0])
        #add to df
        obfuscated_query_distance = ' '.join(new_query_distance)
        obfuscated_query_angle = ' '.join(new_query_angle)
        obfuscated_query_ratio = ' '.join(new_query_ratio)

        df_row = {'original_query': query_og, 
                  'obfuscated_query_distance': obfuscated_query_distance, 
                  'obfuscated_query_angle': obfuscated_query_angle, 
                  'obfuscated_query_ratio': obfuscated_query_ratio}
        
        df = pd.concat([df, pd.DataFrame(df_row, index=[0])], ignore_index=True)

    #save df
    df.to_csv('results/pipeline/obfuscated_queries.csv', index=False)
    print('Finished obfuscation distance based in {:.2f} s.'.format(time.time()-t_0))
    
if __name__ == '__main__':
    main()