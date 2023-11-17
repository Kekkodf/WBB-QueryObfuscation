#mechanism script
import preprocess as pre
import obfuscation as obf
#import evaluate as eva
import time
#usefull imports
import numpy as np
import pandas as pd
from custom_distribution import CustomDistribution


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
    query_og = 'what slows down the flow of blood' #melanoma skin cancer symptoms #what causes ankle blisters  
                                                #what slows down the flow of blood
    #create a df with columns: original query, obfuscated query
    df = pd.DataFrame(columns=['original_query', 'obfuscated_query'])
    #tokenize query and get POS tags
    query = pre.get_POS_tags(pre.tokenize_query(query_og))

    print('Finished preprocessing in {} seconds'.format(time.time() - t_0))

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
    n = 5 #size of candidates_box, default = 10
    distribution = ('gamma', (1, 2)) #(name, param_1, ..., param_n)

    #OBFUSCATION DISTANCE BASED
    new_query = []
    feature = 'distance' 
    print('------------------------------------------')
    print('Parameters:')
    print('k: {}'.format(k))
    print('n: {}'.format(n))
    print('distribution: {}'.format(distribution))
    print('feature: {}'.format(feature))
    #obfuscation
    for i in range(0,10):
        for word in query:
            if word[1] == 'NN' or word[1] == 'NNS' or word[1] == 'JJ' or word[1] == 'NNP' or word[1] == 'NNPS':
                #get rank
                rank = obf.compute_rank(word[0], vocab, model, feature)
                safe_box, candidates_box = obf.partitions(rank, k, n)
                #print('Safe box for word {}: {}'.format(word, safe_box))
                #obfuscation
                substitution = obf.candidate_extraction(candidates_box, distribution)
                #print('Substitution for word {}: {}'.format(word, substitution))
                new_query.append(substitution)
            else:
                new_query.append(word[0])
        df.loc[i] = [query_og, ' '.join(new_query)]
        new_query = []
    #save df
    df.to_csv('results/pipeline/obfuscated_queries_distance.csv', index=False)

    #OBFUSCATION ANGLE BASED
    new_query = []
    feature = 'angle'
    print('------------------------------------------')
    print('Parameters:')
    print('k: {}'.format(k))
    print('n: {}'.format(n))
    print('distribution: {}'.format(distribution))
    print('feature: {}'.format(feature))
    #obfuscation
    for i in range(0,10):
        for word in query:
            if word[1] == 'NN' or word[1] == 'NNS' or word[1] == 'JJ' or word[1] == 'NNP' or word[1] == 'NNPS':
                #get rank
                rank = obf.compute_rank(word[0], vocab, model, feature)
                safe_box, candidates_box = obf.partitions(rank, k, n)
                #print('Safe box for word {}: {}'.format(word, safe_box))
                #obfuscation
                substitution = obf.candidate_extraction(candidates_box, distribution)
                #print('Substitution for word {}: {}'.format(word, substitution))
                new_query.append(substitution)
            else:
                new_query.append(word[0])
        df.loc[i] = [query_og, ' '.join(new_query)]
        new_query = []
    #save df
    df.to_csv('results/pipeline/obfuscated_queries_angle.csv', index=False)

    #OBFUSCATION RATIO BASED    
    new_query = []
    feature = 'ratio'
    print('------------------------------------------')
    print('Parameters:')
    print('k: {}'.format(k))
    print('n: {}'.format(n))
    print('distribution: {}'.format(distribution))
    print('feature: {}'.format(feature))
    #obfuscation
    for i in range(0,10):
        for word in query:
            if word[1] == 'NN' or word[1] == 'NNS' or word[1] == 'JJ' or word[1] == 'NNP' or word[1] == 'NNPS':
                #get rank
                rank = obf.compute_rank(word[0], vocab, model, feature)
                safe_box, candidates_box = obf.partitions(rank, k, n)
                #print('Safe box for word {}: {}'.format(word, safe_box))
                #obfuscation
                substitution = obf.candidate_extraction(candidates_box, distribution)
                #print('Substitution for word {}: {}'.format(word, substitution))
                new_query.append(substitution)
            else:
                new_query.append(word[0])
        df.loc[i] = [query_og, ' '.join(new_query)]
        new_query = []
    #save df
    df.to_csv('results/pipeline/obfuscated_queries_ratio.csv', index=False)
    
if __name__ == '__main__':
    main()