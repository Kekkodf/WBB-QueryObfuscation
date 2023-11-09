import glove
import queries
import seaborn as sns
import obfuscator
import math
import prepare_data as prep
import random
import tqdm
import nltk
#nltk.download('averaged_perceptron_tagger')

def main():
    model = glove.model(dimension=50)
    #get glove embeddings
    model = glove.load(model)

    data = queries.get_data()

    vocab_embeddings_queries = prep.get_vocab_embeddings(data, model)

    query = 'how many people use google in one day' #melanoma skin cancer symptoms #what causes ankle blisters  
                                                #what slows down the flow of blood
    
    print('-'*50)
    tokenized_query = query.split()
    tokenized_query = nltk.pos_tag(tokenized_query)
    print('Tokenized query: {}'.format(tokenized_query))
    #define mechanism parameters
    k=100
    scale_param = 2
    shape = 1
    new_query_distance_based = []
    new_query_angle_based = []
    for word, tag in tokenized_query:
        #if tag == 'NN' or tag == 'NNS' or tag == 'JJ':
            #get boxes
            rank_distance, candidates_distance, safe_box_distance = obfuscator.get_safe_box(word, vocab_embeddings_queries, model, k, 'distance')
            rank_angle, candidates_angle, safe_box_distance = obfuscator.get_safe_box(word, vocab_embeddings_queries, model, k, 'angle')
        #else:
            candidates_distance = [(word, 0)]
            candidates_angle = [(word, 0)]

        #print('-'*50)
        #print('Word: {}'.format(word))
        #print('Candidates distance based: {}'.format(candidates_distance))
        #print('Candidates angle based: {}'.format(candidates_angle))
        #print('-'*50)
        
        #get weights
        #laplace
        #weights = obfuscator.get_weights('laplace', len(candidates_distance), scale_param) 
        #normal
        #weights = obfuscator.get_weights('normal', len(candidates_distance), scale_param) 
        #gamma
        #weights = obfuscator.get_weights('gamma', len(candidates_distance), scale_param, shape) 
        #uniform
            weights = obfuscator.get_weights('uniform', len(candidates_distance), scale_param) 

            #get new query
            new_word_distance_based = obfuscator.get_new_word(candidates_distance, weights)
            new_word_angle_based = obfuscator.get_new_word(candidates_angle, weights)

            new_query_distance_based.append(new_word_distance_based)
            new_query_angle_based.append(new_word_angle_based)

    print('Parameter used: k = {}'.format(k))
    print('-'*50)
    print('Original query: {}'.format(query))
    print('New query distance based: {}'.format(' '.join(new_query_distance_based)))
    print('New query angle based: {}'.format(' '.join(new_query_angle_based)))

if __name__ == '__main__':
    main()