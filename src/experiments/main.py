import glove
import queries
import seaborn as sns
import obfuscator
import math
import prepare_data as prep
import random
import tqdm
import nltk
nltk.download('averaged_perceptron_tagger')
sns.set_style("whitegrid")
sns.set_context("paper")
sns.set(font_scale=1.5)

def main():
    model = glove.model(dimension=50)
    #get glove embeddings
    model = glove.load(model)

    data = queries.get_data()

    vocab_embeddings_queries = prep.get_vocab_embeddings(data, model)

    query = 'what slows down the flow of blood'        
    
    print('-'*50)
    tokenized_query = query.split()
    tokenized_query = nltk.pos_tag(tokenized_query)
    print('Tokenized query: {}'.format(tokenized_query))
    
    k=10
    new_query_distance_based = []
    new_query_angle_based = []
    for word, tag in tokenized_query:
        if tag == 'NN':
            #obfuscate
            rank_distance, secure_box_distance = obfuscator.get_safe_box(word, vocab_embeddings_queries, model, k, 'distance')
            rank_angle, secure_box_angle = obfuscator.get_safe_box(word, vocab_embeddings_queries, model, k, 'angle')
            #remove secure box from rank
            rank_distance = [x for x in rank_distance if x not in secure_box_distance]
            rank_angle = [x for x in rank_angle if x not in secure_box_angle]
            #get new query
        else:
            secure_box_distance = [(word, 0)]
            secure_box_angle = [(word, 0)]
        new_query_distance_based.append(random.choice(secure_box_distance)[0])
        new_query_angle_based.append(random.choice(secure_box_angle)[0])

    print('Parameter used: k = {}'.format(k))
    print('-'*50)
    print('Original query: {}'.format(query))
    print('New query distance based: {}'.format(' '.join(new_query_distance_based)))
    print('New query angle based: {}'.format(' '.join(new_query_angle_based)))

if __name__ == '__main__':
    main()