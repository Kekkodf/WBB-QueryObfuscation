import glove
import queries
import seaborn as sns
import obfuscator
import prepare_data as prep
import nltk
#nltk.download('averaged_perceptron_tagger')

def main():
    #get most similar words
    print('-'*50)
    print('Starting experiment...')
    print('-'*50)
    model = glove.model(dimension=50)
    #get glove embeddings
    model = glove.load(model)

    print(model.most_similar('flow'))

    data = queries.get_data()

    vocab_embeddings_queries = prep.get_vocab_embeddings(data, model)

    query = 'what slows down the flow of blood' #melanoma skin cancer symptoms #what causes ankle blisters  
                                                #what slows down the flow of blood
    
    #print('-'*50)
    tokenized_query = query.split()
    tokenized_query = nltk.pos_tag(tokenized_query)
    #define mechanism parameters
    k = 3
    scale_param = 2
    shape = 1
    feature = 'angle'
    distribution = 'gamma'
    N = 100
    #do obfuscation
    new_query = obfuscator.get_obfuscated_query(tokenized_query, feature, k, distribution, scale_param, shape, model, vocab_embeddings_queries, N)
    print('-'*50)
    print('Parameter used: k={}, scale_param={}, shape={}, feature={}, distribution={}, number of iterations={}'.format(k, scale_param, shape, feature, distribution, N))
    print('-'*50)
    print('Original query: {}'.format(query))
    print('New query: {}'.format(' '.join(new_query)))

if __name__ == '__main__':
    main()