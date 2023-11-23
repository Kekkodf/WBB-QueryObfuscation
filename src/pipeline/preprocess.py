import gensim.downloader as api
import nltk
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')

def model(dimension):
    if dimension == 50:
        return api.load('glove-wiki-gigaword-50')
    elif dimension == 100:
        return api.load('glove-wiki-gigaword-100')
    elif dimension == 200:
        return api.load('glove-wiki-gigaword-200')
    elif dimension == 300:
        return api.load('glove-wiki-gigaword-300')
    else:
        raise ValueError('Invalid dimension: {}'.format(dimension))
    
def load(model):
    return model

'''
Prepare model:
- get vocab
- get embeddings
'''

def get_vocab(model):
    vocab = model.index_to_key
    #with open ('./data/msmarco-passage-vocab.txt', 'r') as f:
    #    filtered = f.read().split('\n')
    ##embeddings = {word: model[word] for word in filtered}
    #vocab_filtered = set(filtered).intersection(set(vocab))
    #print('Vocab size: {}'.format(len(vocab_filtered)))
    #print('Filtered vocab size: {}'.format(len(filtered)))
    #print('Intersection: {}'.format(len(vocab_filtered.intersection(set(filtered)))))
    #remove all words that contain numbers
    vocab = [word for word in vocab if not any(char.isdigit() for char in word)]
    return vocab#, embeddings

'''
Prepare queries:
- get query
- tokenize query
- get POS tags
'''

#def get_query(dataset):

def tokenize_query(query):
    return nltk.word_tokenize(query)
#
def get_POS_tags(query):
    return nltk.pos_tag(query)