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
    # Access the vocabulary using index_to_key
    vocab = model.index_to_key
    # Access the embeddings using get_vecattr
    embeddings = {word: model[word] for word in vocab}
    return vocab, embeddings

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