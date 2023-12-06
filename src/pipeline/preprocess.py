import gensim.downloader as api
import nltk
import numpy as np
from tqdm import tqdm
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')
path = './data/embeddings/'

def model(dimension):
    if dimension == 50:
        return api.load('glove-wiki-gigaword-50'), api.load('glove-wiki-gigaword-50').index_to_key
    elif dimension == 100:
        return api.load('glove-wiki-gigaword-100'), api.load('glove-wiki-gigaword-100').index_to_key
    elif dimension == 200:
        return api.load('glove-wiki-gigaword-200'), api.load('glove-wiki-gigaword-200').index_to_key
    elif dimension == 300:
        return api.load('glove-wiki-gigaword-300'), api.load('glove-wiki-gigaword-300').index_to_key
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

def initialize_glove_optimized(optimization = False):
    if optimization == True:
        with open(path + 'glove.6B.300d.txt', 'r', encoding='utf-8') as f:
            glove_normal = f.readlines()
        with open(path + 'glove_counterFitt.txt', 'r', encoding='utf-8') as f:
            glove_optimal = f.readlines()
        glove_normal = [line.strip().split(' ') for line in tqdm(glove_normal)]
        glove_optimal = [line.strip().split(' ') for line in tqdm(glove_optimal)]
        vocab_normal = [word[0] for word in glove_normal]
        vocab_optimal = [word[0] for word in glove_optimal]
        glove_normal = {word[0]: np.array(word[1:], dtype=np.float32) for word in tqdm(glove_normal)}
        glove_optimal = {word[0]: np.array(word[1:], dtype=np.float32) for word in tqdm(glove_optimal)}
        #substitute the optimal embeddings to the normal ones
        for word in tqdm(vocab_optimal):
            if word in vocab_normal:
                glove_normal[word] = glove_optimal[word]
        return glove_normal, vocab_normal
    elif optimization == False:
        with open(path + 'glove.6B.300d.txt', 'r', encoding='utf-8') as f:
            glove = f.readlines()
        glove = [line.strip().split(' ') for line in tqdm(glove)]
        vocab = [word[0] for word in glove]
        glove = {word[0]: np.array(word[1:], dtype=np.float32) for word in tqdm(glove)}
        return glove, vocab
    else:
        raise ValueError('optimization must be True or False')