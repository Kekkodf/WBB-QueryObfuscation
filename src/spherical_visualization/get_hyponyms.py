import numpy as np
import pandas as pd

import nltk
from nltk.corpus import wordnet as wn


def get_hyponyms(word):
    """Returns a list of hyponyms for a word"""
    hyponyms = []
    for syn in wn.synsets(word):
        for l in syn.hyponyms():
            #remove all after dots
            l = str(l.name()).split('.')[0]
            hyponyms.append(l)

    return list(set(hyponyms))

def get_hyponyms_embedding(embeddings_dict, hyponyms, word):
    embeddings_hyp = {hyp : embeddings_dict[hyp] for hyp in hyponyms if hyp in embeddings_dict.keys() and hyp!=word}
    return embeddings_hyp