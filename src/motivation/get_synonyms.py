import numpy as np
import pandas as pd

import nltk
from nltk.corpus import wordnet as wn


def get_synonyms(word):
    """Returns a list of synonyms for a word"""
    synonyms = []
    for syn in wn.synsets(word):
        for l in syn.lemmas():
            synonyms.append(l.name())
    return list(set(synonyms))

def get_synonyms_embedding(embeddings_dict, synonyms, word):
    embeddings_syn = {syn : embeddings_dict[syn] for syn in synonyms if syn in embeddings_dict.keys() and syn!=word}
    return embeddings_syn