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