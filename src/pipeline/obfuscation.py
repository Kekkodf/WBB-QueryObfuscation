import numpy as np
from scipy.spatial.distance import euclidean, cosine
import random
import pandas as pd
from tqdm import tqdm
tqdm.pandas()
from functools import partial
'''
Obfusctaion phases:
    - Rank computation
    - Safe box extraction
    - Candidate extraction
    - Candidate selection
'''

'''
Parameters:
    - N:= Number of extractions
    - k:= Size of safe_box
    - distribution:= Distribution for candidates selection
'''

desired_pos_tags = ['NN', 'NNS', 'NNP', 'NNPS', 'JJ']

def obfuscate(row, model, k, n, distribution, feature):
    '''
    The method obfuscates a query using the distance based obfuscation technique.
    '''
    new_query = row.progress_apply(compute_query, args=(model, k, n, distribution, feature))
    return new_query
    
def compute_query(row, model, k, n, distribution, feature):
    query_obfuscated = [word[0] for word in row]
    try:
        x_words_embs = np.array([model.get_word_embedding(word[0]) for word in row if word[1] in desired_pos_tags])
        partial_get_candidates = partial(model.get_k_closest_terms, k=k, n=n, feature=feature)
        candidates, safe_box = partial_get_candidates(x_words_embs)
        extracted = model.candidate_extraction(candidates, distribution)
        query_obfuscated = [word[0] if word[1] not in desired_pos_tags else extracted.pop(0) for word in row]
    except:
        pass
    return ' '.join(query_obfuscated)
