import numpy as np
from scipy.spatial.distance import euclidean, cosine
import random
import pandas as pd
from tqdm import tqdm
tqdm.pandas()
import time
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

desired_pos_tags = ['NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJS']

def obfuscate(row, model, k, n, epsilon, feature):
    '''
    The method obfuscates a query using the distance based obfuscation technique.
    '''
    new_query = row.progress_apply(compute_query, args=(model, k, n, epsilon, feature))
    return new_query
    
def compute_query(row, model, k, n, epsilon, feature):
        try:
            x_words_embs = np.array([model.get_word_embedding(word[0]) for word in row if word[1] in desired_pos_tags])
            candidates, scores, safe_box = model.get_k_closest_terms(x_words_embs, k, n, epsilon, feature)
            #candidates and scores must be of the same dimensions

            # List to store the selected candidates
            selected_candidates = []
            # Iterate over each row
            for i in range(candidates.shape[0]):
                # Select one candidate from the current row
                selected = random.choices(candidates[i], weights=scores[i], k=1)[0]
                # Append the selected candidate to the list
                selected_candidates.append(selected)

            #print(selected_candidates)
            
            #print('Extracted: ', extracted)
            query_obfuscated = [word[0] if word[1] not in desired_pos_tags else selected_candidates.pop(0) for word in row]
            #print('Query obfuscated: ', query_obfuscated)
            #time.sleep(30)
            res = ' '.join(query_obfuscated)
            #print(res)
            return res
        except KeyError:
            res = ' '.join([word[0] for word in row])
            #print('Key Error: ', res)
            return res
        except IndexError:
            res = ' '.join([word[0] for word in row])
            #print('Index Error: ', res)
            return res
        
    
