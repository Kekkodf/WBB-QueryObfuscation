import numpy as np
import pandas as pd
import distance
import math

def get_safe_box(word, word_embeddings, model, feature, a=math.pi/4, d=4):
    if feature == 'distance':
        safe_box_distances = distance.get_most_euclidean_distant(word, d, word_embeddings, model)
        return safe_box_distances
    elif feature == 'angle':
        safe_box_angles = distance.get_most_angular_distant(word, a, word_embeddings, model)
        return safe_box_angles
    else:
        raise ValueError('Feature not supported')
    
def get_boundary_word(safe_box, feature):
    if feature == 'angle':
        #get first tuple in safe_box
        boundary_word = safe_box[-1][0]
        return boundary_word
    elif feature == 'distance':
        #get last tuple in safe_box
        boundary_word = safe_box[-1][0]
        return boundary_word
    
def get_obfuscated_word(boundary_word, word_embeddings, model, feature, topn=10):
    if feature == 'angle':
        obfuscated_word = distance.get_most_angular_distant(boundary_word, math.radians(180), word_embeddings, model)
        return obfuscated_word[:topn]
    elif feature == 'distance':
        obfuscated_words = distance.get_most_euclidean_distant(boundary_word, 20, word_embeddings, model)
        return obfuscated_words[:topn]
