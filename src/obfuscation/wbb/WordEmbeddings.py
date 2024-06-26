import numpy as np
import random
import nltk
import time
#nltk.download('words')
#nltk.download('stopwords')

words = set(nltk.corpus.words.words())
stopwords = nltk.corpus.stopwords.words('english')


class WordEmbeddings:
    def __init__(self, model, vocab):
        # Filter out words with any non-alphabetic characters from the vocabulary
        self.vocab = [word for word in vocab if word not in stopwords and word.isalpha()]
        self.model = model
        #size of the embeddings
        self._embeddings = {word: model[word] for word in self.vocab}
        self._embsize = len(self._embeddings[list(self._embeddings.keys())[0]])
        self._word2int = {w: i for i, w in enumerate(self._embeddings.keys())}
        self._int2word = {i: w for w, i in self._word2int.items()}
        self._embeddings_matrix = np.zeros((len(self._embeddings), self._embsize))
        for w, i in self._word2int.items():
            self._embeddings_matrix[i] = self._embeddings[w]
    
    def get_embeddings_matrix(self):
        return self._embeddings_matrix
    
    def get_word_embedding(self, word):
        return self._embeddings[word]
    
    def get_k_closest_terms(self, vectors, k, n, epsilon, feature):
        #print('Vector shape: ', vectors.shape)
        
        if feature == 'distance':
            distance = euclidean_distance_matrix(vectors, self._embeddings_matrix)
            found_candidates = distance.argsort(axis=1)[:, k:k+n] #n candidates sorted by distance
            found_safe = distance.argsort(axis=1)[:, :k] #k safe sorted by distance
            #get values of distances candidates
            candidates_distances = np.take_along_axis(distance, found_candidates, axis=1)
        elif feature == 'angle':
            distance = cosine_distance_matrix(vectors, self._embeddings_matrix)
            found_candidates = distance.argsort(axis=1)[:, k:k+n] #n candidates sorted by distance
            found_safe = distance.argsort(axis=1)[:, :k] #k safe sorted by distance
            #get values of distances candidates
            candidates_distances = np.take_along_axis(distance, found_candidates, axis=1)
        elif feature == 'product':
            distance = product_metrix(vectors, self._embeddings_matrix)
            found_candidates = distance.argsort(axis=1)[:, k:k+n] #n candidates sorted by distance
            found_safe = distance.argsort(axis=1)[:, :k] #k safe sorted by distance
            #get values of distances candidates
            candidates_distances = np.take_along_axis(distance, found_candidates, axis=1)
        else:
            raise ValueError('Feature not supported!')
        
        #compute candidates scores based on distances, the lower the distance the higher the score
        mean = candidates_distances.mean()
        std = candidates_distances.std()
        candidates_scores = [(1/(1+np.exp((distance - mean)/(std))))*epsilon/2 for distance in candidates_distances]   
        candidates_scores = candidates_scores/np.sum(candidates_scores)
        
        # Convert the indexes to words
        candidates = self.indexes_to_words(found_candidates)
        #convert candidates in a matrix len(vectors) x n
        candidates = np.array(candidates).reshape(len(vectors), n)

        #create tuples (word, distance)
        safe = self.indexes_to_words(found_safe)
        
        return candidates, candidates_scores, safe
        
    def indexes_to_words(self, indexes):
        return [self._int2word[e] for f in indexes for e in f]

    def candidate_extraction(self, candidates_boxes, distribution):
        probabilities = get_probabilities_of_extraction(candidates_boxes, distribution)
        return list(random.choices(candidates_boxes, weights=probabilities, k=len(candidates_boxes)))

@staticmethod
def euclidean_distance_matrix(x, y):
    x_expanded = x[:, np.newaxis, :]  # Shape: [n, 1, d]
    y_expanded = y[np.newaxis, :, :]  # Shape: [1, m, d]
    return np.sqrt(np.sum((x_expanded - y_expanded) ** 2, axis=2))

@staticmethod
def cosine_distance_matrix(x, y):
    x_expanded = x[:, np.newaxis, :]
    y_expanded = y[np.newaxis, :, :]
    return 1 - np.sum(x_expanded * y_expanded, axis=2) / (np.linalg.norm(x_expanded, axis=2) * np.linalg.norm(y_expanded, axis=2))

@staticmethod
def product_metrix(x, y):
    return cosine_distance_matrix(x, y) * euclidean_distance_matrix(x, y)

@staticmethod
def get_probabilities_of_extraction(candidates_box, distribution):
    if distribution[0] == 'laplace':
            probabilities = np.random.laplace(loc = distribution[1][0], scale = distribution[1][1], size = len(candidates_box))
            probabilities = np.abs(probabilities)
            probabilities = probabilities/np.sum(probabilities)
            #sort probabilities
            probabilities = sorted(probabilities, reverse=True)
            return probabilities
    elif distribution[0] == 'uniform':
        probabilities = np.random.uniform(low = distribution[1][0], high = distribution[1][1], size = len(candidates_box))
        probabilities = sorted(probabilities, reverse=True)
        return probabilities
    elif distribution[0] == 'normal':
        probabilities = np.random.normal(loc = distribution[1][0], scale = distribution[1][1], size = len(candidates_box))
        probabilities = probabilities/np.sum(probabilities)
        #sort probabilities
        probabilities = sorted(probabilities, reverse=True)
        return probabilities
    elif distribution[0] == 'gamma':
        probabilities = np.random.exponential(scale=distribution[1][1], size=len(candidates_box))
        probabilities = probabilities/np.sum(probabilities)
        #sort probabilities
        probabilities = sorted(probabilities, reverse=True)
        return probabilities
    else:
        raise ValueError('Distribution not supported')