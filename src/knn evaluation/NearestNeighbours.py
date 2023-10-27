import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import embeddings as emb

class NearestNeighbours:

    def __init__(self, vector, k):
        self.vector = vector
        self.k = k
        self.scaler = StandardScaler()
        self.embeddings = np.array(list(emb.embeddings_dict.values()))
        self.embeddings = self.scaler.fit_transform(self.embeddings)
        self.nbrs = NearestNeighbors(n_neighbors=self.k, algorithm='ball_tree').fit(self.embeddings)
        self.distances, self.indices = self.nbrs.kneighbors(self.vector.reshape(1, -1))
        self.distances = self.distances.flatten()
        self.indices = self.indices.flatten()

    def get_distances(self):
        return self.distances
    
    def get_indices(self):
        return self.indices