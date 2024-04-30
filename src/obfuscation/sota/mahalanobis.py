import numpy as np
import numpy.random as npr
"""
Xu, Zekun, Abhinav Aggarwal, Oluwaseyi Feyisetan, and Nathanael Teissier. "A Differentially Private Text Perturbation Method Using Regularized
Mahalanobis Metric." In Proceedings of the Second Workshop on Privacy in NLP, pp. 7-17. 2020.
"""
def noise_sampling(self):
    N = npr.multivariate_normal(np.zeros(self.m), np.eye(self.m))
    X = N / np.sqrt(np.sum(N ** 2)) #direction
    X = np.matmul(self.sigma_loc, X)
    X = X / np.sqrt(np.sum(X ** 2))
    Y = npr.gamma(self.m, 1 / self.epsilon) #distance
    Z = X*Y
    #Z = Y * np.matmul(self.sigma_loc, X)
    return Z
