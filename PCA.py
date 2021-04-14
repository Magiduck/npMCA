import numpy as np

class PCA:
    def __init__(self, normalize=False):
        self.normalize = normalize
        self.vals = None
        self.vecs = None
    
    def transform(self, X):
        # center
        X = self.__center_data(X)
        self.__fit(X)
        
        # project
        return np.dot(X, self.vecs.T)

    def __center_data(self, X):
        X = X - X.mean(axis=0)
        if self.normalize:
            X /= X.std(axis=0)
        return X
    
    def __fit(self, X):
        # covariance
        cov = np.cov(X.T)
        
        # eigenvalues, eigenvectors
        vals, vecs = np.linalg.eig(cov)
        
        # sort eigenvectors
        vecs = vecs.T
        idxs = np.argsort(vals)[::-1]
        self.vals = vals[idxs]
        self.vecs = vecs[idxs]
