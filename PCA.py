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
        return np.dot(X, self.vecs)

    def __center_data(self, X):
        mean = np.mean(X, axis=0)
        X = X - mean
        if self.normalize:
            X /= X.std(axis=0)
        return X

    def __fit(self, X):
        # covariance
        cov = np.cov(X.T)

        # eigenvalues, eigenvectors
        vals, vecs = np.linalg.eig(cov)

        # sort eigenvectors
        idxs = vals.argsort()[::-1]
        self.vals = vals[idxs]
        self.vecs = vecs[:, idxs]
