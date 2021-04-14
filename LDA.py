import numpy as np

class LDA:
    def __init__(self):
        self.vals = None
        self.vecs = None
    
    def transform(self, X, y):
        self.__fit(X, y)
        
        return np.dot(X, self.vecs.T)
    
    def __fit(self, X, y):
        n_features = X.shape[1]
        class_labels = np.unique(y)
        
        # S_W, S_B
        mean_overall = np.mean(X, axis=0)
        S_W = np.zeros((n_features, n_features))
        S_B = np.zeros((n_features, n_features))
        for c in class_labels:
            X_c = X[y == c]
            mean_c = np.mean(X_c, axis=0)
            S_W += (X_c - mean_c).T.dot(X_c - mean_c)
            
            n_c = X_c.shape[0]
            mean_diff = (mean_c - mean_overall).reshape(n_features, 1)
            S_B += n_c * (mean_diff).dot(mean_diff.T)
            
        A = np.linalg.inv(S_W).dot(S_B)
        vals, vecs = np.linalg.eig(A)
        vecs = vecs.T
        idxs = np.argsort(abs(vals))[::-1]
        self.vals = vals[idxs]
        self.vecs = vecs[idxs]
