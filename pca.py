import numpy as np

class PCA:    
    def fit_compress(self, X, k):
        X = np.transpose(X)
        mu = np.array([x.mean() for x in X])
        variance = (X -  mu[np.newaxis].T)
        Sigma = (1 / len(X)) * np.matmul(variance, variance.T)
        U, S, V = np.linalg.svd(Sigma)

        self.U = U[:, :k]

        X = np.transpose(X)
        return np.matmul(X, self.U)

    def compress(self, X):
        return np.matmul(X, self.U)

    def expand(self, X):
        return np.matmul(X, self.U.T)