import numpy as np
from particle import *

def pso_fcm(c, P=60, w=1, c1=2, c2=2, maxit=100):
    swarm = list()

    for _ in range(P):
        particle = Particle(c1, c2, w)
        swarm.append(particle)

class FCM:
    def __init__(self, n_clusters=10, max_iter=150, m=2, error=1e-5, random_state=42):
        assert m > 1
        self.u, self.centers = None, None
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.m = m
        self.error = error        

    def fit(self, X, centers):
        self.n_samples = X.shape[0]        
        self.u = np.random.uniform(0, 1, size=(self.n_clusters, self.n_samples))        
        print(self.u.shape)
        for iteration in range(self.max_iter):
            u_old = self.u.copy()
            self.centers = centers # determined by PSO particles
            
            self.u = self.__predict(X)
            # print(self.u)

            # Stopping rule
            if np.linalg.norm(self.u - u_old) < self.error:
                break  

        return self.u
    
    @staticmethod
    def _dist(y, z):
        """ Compute the euclidean distance two vectors """        
        return np.linalg.norm(y - z)

    @staticmethod
    def find_term(Z, Y, i, j, m, c):        
        denominator = 0.0
        for r in range(c):
            term = (FCM._dist(Y[j], Z[i]) / FCM._dist(Y[j], Z[r])) ** float(2/(m-1))
            denominator += term        
        return denominator

    def __predict(self, X):
        u = np.zeros(shape=(self.n_clusters, self.n_samples))
        print(self.n_samples * self.n_clusters)
        for i in range(self.n_clusters):
            for j in range(self.n_samples):
                denominator_ = FCM.find_term(self.centers, X, i, j, self.m, self.n_clusters)
                u[i,j] = 1/denominator_
        return u

    def predict(self, X):
        X = np.expand_dims(X, axis=0) if len(X.shape) == 1 else X
        return self.__predict(X).argmax(axis=-1)
    
    # partition coefficient (Equation 12a of https://doi.org/10.1016/0098-3004(84)90020-7)
    @property
    def partition_coefficient(self):
        if hasattr(self, 'u'):
            return np.sum(self.u ** 2) / self.n_samples
        else:
            raise ReferenceError("You need to train the model first. You can use `.fit()` method to this.")

if __name__ == '__main__':
    pso_fcm(3)    