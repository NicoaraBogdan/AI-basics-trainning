import numpy as np
from collections import Counter


def euclidean_dist(x, y):
    return np.sqrt(np.sum(x - y)**2)



class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, Y):
        self.X_train = X
        self.Y_train = Y
    
    def predict(self, X):
        predicted_labels = [self._predict(x) for x in X]
        return np.array(predicted_labels)
    
    def _predict(self, X):
        #compute distances
        distances = [euclidean_dist(X, x_train) for x_train in self.X_train]

        # get k nearest samples, labels
        k_ind = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.Y_train[i] for i in k_ind]

        # majority vote, most common class label
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]