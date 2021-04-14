import numpy as np

class LinReg:
    #
    # y = w*x + b
    #
    def __init__(self, learn_rate=0.001, n_iter = 1000):
        print("learn_rate:   " + str(learn_rate))
        print("Nr_itre: " + str(n_iter))
        self.learn_rate = learn_rate
        self.n_iter = n_iter
        self.weight = None
        self.bias = None

    def fit(self, X, Y):
        n_samples, n_features = X.shape
        self.weight = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iter):
            y_predicted = np.dot(X, self.weight) + self.bias
            
            dw = (1/n_samples) * np.dot(X.T, (y_predicted - Y))
            db = (1/n_samples) * np.sum(y_predicted - Y)

            self.weight -= self.learn_rate * dw
            self.bias -= self.learn_rate * db

    def predict(self, X):
        y_predicted = np.dot(X, self.weight) + self.bias
        return y_predicted
