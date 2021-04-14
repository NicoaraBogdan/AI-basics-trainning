import numpy as np

class LogisticRegresion:

    def __init__(self, lr = 0.00001, nr_iter = 1000):
        self.lr = lr
        self.nr_iter = nr_iter
        self.weight = None
        self.bais = None

    def fit(self, x, y):
        n_samples, n_features = x.shape
        self.weight = np.zeros(n_features)
        self.bais = 0

        for _ in range(self.nr_iter):
            #sigmoid function
            e_pow = np.dot(x, self.weight) + self.bais
            y_pred = 1 / (1 + np.exp(-e_pow))

            dw = (1 / n_samples) * np.dot(x.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            self.weight -= self.lr * dw
            self.bais -= self.lr * db

    def predict(self, x):
        e_pow = np.dot(x, self.weight) + self.bais
        y_pred = 1 / (1 + np.exp(-e_pow))
        y_pred_cls = [1 if prob > 0.5 else 0 for prob in y_pred]
        return y_pred_cls


