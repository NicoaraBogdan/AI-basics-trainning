import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

data = datasets.load_breast_cancer()

x, y = data.data, data.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1234)

from log_reg import LogisticRegresion

def accuracy(y_test, y_pred):
    accuracy = np.sum(y_test == y_pred) / len(y_test)
    return accuracy

log_reg = LogisticRegresion()
log_reg.fit(x_train, y_train)
y_predicted = log_reg.predict(x_test)

print("Log Reg accuracy: ", accuracy(y_test, y_predicted))