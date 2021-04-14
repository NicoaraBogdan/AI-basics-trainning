import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

cmap = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

iris = datasets.load_iris()
X, y = iris.data, iris.target

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state = 1234)

# print(X_train[0])

# plt.figure()
# plt.scatter(X[:,0], X[:, 2], c=y, cmap=cmap, edgecolor ='k', s=20)
# plt.show()

# from knnTest import KNN
    
# clf = KNN(k=3)
# clf.fit(X_train, Y_train)
# predictions = clf.predict(X_test)

# accuracy = np.sum(predictions == Y_test) / len(Y_test)
# print(accuracy)