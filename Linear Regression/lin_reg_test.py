import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from lin_reg import LinReg

X, Y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1234)

# fig = plt.figure(figsize=(8,6))
# plt.scatter(X[:, 0], Y, color = "b", marker = "o", s = 30)
# plt.show()


lin_reg = LinReg(0.01, n_iter= 1000)
lin_reg.fit(X_train, Y_train)
predicted = lin_reg.predict(X_test)

def mse(Y_true, predict):
    return np.mean((Y_true - predict)**2)
    
mse_value = mse(Y_test, predicted)
print(mse_value)

y_pred_line = lin_reg.predict(X)
cmap = plt.get_cmap()
fig = plt.figure(figsize=(8,6))
plt.scatter(X_train, Y_train, color='red', s=10)
plt.scatter(X_test, Y_test, color='yellow', s=10)
plt.plot(X, y_pred_line, color='black', linewidth = 2, label = 'prediction')
plt.show()