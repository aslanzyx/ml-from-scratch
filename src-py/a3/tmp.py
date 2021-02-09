import numpy as np

X = np.array([1, 2, 3, 4]).reshape([2, 2])
print(X)
print(sum(X))
# X = np.append(X, X**2, axis=1)
X = X != 1
print(X)
print(len(X))
