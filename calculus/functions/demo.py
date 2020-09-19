import numpy as np

a = np.array([1, 2, 3, 4, 5, 6])
a = np.reshape(a, (2, 3))

print(a)
print(a.T)
print(a*a)
