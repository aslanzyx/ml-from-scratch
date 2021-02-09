import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def f(a, b):
    print(a.shape)
    print(b.shape)
    return a@b.T


fig = plt.figure()
ax = Axes3D(fig)

x = np.arange(0, 16).reshape([4, 4])
y = np.arange(16, 32).reshape([4, 4])
z = np.arange(32, 48).reshape([4, 4])

z[1, 3] = -99

ax.plot_wireframe(x, y, z)
plt.show()
