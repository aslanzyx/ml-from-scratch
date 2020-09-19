import numpy as np
from function import function


class product_function(function):

    def calculate(self, x: np.ndarray):
        return np.product(x)

    def gradient(self, x: np.ndarray):
        product = self.calculate(x)
        return np.array([product/x_i for x_i in x])
