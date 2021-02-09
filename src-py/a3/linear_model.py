import numpy as np
from numpy.linalg import solve
from findMin import findMin
from scipy.optimize import approx_fprime
import utils

# Ordinary Least Squares


class LeastSquares:
    def fit(self, X, y):
        self.w = solve(X.T@X, X.T@y)

    def predict(self, X):
        return X@self.w

# Least squares where each sample point X has a weight associated with it.


# inherits the predict() function from LeastSquares
class WeightedLeastSquares(LeastSquares):
    def fit(self, X, y, z):
        ''' YOUR CODE HERE '''
        self.w = solve(X.T@(z*X), X.T@(z*y))


class LinearModelGradient(LeastSquares):

    def fit(self, X, y):
        n, d = X.shape

        # Initial guess
        self.w = np.zeros((d))

        # check the gradient
        estimated_gradient = approx_fprime(
            self.w, lambda w: self.funObj(w, X, y)[0], epsilon=1e-6)
        implemented_gradient = self.funObj(self.w, X, y)[1]
        if np.max(np.abs(estimated_gradient - implemented_gradient) > 1e-4):
            print('User and numerical derivatives differ: %s vs. %s' %
                  (estimated_gradient, implemented_gradient))
        else:
            print('User and numerical derivatives agree.')

        self.w, f = findMin(self.funObj, self.w, 100, X, y)

    def funObj(self, w, X, y):
        ''' MODIFY THIS CODE '''
        # Calculate the function value
        # f = 0.5*np.sum((X@w - y)**2)
        # r = X@w-y
        # f = np.sum(np.log(np.exp(r)+np.exp(-r)))

        # Calculate the gradient value
        # g = X.T@(X@w-y)
        # g = X.T@np.tanh(r)

        f = g = 0
        for i in range(X.shape[0]):
            r = X[i]@w-y[i]
            f += np.log(np.exp(r) + np.exp(-r))
            g += X[i]*np.tanh(r)

        return (f, g)


# Least Squares with a bias added
class LeastSquaresBias:

    def fit(self, X, y):
        ''' YOUR CODE HERE '''
        # raise NotImplementedError()
        X_bias = np.append(X, X**0, axis=1)
        self.w = solve(X_bias.T@X_bias, X_bias.T@y)

    def predict(self, X):
        ''' YOUR CODE HERE '''
        # raise NotImplementedError()
        X_bias = np.append(X, X**0, axis=1)
        return X_bias@self.w


# Least Squares with polynomial basis


class LeastSquaresPoly:
    def __init__(self, p):
        self.leastSquares = LeastSquares()
        self.p = p

    def fit(self, X, y):
        ''' YOUR CODE HERE '''
        # raise NotImplementedError()

        # Assuming only 1 feature in X
        self.leastSquares.fit(self.__polyBasis(X), y)

    def predict(self, X):
        ''' YOUR CODE HERE '''
        # raise NotImplementedError()

        return self.leastSquares.predict(self.__polyBasis(X))

    # A private helper function to transform any matrix X into
    # the polynomial basis defined by this class at initialization
    # Returns the matrix Z that is the polynomial basis of X.
    def __polyBasis(self, X):
        ''' YOUR CODE HERE '''
        # raise NotImplementedError()
        X_poly = np.ndarray([X.shape[0], 0])
        for i in range(self.p):
            X_poly = np.append(X_poly, X**i, axis=1)
        return X_poly
