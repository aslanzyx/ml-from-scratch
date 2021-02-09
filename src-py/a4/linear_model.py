import numpy as np
from numpy.linalg import solve
import findMin
from scipy.optimize import approx_fprime
import utils
import os
import gzip
import pickle


class logReg:
    # Logistic Regression
    def __init__(self, verbose=0, maxEvals=100):
        self.verbose = verbose
        self.maxEvals = maxEvals
        self.bias = True

    def funObj(self, w, X, y):
        yXw = y * X.dot(w)

        # Calculate the function value
        f = np.sum(np.log(1. + np.exp(-yXw)))

        # Calculate the gradient value
        res = - y / (1. + np.exp(yXw))
        g = X.T.dot(res)

        return f, g

    def fit(self, X, y):
        n, d = X.shape

        # Initial guess
        self.w = np.zeros(d)
        utils.check_gradient(self, X, y)
        (self.w, f) = findMin.findMin(self.funObj, self.w,
                                      self.maxEvals, X, y, verbose=self.verbose)

    def predict(self, X):
        return np.sign(X@self.w)


class logRegL0(logReg):
    # L0 Regularized Logistic Regression
    def __init__(self, L0_lambda=1.0, verbose=2, maxEvals=400):
        self.verbose = verbose
        self.L0_lambda = L0_lambda
        self.maxEvals = maxEvals

    def fit(self, X, y):
        n, d = X.shape

        def minimize(ind): return findMin.findMin(self.funObj,
                                                  np.zeros(len(ind)),
                                                  self.maxEvals,
                                                  X[:, ind], y, verbose=0)
        selected = set()
        selected.add(0)
        minLoss = np.inf
        oldLoss = 0
        bestFeature = -1

        while minLoss != oldLoss:
            oldLoss = minLoss
            print("Epoch %d " % len(selected))
            print("Selected feature: %d" % (bestFeature))
            print("Min Loss: %.3f\n" % minLoss)

            for i in range(d):
                if i in selected:
                    continue

                # tentatively add feature "i" to the seected set
                selected_new = selected | {i}

                # TODO for Q2.3: Fit the model with 'i' added to the features,
                w, curLoss = minimize(list(selected_new))
                curLoss += self.L0_lambda*len(selected_new)
                if curLoss < minLoss:
                    minLoss = curLoss
                    bestFeature = i

            selected.add(bestFeature)

        self.w = np.zeros(d)
        self.w[list(selected)], _ = minimize(list(selected))


class leastSquaresClassifier:
    def fit(self, X, y):
        n, d = X.shape
        self.n_classes = np.unique(y).size

        # Initial guess
        self.W = np.zeros((self.n_classes, d))

        for i in range(self.n_classes):
            ytmp = y.copy().astype(float)
            ytmp[y == i] = 1
            ytmp[y != i] = -1

            # solve the normal equations
            # with a bit of regularization for numerical reasons
            self.W[i] = np.linalg.solve(X.T@X+0.0001*np.eye(d), X.T@ytmp)

    def predict(self, X):
        return np.argmax(X@self.W.T, axis=1)


class logLinearClassifier(leastSquaresClassifier):
    def __init__(self, maxEvals=100, verbose=0):
        self.verbose = verbose
        self.maxEvals = maxEvals

    def fit(self, X, y):
        n, d = X.shape
        self.n_classes = np.unique(y).size

        # Initial guess
        self.W = np.zeros((self.n_classes, d))

        for i in range(self.n_classes):
            ytmp = y.copy().astype(float)
            ytmp[y == i] = 1
            ytmp[y != i] = -1

            # solve the normal equations
            # with a bit of regularization for numerical reasons
            # self.W[i] = np.linalg.solve(X.T@X+0.0001*np.eye(d), X.T@ytmp)
            model = logReg(self.verbose, self.maxEvals)
            model.fit(X, ytmp)
            self.W[i] = model.w


class logRegL2(logReg):
    # L2 Regularized Logistic Regression
    def __init__(self, lammy=1.0, verbose=0, maxEvals=100):
        self.verbose = verbose
        self.L2_lambda = lammy
        self.maxEvals = maxEvals

    def funObj(self, w, X, y):
        yXw = y * X.dot(w)

        # Calculate the function value
        f = np.sum(np.log(1. + np.exp(-yXw))) + self.L2_lambda/2 * np.sum(w**2)

        # Calculate the gradient value
        res = - y / (1. + np.exp(yXw))
        g = X.T.dot(res) + self.L2_lambda*w

        return f, g


class logRegL1(logReg):
    # L1 Regularized Logistic Regression
    def __init__(self, L1_lambda=1.0, verbose=0, maxEvals=100):
        self.verbose = verbose
        self.maxEvals = maxEvals
        self.L1_lambda = L1_lambda

    def fit(self, X, y):
        n, d = X.shape

        # Initial guess
        self.w = np.zeros(d)
        utils.check_gradient(self, X, y)
        (self.w, f) = findMin.findMinL1(self.funObj, self.w, self.L1_lambda,
                                        self.maxEvals, X, y, verbose=self.verbose)


class softmaxClassifier:
    def __init__(self, verbose=0, maxEvals=100):
        self.verbose = verbose
        self.maxEvals = maxEvals

    def funObj(self, w, X, y):
        n, d = X.shape
        w = w.reshape([self.n_classes, d])
        expLoss = np.exp(X@w.T)
        sumOfExp = np.sum(expLoss, axis=1)

        # Calculate the function value
        f = 0
        for i in range(n):
            f += np.log(sumOfExp[i]) - w[y[i]]@X[i].T

        # Calculate the gradient value
        g = np.zeros([self.n_classes, d])
        for c in range(self.n_classes):
            for j in range(d):
                for i in range(n):
                    g[c, j] += X[i, j]*(expLoss[i, c]/sumOfExp[i])
                    if y[i] == c:
                        g[c, j] -= X[i, j]

        return f, g.reshape(self.n_classes*d)

    def fit(self, X, y):
        n, d = X.shape
        self.n_classes = np.unique(y).size

        # Initial guess
        self.w = np.zeros(self.n_classes*d)
        utils.check_gradient(self, X, y)
        (self.w, f) = findMin.findMin(self.funObj, self.w,
                                      self.maxEvals, X, y, verbose=self.verbose)
        print("training finished")

    def predict(self, X):
        t, d = X.shape
        return np.argmax(X@self.w.reshape([self.n_classes, d]).T, axis=1)


if __name__ == '__main__':
    with gzip.open(os.path.join('..', 'data', 'mnist.pkl.gz'), 'rb') as f:
        train_set, valid_set, test_set = pickle.load(f, encoding="latin1")
    X, y = train_set
    Xtest, ytest = test_set

    X = X[:1000]
    y = y[:1000]

    print("n =", X.shape[0])
    print("d =", X.shape[1])

    model = softmaxClassifier()
    model.fit(X, y)

    # Compute training error
    yhat = model.predict(X)
    trainError = np.mean(yhat != y)
    print("Training error = ", trainError)
