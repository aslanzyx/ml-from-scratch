import numpy as np
from math import log, pi


class NaiveBayes:

    def __init__(self):
        pass

    def fit(self, X: np.ndarray, y: np.ndarray):
        # TODO: we need to modify variables:
        # M: a matrix whose entry m_cd is the mean of all data at feature d and of class c
        # S: a matrix whose entry s_cd is the standard deviation of all data at feature d and of class c

        # NOTE: we only have 2 class labels 1 for Trump and 0 for Biden in the context of this question
        # For simplicity I am assuming the class number is static and equals 2

        X_0 = X[y == 0]
        X_1 = X[y == 1]

        self.m = np.array(
            [
                np.mean(X_0, axis=0),
                np.mean(X_1, axis=0)
            ]
        )

        self.s = np.array(
            [
                np.std(X_0, axis=0),
                np.std(X_1, axis=0)
            ]
        )

    def predict(self, X: np.ndarray):
        # TODO: return a vector of predictions
        # NOTE: would return 1 (Trump) if the probabilities matches up
        retval: list = []
        for x_i in X:
            p_subtract = 0
            for d in range(len(x_i)):
                p_subtract += log_gaussian_likelihood(
                    x_i[d], self.m[0, d], self.s[0, d])
                p_subtract -= log_gaussian_likelihood(
                    x_i[d], self.m[1, d], self.s[1, d])
            if p_subtract > 0:
                retval.append(0)
            else:
                retval.append(1)
        return np.array(retval)


def log_gaussian_likelihood(x, m, s):
    return -(.5*((x-m)/s)**2+log(s*(2*pi)**.5))


test_file_raw = open("./data/wordvec_test.csv")
test_data_raw = test_file_raw.read()
test_data_raw = test_data_raw.split("\n")[1:]
test_data_raw = [line.split(",") for line in test_data_raw]
print(len(test_data_raw))
print(len(test_data_raw[0]))
# x = np.array(test_data_raw)[:, :200]
# y = np.array(test_data_raw)[:, 200:]

# print(x[0])

test_file_raw.close()
