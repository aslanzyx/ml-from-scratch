import os
import pickle
import argparse
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import Axes3D

import utils
import logReg
from logReg import logRegL2, kernelLogRegL2
from pca import PCA, AlternativePCA, RobustPCA


def load_dataset(filename):
    with open(os.path.join('..', 'data', filename), 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-q', '--question', required=True)

    io_args = parser.parse_args()
    question = io_args.question

    if question == "1":
        dataset = load_dataset('nonLinearData.pkl')
        X = dataset['X']
        y = dataset['y']

        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, random_state=0)

        # standard logistic regression
        lr = logRegL2(lammy=1)
        lr.fit(Xtrain, ytrain)

        print("Training error %.3f" % np.mean(lr.predict(Xtrain) != ytrain))
        print("Validation error %.3f" % np.mean(lr.predict(Xtest) != ytest))

        utils.plotClassifier(lr, Xtrain, ytrain)
        utils.savefig("logReg.png")

        # kernel logistic regression with a linear kernel
        lr_kernel = kernelLogRegL2(kernel_fun=logReg.kernel_linear, lammy=1)
        lr_kernel.fit(Xtrain, ytrain)

        print("Training error %.3f" %
              np.mean(lr_kernel.predict(Xtrain) != ytrain))
        print("Validation error %.3f" %
              np.mean(lr_kernel.predict(Xtest) != ytest))

        utils.plotClassifier(lr_kernel, Xtrain, ytrain)
        utils.savefig("logRegLinearKernel.png")

    elif question == "1.1":
        dataset = load_dataset('nonLinearData.pkl')
        X = dataset['X']
        y = dataset['y']

        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, random_state=0)

        # YOUR CODE HERE
        pl_kernel = kernelLogRegL2(
            kernel_fun=logReg.kernel_poly, lammy=.01, p=2)
        pl_kernel.fit(Xtrain, ytrain)

        print("Training error %.3f" %
              np.mean(pl_kernel.predict(Xtrain) != ytrain))
        print("Validation error %.3f" %
              np.mean(pl_kernel.predict(Xtest) != ytest))

        utils.plotClassifier(pl_kernel, Xtrain, ytrain)
        utils.savefig("logRegPloyKernel.png")

        # 0.403
        # 0.340

        rbf_kernel = kernelLogRegL2(
            kernel_fun=logReg.kernel_RBF, lammy=.01, sigma=.5)
        rbf_kernel.fit(Xtrain, ytrain)

        print("Training error %.3f" %
              np.mean(rbf_kernel.predict(Xtrain) != ytrain))
        print("Validation error %.3f" %
              np.mean(rbf_kernel.predict(Xtest) != ytest))

        utils.plotClassifier(rbf_kernel, Xtrain, ytrain)
        print("finished graphing")
        utils.savefig("logRegRBFKernel.png")

        # 0.127
        # 0.090

    elif question == "1.2":
        dataset = load_dataset('nonLinearData.pkl')
        X = dataset['X']
        y = dataset['y']

        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, random_state=0)

        # YOUR CODE HERE
        sigma_pow_range = range(-2, 3)
        lambda_pow_range = range(-4, 1)
        base = 10
        train_errs = np.ndarray([5, 5])
        test_errs = np.ndarray([5, 5])

        min_train_err = np.inf
        min_test_err = np.inf
        min_train_err_model = None
        min_test_err_model = None
        for i in sigma_pow_range:
            for j in lambda_pow_range:
                model = kernelLogRegL2(
                    kernel_fun=logReg.kernel_RBF, lammy=base**j, sigma=base**i
                )
                model.fit(Xtrain, ytrain)
                train_errs[i, j] = np.mean(model.predict(Xtrain) != ytrain)
                test_errs[i, j] = np.mean(model.predict(Xtest) != ytest)
                if (train_errs[i, j] < min_train_err):
                    min_train_err = train_errs[i, j]
                    min_train_err_model = model

                if (test_errs[i, j] < min_test_err):
                    min_test_err = test_errs[i, j]
                    min_test_err_model = model
        utils.plotClassifier(min_train_err_model, Xtrain, ytrain)
        utils.savefig("logRegRBFKernelTest.png")
        utils.plotClassifier(min_test_err_model, Xtest, ytest)
        utils.savefig("logRegRBFKernelTest.png")

        print(min_train_err)
        print(min_test_err)
        print(train_errs)
        print(test_errs)

    elif question == '4.1':
        X = load_dataset('highway.pkl')['X'].astype(float)/255
        n, d = X.shape
        print(n, d)
        h, w = 64, 64      # height and width of each image

        k = 5            # number of PCs
        threshold = 0.1  # threshold for being considered "foreground"

        model = AlternativePCA(k=k)
        model.fit(X)
        Z = model.compress(X)
        Xhat_pca = model.expand(Z)

        model = RobustPCA(k=k)
        model.fit(X)
        Z = model.compress(X)
        Xhat_robust = model.expand(Z)

        fig, ax = plt.subplots(2, 3)
        for i in range(10):
            ax[0, 0].set_title('$X$')
            ax[0, 0].imshow(X[i].reshape(h, w).T, cmap='gray')

            ax[0, 1].set_title('$\hat{X}$ (L2)')
            ax[0, 1].imshow(Xhat_pca[i].reshape(h, w).T, cmap='gray')

            ax[0, 2].set_title('$|x_i-\hat{x_i}|$>threshold (L2)')
            ax[0, 2].imshow((np.abs(X[i] - Xhat_pca[i]) <
                             threshold).reshape(h, w).T, cmap='gray', vmin=0, vmax=1)

            ax[1, 0].set_title('$X$')
            ax[1, 0].imshow(X[i].reshape(h, w).T, cmap='gray')

            ax[1, 1].set_title('$\hat{X}$ (L1)')
            ax[1, 1].imshow(Xhat_robust[i].reshape(h, w).T, cmap='gray')

            ax[1, 2].set_title('$|x_i-\hat{x_i}|$>threshold (L1)')
            ax[1, 2].imshow((np.abs(X[i] - Xhat_robust[i]) <
                             threshold).reshape(h, w).T, cmap='gray', vmin=0, vmax=1)

            utils.savefig('highway_{:03d}.jpg'.format(i))

    else:
        print("Unknown question: %s" % question)
