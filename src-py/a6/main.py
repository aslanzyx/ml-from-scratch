import os
import pickle
import gzip
import argparse
import time
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import LabelBinarizer

from neural_net import NeuralNet
from manifold import MDS, ISOMAP
import utils


def load_dataset(filename):
    with open(os.path.join('..', 'data', filename), 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-q', '--question', required=True)

    io_args = parser.parse_args()
    question = io_args.question

    if question == "1":
        dataset = load_dataset('animals.pkl')
        X = dataset['X'].astype(float)
        animals = dataset['animals']
        n, d = X.shape
        print("n =", n)
        print("d =", d)

        # f1, f2 = np.random.choice(d, size=2, replace=False)
        print("var before =", np.sum(np.var(X, axis=0)))
        model = PCA(n_components=4)
        model.fit(X)
        z = model.transform(X)
        print("var after =", np.sum(np.var(z, axis=0)))
        # print(MDS(2)._fun_obj_z(z.flatten(), np.sqrt(utils.euclidean_dist_squared(X, X)))[0])
        f1, f2 = 0, 1

        plt.figure()
        plt.scatter(z[:, f1], z[:, f2])
        plt.xlabel("$z_{%d}$" % f1)
        plt.ylabel("$z_{%d}$" % f2)
        for i in range(n):
            plt.annotate(animals[i], (z[i, f1], z[i, f2]))

        # utils.savefig('two_random_features.png')
        utils.savefig('PCA_nPCs.png')

    elif question == '1.3':

        dataset = load_dataset('animals.pkl')
        X = dataset['X'].astype(float)
        animals = dataset['animals']
        n, d = X.shape

        model = MDS(n_components=2)
        Z = model.compress(X)

        fig, ax = plt.subplots()
        ax.scatter(Z[:, 0], Z[:, 1])
        plt.ylabel('z2')
        plt.xlabel('z1')
        plt.title('MDS')
        for i in range(n):
            ax.annotate(animals[i], (Z[i, 0], Z[i, 1]))
        utils.savefig('MDS_animals.png')

    elif question == '1.4':
        dataset = load_dataset('animals.pkl')
        X = dataset['X'].astype(float)
        animals = dataset['animals']
        n, d = X.shape

        for n_neighbours in [2, 3]:
            model = ISOMAP(n_components=2, n_neighbours=n_neighbours)
            Z = model.compress(X)

            fig, ax = plt.subplots()
            ax.scatter(Z[:, 0], Z[:, 1])
            plt.ylabel('z2')
            plt.xlabel('z1')
            plt.title('ISOMAP with NN=%d' % n_neighbours)
            for i in range(n):
                ax.annotate(animals[i], (Z[i, 0], Z[i, 1]))
            utils.savefig('ISOMAP%d_animals.png' % n_neighbours)

    elif question == '1.5':
        dataset = load_dataset('animals.pkl')
        X = dataset['X'].astype(float)
        animals = dataset['animals']
        n, d = X.shape

        model = TSNE(n_components=2)
        Z = model.fit_transform(X)

        fig, ax = plt.subplots()
        ax.scatter(Z[:, 0], Z[:, 1])
        plt.ylabel('z2')
        plt.xlabel('z1')
        plt.title('TSNE')
        for i in range(n):
            ax.annotate(animals[i], (Z[i, 0], Z[i, 1]))
        utils.savefig('TSNE.png')

    elif question == "2":

        with gzip.open(os.path.join('..', 'data', 'mnist.pkl.gz'), 'rb') as f:
            train_set, valid_set, test_set = pickle.load(f, encoding="latin1")
        X, y = train_set
        Xtest, ytest = test_set

        binarizer = LabelBinarizer()
        Y = binarizer.fit_transform(y)

        hidden_layer_sizes = [50]
        model = NeuralNet(hidden_layer_sizes)

        t = time.time()
        model.fit(X, Y)
        print("Fitting took %d seconds" % (time.time()-t))

        # Comput training error
        yhat = model.predict(X)
        trainError = np.mean(yhat != y)
        print("Training error = ", trainError)

        # Compute test error
        yhat = model.predict(Xtest)
        testError = np.mean(yhat != ytest)
        print("Test error     = ", testError)

    elif question == "2.4":
        with gzip.open(os.path.join('..', 'data', 'mnist.pkl.gz'), 'rb') as f:
            train_set, valid_set, test_set = pickle.load(f, encoding="latin1")
        X, y = train_set
        Xtest, ytest = test_set

        print("n =", X.shape[0])
        print("d =", X.shape[1])

        print(y)

        model = MLPClassifier(batch_size=500)
        model.fit(X, y)

        # Compute training error
        yhat = model.predict(X)
        trainError = np.mean(yhat != y)
        print("Training error = ", trainError)

        # Compute test error
        yhat = model.predict(Xtest)
        testError = np.mean(yhat != ytest)
        print("Test error     = ", testError)

    else:
        print("Unknown question: %s" % question)
