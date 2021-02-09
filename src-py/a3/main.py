
# basics
import argparse
import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# sklearn imports
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

# our code
import linear_model
import utils

url_amazon = "https://www.amazon.com/dp/%s"


def load_dataset(filename):
    with open(os.path.join('..', 'data', filename), 'rb') as f:
        return pickle.load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-q', '--question', required=True)
    io_args = parser.parse_args()
    question = io_args.question

    if question == "1":

        filename = "ratings_Patio_Lawn_and_Garden.csv"
        with open(os.path.join("..", "data", filename), "rb") as f:
            ratings = pd.read_csv(
                f, names=("user", "item", "rating", "timestamp"))

        print("Number of ratings:", len(ratings))
        print("The average rating:", np.mean(ratings["rating"]))

        n = len(set(ratings["user"]))
        d = len(set(ratings["item"]))
        print("Number of users:", n)
        print("Number of items:", d)
        print("Fraction nonzero:", len(ratings)/(n*d))

        X, user_mapper, item_mapper, user_inverse_mapper, item_inverse_mapper, user_ind, item_ind = utils.create_user_item_matrix(
            ratings)
        print(type(X))
        print("Dimensions of X:", X.shape)

    elif question == "1.1":
        filename = "ratings_Patio_Lawn_and_Garden.csv"
        with open(os.path.join("..", "data", filename), "rb") as f:
            ratings = pd.read_csv(
                f, names=("user", "item", "rating", "timestamp"))
        X, user_mapper, item_mapper, user_inverse_mapper, item_inverse_mapper, user_ind, item_ind = utils.create_user_item_matrix(
            ratings)
        X_binary = X != 0

        # YOUR CODE HERE FOR Q1.1.1
        print("complete")
        # s = sum(X)
        # m = np.argmax(s)
        # print(s[m])
        # print(item_inverse_mapper[10959])

        # YOUR CODE HERE FOR Q1.1.2
        # print(X_binary[0,2])

        # s = np.sum(X_binary)
        rates = []
        for user in X:
            rates.append(user[user != 0].shape[1])
        # print(rates)
        # m = np.argmax(rates)

        # print(m)
        # print(rates[m])
        # print(user_inverse_mapper[m])
        # print(item_inverse_mapper[10959])

        # rates = []
        # for user in X_binary:
        #     rate = 0
        #     for item in user:
        #         print(item)
        #         if (item):
        #             rate += 1

        #     rates.append(rate)
        # m = np.argmax(rates)
        # print(m)
        # print(rates[m])
        # print(user_inverse_mapper[m])

        # YOUR CODE HERE FOR Q1.1.3
        print(len(rates))
        plt.hist(np.array(rates))
        plt.title(r'number of rates per user')
        plt.show()

    elif question == "1.2":
        filename = "ratings_Patio_Lawn_and_Garden.csv"
        with open(os.path.join("..", "data", filename), "rb") as f:
            ratings = pd.read_csv(
                f, names=("user", "item", "rating", "timestamp"))
        X, user_mapper, item_mapper, user_inverse_mapper, item_inverse_mapper, user_ind, item_ind = utils.create_user_item_matrix(
            ratings)
        X_binary = X != 0

        grill_brush = "B00CFM0P7Y"
        grill_brush_ind = item_mapper[grill_brush]
        grill_brush_vec = X[:, grill_brush_ind]

        print(url_amazon % grill_brush)

        # YOUR CODE HERE FOR Q1.2
        # model = NearestNeighbors(n_neighbors=6)
        # model = NearestNeighbors(n_neighbors=6, metric = 'cosine')
        model = NearestNeighbors(n_neighbors=6)
        model.fit(normalize(X.T))
        result = model.kneighbors(normalize(grill_brush_vec.T))
        neighbors = result[1][0]
        for i in neighbors:
            print(item_inverse_mapper[i])
            print(np.sum(X.T[i]))

        # YOUR CODE HERE FOR Q1.3

    elif question == "3":
        data = load_dataset("outliersData.pkl")
        X = data['X']
        y = data['y']

        # Fit least-squares estimator
        model = linear_model.LeastSquares()
        model.fit(X, y)
        print(model.w)

        utils.test_and_plot(model, X, y, title="Least Squares",
                            filename="least_squares_outliers.pdf")

    elif question == "3.1":
        data = load_dataset("outliersData.pkl")
        X = data['X']
        y = data['y']

        # YOUR CODE HERE
        model = linear_model.WeightedLeastSquares()
        model.fit(X[:400], y[:400], 1)
        model.fit(X[400:], y[400:], 0.1)
        print(model.w)

        utils.test_and_plot(model, X, y, title="Least Squares",
                            filename="least_squares_outliers_weighted.pdf")

    elif question == "3.3":
        # loads the data in the form of dictionary
        data = load_dataset("outliersData.pkl")
        X = data['X']
        y = data['y']

        # Fit least-squares estimator
        model = linear_model.LinearModelGradient()
        model.fit(X, y)
        print(model.w)

        utils.test_and_plot(
            model, X, y, title="Robust (L1) Linear Regression", filename="least_squares_robust.pdf")

    elif question == "4":
        data = load_dataset("basisData.pkl")
        X = data['X']
        y = data['y']
        Xtest = data['Xtest']
        ytest = data['ytest']

        # Fit least-squares model
        model = linear_model.LeastSquares()
        model.fit(X, y)

        utils.test_and_plot(model, X, y, Xtest, ytest,
                            title="Least Squares, no bias", filename="least_squares_no_bias.pdf")

    elif question == "4.1":
        data = load_dataset("basisData.pkl")
        X = data['X']
        y = data['y']
        Xtest = data['Xtest']
        ytest = data['ytest']

        # YOUR CODE HERE
        model = linear_model.LeastSquaresBias()
        model.fit(X, y)

        utils.test_and_plot(model, X, y, Xtest, ytest,
                            title="Least Squares, with bias", filename="least_squares_bias.pdf")

    elif question == "4.2":
        data = load_dataset("basisData.pkl")
        X = data['X']
        y = data['y']
        Xtest = data['Xtest']
        ytest = data['ytest']

        tr_errs = []
        te_errs = []
        for p in range(11):
            print("p=%d" % p)
            # YOUR CODE HERE
            model = linear_model.LeastSquaresPoly(p)
            model.fit(X, y)

            # This method is modified that it returns the training and testing error
            tr_err, te_err = utils.test_and_plot(model, X, y, Xtest, ytest,
                                                 title="Least Squares, in power {}".format(p), filename="least_squares_poly.pdf")

            tr_errs.append(tr_err)
            te_errs.append(te_err)
        plt.show()
        plt.close()

        plt.plot([i for i in range(11)], tr_errs)
        plt.plot([i for i in range(11)], te_errs)
        plt.title(r'training error and testing error')
        plt.show()

    else:
        print("Unknown question: %s" % question)
