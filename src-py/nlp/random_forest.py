import numpy as np
import utils

class DecisionStumpErrorRate:

    def __init__(self):
        pass

    def fit(self, X, y):
        """ YOUR CODE HERE FOR Q2.1 """
        N, D = X.shape

        # Get an array with the number of 0's, number of 1's, etc.
        count = np.bincount(y)

        # Get the index of the largest value in count.
        # Thus, y_mode is the mode (most popular value) of y
        y_mode = np.argmax(count)

        self.splitSat = y_mode
        self.splitNot = None
        self.splitVariable = None
        self.splitValue = None

        # If all the labels are the same, no need to split further
        if np.unique(y).size <= 1:
            return

        minError = np.sum(y != y_mode)

        # Loop over features looking for the best split
        for d in range(D):
            for n in range(N):
                # Choose value to equate to
                value = X[n, d]

                # Find most likely class for each split
                y_sat = utils.mode(y[X[:,d] > value])
                y_not = utils.mode(y[X[:,d] <= value])

                # Make predictions
                y_pred = y_sat * np.ones(N)
                y_pred[X[:, d] <= value] = y_not

                # Compute error
                errors = np.sum(y_pred != y)

                # Compare to minimum error so far
                if errors < minError:
                    # This is the lowest error, store this value
                    minError = errors
                    self.splitVariable = d
                    self.splitValue = value
                    self.splitSat = y_sat
                    self.splitNot = y_not

    def predict(self, X):
        """ YOUR CODE HERE FOR Q2.1 """
        splitVariable = self.splitVariable
        splitValue = self.splitValue
        splitSat = self.splitSat
        splitNot = self.splitNot

        M, D = X.shape

        if splitVariable is None:
            return splitSat * np.ones(M)

        yhat = np.zeros(M)

        for m in range(M):
            if X[m, splitVariable] > splitValue:
                yhat[m] = splitSat
            else:
                yhat[m] = splitNot

        return yhat


class DecisionStumpGiniIndex(DecisionStumpErrorRate):


    def fit(self, X, y, split_features=None):
        pass

        "Insert your codes here "



    """
    A helper function that computes the Gini_impurity of the
    discrete distribution p
    """
    def Gini_impurity(p):
        pass

        "Insert your code here "

class DecisionTree:

    def __init__(self, max_depth, stump_class=DecisionStumpErrorRate):
        self.max_depth = max_depth
        self.stump_class = stump_class


    def fit(self, X, y):
        # Fits a decision tree using greedy recursive splitting
        N, D = X.shape

        # Learn a decision stump
        splitModel = self.stump_class()
        splitModel.fit(X, y)

        if self.max_depth <= 1 or splitModel.splitVariable is None:
            # If we have reached the maximum depth or the decision stump does
            # nothing, use the decision stump

            self.splitModel = splitModel
            self.subModel1 = None
            self.subModel0 = None
            return

        # Fit a decision tree to each split, decreasing maximum depth by 1
        j = splitModel.splitVariable
        value = splitModel.splitValue

        # Find indices of examples in each split
        splitIndex1 = X[:,j] > value
        splitIndex0 = X[:,j] <= value

        # Fit decision tree to each split
        self.splitModel = splitModel
        self.subModel1 = DecisionTree(self.max_depth-1, stump_class=self.stump_class)
        self.subModel1.fit(X[splitIndex1], y[splitIndex1])
        self.subModel0 = DecisionTree(self.max_depth-1, stump_class=self.stump_class)
        self.subModel0.fit(X[splitIndex0], y[splitIndex0])


    def predict(self, X):
        M, D = X.shape
        y = np.zeros(M)

        # GET VALUES FROM MODEL
        splitVariable = self.splitModel.splitVariable
        splitValue = self.splitModel.splitValue
        splitSat = self.splitModel.splitSat

        if splitVariable is None:
            # If no further splitting, return the majority label
            y = splitSat * np.ones(M)

        # the case with depth=1, just a single stump.
        elif self.subModel1 is None:
            return self.splitModel.predict(X)

        else:
            # Recurse on both sub-models
            j = splitVariable
            value = splitValue

            splitIndex1 = X[:,j] > value
            splitIndex0 = X[:,j] <= value

            y[splitIndex1] = self.subModel1.predict(X[splitIndex1])
            y[splitIndex0] = self.subModel0.predict(X[splitIndex0])

        return y
        

class RandomStumpGiniIndex(DecisionStumpGiniIndex):

    def fit(self, X, y, thresholds=None):

        # Randomly select k features.
        # This can be done by randomly permuting
        # the feature indices and taking the first k
        D = X.shape[1]
        k = int(np.floor(np.sqrt(D)))

        chosen_features = np.random.choice(D, k, replace=False)

        DecisionStumpGiniIndex.fit(self, X, y, split_features=chosen_features)




class RandomTree(DecisionTree):

    def __init__(self, max_depth):
        DecisionTree.__init__(self, max_depth=max_depth, stump_class=RandomStumpGiniIndex)

    def fit(self, X, y):
        N = X.shape[0]
        boostrap_inds = np.random.choice(N, N, replace=True)
        bootstrap_X = X[boostrap_inds]
        bootstrap_y = y[boostrap_inds]

        DecisionTree.fit(self, bootstrap_X, bootstrap_y)





class RandomForest():

    def __init__(self, max_depth, num_trees):
        self.max_depth = max_depth
        self.num_trees = num_trees
        self.thresholds = None

    def fit(self, X, y):
        N, D = X.shape
        trees = [None] * self.num_trees
        for i in range(self.num_trees):
            model = RandomTree(max_depth=self.max_depth)
            model.fit(X, y, thresholds=self.thresholds)
            trees[i] = model
        self.trees = trees

    def predict(self, Xtest):
        T, D = Xtest.shape
        predictions = np.zeros([T, self.num_trees])
        for j in range(self.num_trees):
            predictions[:, j] = self.trees[j].predict(Xtest)
        predictions_mode = np.zeros(T)
        for i in range(T):
            predictions_mode[i] = utils.mode(predictions[i,:])
        return predictions_mode
