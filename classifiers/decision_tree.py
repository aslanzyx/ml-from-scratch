import numpy as np
from util import score, most


class DecisionStamp:

    # stamp properties
    def __init__(self):
        self.yes_label = None
        self.no_label = None
        self.threshold = None
        self.feature = None
        pass

    def fit(self, train_data: np.ndarray, train_label: np.ndarray):
        # get dimensions from the train data
        n, d = train_data.shape
        # iterate for each train data at each possible feature
        score = 0
        for feature in train_data:
            possible_thredshold = feature  # more to be implemented
            for threshold in possible_thredshold:
                # apply hypothetical properties
                yes_set = train_label[feature > threshold]
                no_set = train_label[feature <= threshold]

                # get labels
                yes_label = most(yes_set)
                no_label = most(no_set)

                # make predictions and calculate score
                yes_score = score(yes_set, yes_label)
                no_score = score(no_set, yes_label)

                cur_score = yes_score + no_score
                if cur_score > score:
                    self.yes_label = yes_label
                    self.no_label = no_label
                    self.threshold = threshold
                    self.feature = feature
        return

    def predict(self, test_data: np.ndarray):
        if test_data[self.feature] > self.threshold:
            return self.yes_label
        else:
            return self.no_label

    def test_error(self, test_data: np.ndarray, test_label: np.ndarray):
        return test_label[self.predict(test_data) == test_label].size
