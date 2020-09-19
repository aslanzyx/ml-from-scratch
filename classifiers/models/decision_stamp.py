class DecisionStamp:

    _feature = -1
    _threshold = None
    _data = None

    def __init__(self, feature, threshold, data):
        self._feature = feature
        self._threshold = threshold
        self._data = data
        pass

    def make_prediction(self):
        raise NotImplementedError
