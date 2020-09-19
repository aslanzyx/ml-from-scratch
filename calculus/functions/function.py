import numpy as np

class function:
    def gradient(self, x: np.ndarray):
        raise NotImplementedError

    def calculate(self):
        raise NotImplementedError