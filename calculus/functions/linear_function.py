import numpy as np
from function import function


class linear_function(function):
    '''
    A linear function is a function in form of
    f(x)=AX
    where X = [X, X**2, X**3]
    '''

    def __init__(self, param: np.ndarray, constant: float):
        self._param: np.ndarray = param

    def gradient(self, x: np.ndarray) -> float:
        retval = []
        const = sum(self._param[0])
        # for each variable in x
        for x_i in range(self._param[])
        # for each power layer

        for i in range(self.power()):
            retval.append(self._param[i])

    def grad_func(self):
        return None

    def calculate(self, x: np.ndarray) -> float:
        if self._param.shape[1] != x.shape[1]:
            raise Exception("Size error")
        return sum(self._param*[x**i for i in range(self.power())])

    def power(self) -> int:
        return self._param.shape[0]

    def dimension(self) -> int:
        return self._param.shape[1]