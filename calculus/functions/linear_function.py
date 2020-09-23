import numpy as np
from function import function

class linear_function:
    def __init__(self, param: np.ndarray, constant: float):
        self._param: np.ndarray = param

    def gradient(self, x: np.ndarray):
        retval = []
        const = sum(self._param[0])
        # for each variable in x
        for i in range(self._param[0]):
            # for each power layer
            retval.append(0)
            for j in range(self.power()):
                retval[i] += self._param[i][j]*j*x[i]**(j-1)
        return np.array(retval)
                

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
