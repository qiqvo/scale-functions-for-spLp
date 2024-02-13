from abc import ABC, abstractmethod
from typing import Callable
import numpy as np
import scipy


class RandomVariable(ABC):
    _interval_a = 0 
    _interval_b = 1e10 

    def __init__(self) -> None:
        self._table = None 
        # self._table_v = []
        pass

    @abstractmethod
    def characteristic_function(self, t: np.complex64) -> np.complex64:
        return None

    def laplace_transform(self, t: np.float64) -> np.float64:
        return np.exp(-self.psi(t))

    @abstractmethod
    def psi(self, t: np.float64) -> np.float64:
        return None
    
    def phi(self, q:np.float64, a:np.float64=0, b:np.float64=2**10) -> np.float64:
        # if find_min:
        #     a = scipy.optimize.minimize_scalar(self.psi)
        if q == 0 and self.mean() < 0:
            a = 1
            while self.psi(a) > 0:
                b = a
                a = a // 2
        else:
            a = 0

        res = scipy.optimize.brentq(lambda t: self.psi(t) - q, a, b)
        return res

    @abstractmethod
    def pdf(self, x: np.float64) -> np.float64:
        return None

    @abstractmethod
    def cdf(self, x: np.float64) -> np.float64:
        return None
    
    def inverse_cdf(self, x: np.float64) -> np.float64:
        y = scipy.optimize.brentq(lambda t: self.cdf(t) - x, self._interval_a, self._interval_b)
        return y

    @abstractmethod
    def mean(self) -> np.float64:
        return None

    @abstractmethod
    def variance(self) -> np.float64:
        return None

    # TODO: finish: 
    @abstractmethod
    def sample(self, N: int) -> np.ndarray[float]:
        # if self._table is None:
        #     self._table = {}
        #     step = 0.01

        # u = np.random.uniform(0, 1, N)
        # s = self.inverse_cdf()
        return None

    def sample_function(self, N: int, theta: Callable) -> np.ndarray[float]:
        s = self.sample(N)
        return theta(s)