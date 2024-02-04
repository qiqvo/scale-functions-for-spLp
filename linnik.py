from abc import ABC, abstractmethod
import typing
from typing import Any, Callable
import numpy as np
from scipy.stats import gengamma

from random_variable import RandomVariable
from stable_random_variable import StableRandomVariable


class Linnik(RandomVariable):
    def __init__(self, alpha: float, delta: float) -> None:
        self.alpha = alpha
        self.delta = delta
        self._v = gengamma(self.delta, 1 / self.delta)
        self._s = StableRandomVariable(self.alpha)

    def characteristic_function(self, t: np.complex64) -> np.complex64:
        return 1 / np.power((1 + np.power(np.abs(t), self.alpha)),self.delta)

    def laplace_transform(self, t: np.float64) -> np.float64:
        return np.real(self.characteristic_function(t))

    def pdf(self, x: np.float64) -> np.float64:
        return None

    def cdf(self, x: np.float64) -> np.float64:
        return None
    
    #TODO: check
    def mean(self, t: np.float64) -> np.float64:
        return np.infty

    #TODO: check
    def variance(self, t: np.float64) -> np.float64:
        return np.infty

    def sample(self, N: int) -> np.ndarray[float]:
        v = self._v.rvs(N)
        s = self._s.sample(N)
        return np.power(v, 1 / (self.alpha * self.delta)) * s
    
