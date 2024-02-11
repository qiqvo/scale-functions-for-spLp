from abc import ABC, abstractmethod
from typing import Callable
import numpy as np
from scipy import optimize

from random_variable.random_variable import RandomVariable


class SumOfIndependentRandomVariable(RandomVariable):
    def __init__(self, rv1: RandomVariable, rv2: RandomVariable) -> None:
        super().__init__()
        self.rv1 = rv1
        self.rv2 = rv2

    def characteristic_function(self, t: np.complex64) -> np.complex64:
        return self.rv1.characteristic_function(t) * self.rv2.characteristic_function(t)

    def laplace_transform(self, t: np.float64) -> np.float64:
        return self.rv1.laplace_transform(t) * self.rv2.laplace_transform(t)
    
    def psi(self, t: np.float64) -> np.float64:
        return self.rv1.psi(t) + self.rv2.psi(t)
    
    def pdf(self, x: np.float64) -> np.float64:
        return None

    def cdf(self, x: np.float64) -> np.float64:
        return None
    
    def mean(self) -> np.float64:
        return self.rv1.mean() + self.rv2.mean()

    def variance(self) -> np.float64:
        return self.rv1.variance() + self.rv2.variance()

    def sample(self, N: int) -> np.ndarray[float]:
        s1 = self.rv1.sample(N)
        s2 = self.rv2.sample(N)
        return s1 + s2