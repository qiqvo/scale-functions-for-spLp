from abc import ABC, abstractmethod
from typing import Callable
import numpy as np
from scipy import optimize

from random_variable.random_variable import RandomVariable


class MultipleOfRandomVariable(RandomVariable):
    def __init__(self, multiplier: float, random_variable: RandomVariable) -> None:
        super().__init__()
        self.multiplier = multiplier
        self.random_variable = random_variable

    def characteristic_function(self, t: np.complex64) -> np.complex64:
        return self.random_variable.characteristic_function(self.multiplier * t)

    def laplace_transform(self, t: np.float64) -> np.float64:
        return self.random_variable.laplace_transform(self.multiplier * t)

    def pdf(self, x: np.float64) -> np.float64:
        return self.random_variable.pdf(x / self.multiplier) / self.multiplier

    def cdf(self, x: np.float64) -> np.float64:
        return self.random_variable.cdf(x / self.multiplier)

    def mean(self) -> np.float64:
        return self.random_variable.mean() * self.multiplier

    def variance(self) -> np.float64:
        return self.random_variable.variance() * self.multiplier * self.multiplier

    def sample(self, N: int) -> np.ndarray[float]:
        return self.random_variable.sample(N) * self.multiplier