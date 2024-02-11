from abc import ABC, abstractmethod
from typing import Callable
import numpy as np
from scipy import optimize

from random_variable.random_variable import RandomVariable


class ConstRandomVariable(RandomVariable):
    def __init__(self, const: float) -> None:
        super().__init__()
        self.const = const
        
    def characteristic_function(self, t: np.complex64) -> np.complex64:
        return np.exp(1j * t * self.const)

    def psi(self, t: np.float64) -> np.float64:
        return - t * self.const

    def pdf(self, x: np.float64) -> np.float64:
        return 1 if x == self.const else 0

    def cdf(self, x: np.float64) -> np.float64:
        return 1 if x >= self.const else 0
    
    def inverse_cdf(self, x: np.float64) -> np.float64:
        return None

    def mean(self) -> np.float64:
        return self.const

    def variance(self) -> np.float64:
        return 0.0

    def sample(self, N: int) -> np.ndarray[float]:
        return np.full(N, self.const)

    def sample_function(self, N: int, theta: Callable) -> np.ndarray[float]:
        return np.full(N, theta(self.const))