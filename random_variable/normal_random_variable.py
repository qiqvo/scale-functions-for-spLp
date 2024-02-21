from abc import ABC, abstractmethod
from typing import Callable
import numpy as np
import scipy
from i_random import IRandom

from random_variable.random_variable import RandomVariable


class NormalRandomVariable(RandomVariable, IRandom):
    def __init__(self, mu: np.float64, sigma2: np.float64) -> None:
        super().__init__()
        self.mu = mu
        self.sigma2 = sigma2

    def characteristic_function(self, t: np.complex64) -> np.complex64:
        return np.exp(self.mu * 1j * t - self.sigma2 * t*t / 2)

    def laplace_transform(self, t: np.float64) -> np.float64:
        return np.exp(-self.psi(t))

    def psi(self, t: np.float64) -> np.float64:
        return - self.mu * t + self.sigma2 * t*t / 2
    
    def phi(self, q:np.float64, a:np.float64=0, b:np.float64=2**10) -> np.float64:
        res = self.mu + np.sqrt(self.mu*self.mu + 2 * q * self.sigma2)
        res /= self.sigma2
        return res

    def pdf(self, x: np.float64) -> np.float64:
        return np.exp(-x*x / 2 / self.sigma2) / np.sqrt(2 * np.pi * self.sigma2) 

    def cdf(self, x: np.float64) -> np.float64:
        x = (x - self.mu) / np.sqrt(self.sigma2)
        return (1 + scipy.special.erf(x / np.sqrt(2))) / 2
    
    def mean(self) -> np.float64:
        return self.mu

    def variance(self) -> np.float64:
        return self.sigma2

    def sample(self, N: int) -> np.ndarray[float]:
        return self.rng.normal(self.mu, np.sqrt(self.sigma2), N)
