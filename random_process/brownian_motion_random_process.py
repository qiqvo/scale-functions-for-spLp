from typing import Callable
import numpy as np

from random_process.random_process import RandomProcess
from random_variable.normal_random_variable import NormalRandomVariable


class DriftBrownianMotionRandomProcess(RandomProcess):
    def __init__(self, drift, sigma2) -> None:
        super().__init__()
        self.drift = drift
        self.sigma2 = sigma2
        self.xi = self.get_underlying_xi_for_time(1)

    def get_underlying_xi_for_time(self, time: float) -> NormalRandomVariable:
        res = NormalRandomVariable(time * self.drift, time * self.sigma2)
        return res
    
    ### the subsequent methods are simplified due to xi being self-similar 
    def characteristic_function(self, t: np.complex64, time: float, z: np.float64) -> np.complex64:
        return self.xi.characteristic_function(t * np.power(time, 1/2)) * np.exp(1j * z * t)

    def laplace_transform(self, t: np.float64, time: float, z: np.float64) -> np.float64:
        return self.xi.laplace_transform(t * np.power(time, 1/2)) * np.exp(-z * t)

    def mean(self, time: float, z: np.float64) -> np.float64:
        return self.xi.mean() * np.power(time, 1/2) + z

    def variance(self, time: float, z: np.float64) -> np.float64:
        return self.xi.variance() * time

    def sample(self, N: int, time: float, z: np.float64) -> np.ndarray[float]:
        return z + self.xi.sample(N) * np.power(time, 1/2)
    
    def sample_function(self, N: int, theta: Callable, time: float, z: np.float64) -> np.ndarray[float]:
        return theta(self.sample(N, time, z))