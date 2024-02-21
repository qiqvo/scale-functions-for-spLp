from typing import Callable
import numpy as np

from random_process.random_process import RandomProcess
from random_variable.normal_random_variable import NormalRandomVariable


class BrownianMotionRandomProcess(RandomProcess):
    def __init__(self) -> None:
        self.xi = None

    def get_underlying_xi_for_time(self, time: float) -> NormalRandomVariable:
        res = NormalRandomVariable(0, time)
        return res
    
    def characteristic_function(self, t: np.complex64, time: float, z: np.float64) -> np.complex64:
        return self.get_underlying_xi_for_time(time).characteristic_function(t) * np.exp(1j * z * t)

    def laplace_transform(self, t: np.float64, time: float, z: np.float64) -> np.float64:
        return self.get_underlying_xi_for_time(time).laplace_transform(t) * np.exp(-z * t)

    def mean(self, time: float, z: np.float64) -> np.float64:
        return self.get_underlying_xi_for_time(time).mean() + z

    def variance(self, time: float, z: np.float64) -> np.float64:
        return self.get_underlying_xi_for_time(time).variance()

    def sample(self, N: int, time: float, z: np.float64) -> np.ndarray[float]:
        return z + self.get_underlying_xi_for_time(time).sample(N)
    
    def sample_function(self, N: int, theta: Callable, time: float, z: np.float64) -> np.ndarray[float]:
        return theta(self.sample(N, time, z))