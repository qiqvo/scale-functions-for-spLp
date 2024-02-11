from typing import Callable
import numpy as np

from random_process.random_process import RandomProcess
from random_variable.const_random_variable import ConstRandomVariable
from random_variable.random_variable import RandomVariable
from random_variable.sum_of_independent_random_variable import SumOfIndependentRandomVariable


class DriftRandomProcess(RandomProcess):
    def __init__(self, drift: float, process: RandomProcess) -> None:
        self.drift = drift
        self.process = process
    
    def characteristic_function(self, t: np.complex64, time: float, z: np.float64) -> np.complex64:
        return self.process.characteristic_function(t, time, z) * np.exp(1j * t * self.drift * time)

    def laplace_transform(self, t: np.float64, time: float, z: np.float64) -> np.float64:
        return self.process.laplace_transform(t, time, z) * np.exp(-t * self.drift * time)

    def mean(self, time: float, z: np.float64) -> np.float64:
        return self.process.mean(time, z) + self.drift * time

    def variance(self, time: float, z: np.float64) -> np.float64:
        return self.process.variance(time, z)

    def sample(self, N: int, time: float, z: np.float64) -> np.ndarray[float]:
        return self.process.sample(N, time, z) + self.drift * time
    
    def sample_function(self, N: int, theta: Callable, time: float, z: np.float64) -> np.ndarray[float]:
        return theta(self.process.sample(N, time, z))

    def get_underlying_xi_for_time(self, time: float) -> RandomVariable:
        xi = self.process.get_underlying_xi_for_time(time)
        drift_rv = ConstRandomVariable(self.drift * time)
        return SumOfIndependentRandomVariable(xi, drift_rv)
    