from typing import Callable
import numpy as np

from random_process.spectrally_negative_levy_random_process import SpectrallyNegativeLevyRandomProcess
from random_variable.skewed_stable_random_variable import TotallySkewedStableRandomVariable

class TotallySkewedStableRandomProcess(SpectrallyNegativeLevyRandomProcess):
    def __init__(self, alpha: float) -> None:
        self.alpha = alpha
        self._k = 1 - abs(1 - self.alpha)
        self.xi = TotallySkewedStableRandomVariable(alpha)
        super().__init__(self.xi.mu, self.xi.sigma, self.xi.nu, 
                         self.xi.nu_unwarranted, self.xi._max_jump_cutoff)

    def get_underlying_xi_for_time(self, time: float) -> TotallySkewedStableRandomVariable:
        return TotallySkewedStableRandomVariable(self.alpha)
    
    def characteristic_function(self, t: np.complex64, time: float, z: np.float64) -> np.complex64:
        return self.xi.characteristic_function(t * np.power(time, 1/self.alpha)) * np.exp(1j * z * t)

    def laplace_transform(self, t: np.float64, time: float, z: np.float64) -> np.float64:
        return self.xi.laplace_transform(t * np.power(time, 1/self.alpha)) * np.exp(-z * t)

    def mean(self, time: float, z: np.float64) -> np.float64:
        return self.xi.mean() * np.power(time, 1/self.alpha) + z

    def variance(self, time: float, z: np.float64) -> np.float64:
        return self.xi.variance() * np.power(time, 2/self.alpha)

    def sample(self, N: int, time: float, z: np.float64) -> np.ndarray[float]:
        return z + self.xi.sample(N) * np.power(time, 1/self.alpha)
    
    def sample_function(self, N: int, theta: Callable, time: float, z: np.float64) -> np.ndarray[float]:
        return theta(self.sample(N, time, z))