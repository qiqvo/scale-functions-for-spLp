from typing import Callable
import numpy as np

from random_process.random_process import RandomProcess
from random_variable.spectrally_negative_levy_random_variable import SpectrallyNegativeLevyRandomVariable

class SpectrallyNegativeLevyRandomProcess(RandomProcess):
    def __init__(self, mu: float, sigma: float, nu: Callable, nu_unwarranted: Callable=None, max_jump_cutoff:float=2**12) -> None:
        self.mu = mu
        self.sigma = sigma
        self.nu = nu
        self.nu_unwarranted = nu_unwarranted
        self.max_jump_cutoff = max_jump_cutoff

    def get_underlying_xi_for_time(self, time: float) -> SpectrallyNegativeLevyRandomVariable:
        return SpectrallyNegativeLevyRandomVariable(self.mu, 
                                                    self.sigma, 
                                                    lambda x: self.nu(x), 
                                                    lambda x: self.nu_unwarranted(x) if self.nu_unwarranted is not None else None, 
                                                    multiplier=time,
                                                    max_jump_cutoff=self.max_jump_cutoff)

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