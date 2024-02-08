from typing import Callable
import numpy as np

from random_process.skewed_stable_random_process import TotallySkewedStableRandomProcess

class PosDriftTotallySkewedStableRandomProcess(TotallySkewedStableRandomProcess):
    def __init__(self, alpha: float, drift: float) -> None:
        super().__init__(alpha)
        self.drift = drift

    def characteristic_function(self, t: np.complex64, time: float, z: np.float64) -> np.complex64:
        return super().characteristic_function(t, time, z) * np.exp(1j * t * self.drift * time)

    def laplace_transform(self, t: np.float64, time: float, z: np.float64) -> np.float64:
        return super().laplace_transform(t, time, z) * np.exp(-t * self.drift * time)

    def mean(self, time: float, z: np.float64) -> np.float64:
        return super().mean(time, z) + self.drift * time

    def variance(self, time: float, z: np.float64) -> np.float64:
        return super().variance(time, z)

    def sample(self, N: int, time: float, z: np.float64) -> np.ndarray[float]:
        return super().sample(N, time, z) + self.drift * time
    
    def sample_function(self, N: int, theta: Callable, time: float, z: np.float64) -> np.ndarray[float]:
        return theta(self.sample(N, time, z))
