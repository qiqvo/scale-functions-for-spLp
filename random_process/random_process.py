from abc import ABC, abstractmethod
from typing import Callable
import numpy as np

from random_variable.random_variable import RandomVariable


class RandomProcess(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def characteristic_function(self, t: np.complex64, time: float, z: np.float64) -> np.complex64:
        return None

    @abstractmethod
    def laplace_transform(self, t: np.float64, time: float, z: np.float64) -> np.float64:
        return None

    @abstractmethod
    def mean(self, time: float, z: np.float64) -> np.float64:
        return None

    @abstractmethod
    def variance(self, time: float, z: np.float64) -> np.float64:
        return None

    @abstractmethod
    def sample(self, N: int, time: float, z: np.float64) -> np.ndarray[float]:
        return None
    
    @abstractmethod
    def sample_function(self, N: int, theta: Callable, time: float, z: np.float64) -> np.ndarray[float]:
        return None
    
    @abstractmethod
    def get_underlying_xi_for_time(self, time: float) -> RandomVariable:
        return None
    
    def is_infinite_activity(self) -> bool:
        return None