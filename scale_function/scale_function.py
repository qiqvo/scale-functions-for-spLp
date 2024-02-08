from abc import ABC, abstractmethod
import numpy as np
from random_process.random_process import RandomProcess


class ScaleFunction(ABC):
    def __init__(self, q:float, process: RandomProcess) -> None:
        self.q = q
        self.process = process 
        self.m = self.process.mean(1, 0)

    @abstractmethod
    def value(self, x: float) -> float:
        return None
    
    @abstractmethod
    def profile(self, range_x: np.array) -> np.ndarray:
        return None


class AnalyticalScaleFunction(ScaleFunction):
    def profile(self, range_x: np.array) -> np.ndarray:
        return np.array([self.value(x) for x in range_x])
    