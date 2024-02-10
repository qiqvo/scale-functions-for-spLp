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
    def profile(self, a: float, b: float) -> tuple[np.ndarray, np.ndarray]:
        return None


class AnalyticalScaleFunction(ScaleFunction):
    def profile(self, a: float, b: float) -> tuple[np.ndarray, np.ndarray]:
        xs = np.linspace(a, b, endpoint=True)
        values = np.array([self.value(x) for x in xs])
        return xs, values
    