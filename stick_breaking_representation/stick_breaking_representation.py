from abc import ABC, abstractmethod
import numpy as np

from random_process.random_process import RandomProcess


class StickBreakingRepresentation(ABC):
    def __init__(self, process: RandomProcess) -> None:
        self.process = process
        
    @abstractmethod
    def measure(self, t0: float, t1: float) -> None:
        return None
    
    @abstractmethod
    def sample(self, N: int) -> np.ndarray:
        return None
    