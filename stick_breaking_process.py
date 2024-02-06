from abc import ABC, abstractmethod
import numpy as np

from random_process import RandomProcess


class StickBreakingProcess(ABC):
    def __init__(self, process: RandomProcess) -> None:
        rng = np.random.default_rng()
        self.poisson_generator = rng.poisson
        self.process = process
        
    @abstractmethod
    def measure(self, t0: float, t1: float) -> None:
        return None
    
    @abstractmethod
    def sample(self, N: int) -> np.ndarray:
        return None
    