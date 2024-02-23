import numpy as np
from i_random import IRandom

from random_process.random_process import RandomProcess

from stick_breaking_representation.stick_breaking_representation import StickBreakingRepresentation

class FixedIntervalStickBreakingRepresentation(StickBreakingRepresentation, IRandom):
    def __init__(self, process: RandomProcess, T: float, n_sticks: int) -> None:
        super().__init__(process)
        self._T = T
        self._n_sticks = n_sticks
    
    def measure(self, t0: float, t1: float) -> float:
        return None
    
    def quick_measure(self, t0: float, t1: float) -> float:
        return None
    
    def get_T(self):
        return self._T

    def sample(self, N: int) -> np.ndarray:
        s = np.empty((N, 2, self._n_sticks))
        for i in range(N):
            l_xis = np.empty((2, self._n_sticks))
            T = self.get_T()
            for j in range(self._n_sticks):
                l = T * self.rng.uniform(0, 1)
                T -= l
                xi = self.process.sample(1, l, 0)[0]
                l_xis[:, j] = (l, xi)
            s[i] = l_xis
        return s
    

class ExpIntervalStickBreakingRepresentation(FixedIntervalStickBreakingRepresentation):
    def __init__(self, process: RandomProcess, theta: float, n_sticks: int) -> None:
        super().__init__(process, 1/theta, n_sticks)

    def get_T(self):
        return self.rng.exponential(self._T, 1)[0]