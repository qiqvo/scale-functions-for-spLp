import numpy as np
from random_process.random_process import RandomProcess

from stick_breaking_representation.stick_breaking_representation import StickBreakingRepresentation

class FixedIntervalStickBreakingRepresentation(StickBreakingRepresentation):
    def __init__(self, process: RandomProcess, T: float, n_sticks: int) -> None:
        super().__init__(process)
        self._T = T
        self._n_sticks = n_sticks
    
    def measure(self, t0: float, t1: float) -> float:
        return np.infty
    
    def quick_measure(self, t0: float, t1: float) -> float:
        return np.infty
    
    def get_T(self):
        return self._T

    def sample(self, N: int) -> np.ndarray:
        s = []
        for _ in range(N):
            ls = []
            xis = []
            l = self.get_T()
            n = 0
            while n < self._n_sticks:
                ls.append(l * np.random.uniform(0, 1))
                l = ls[-1]
                xi = self.process.sample(1, l, 0)[0]
                xis.append(xi)
                n += 1
            s.append((np.array(ls), np.array(xis)))
        return np.array(s)
    

class ExpIntervalStickBreakingRepresentation(FixedIntervalStickBreakingRepresentation):
    def __init__(self, process: RandomProcess, theta: float, n_sticks: int) -> None:
        super().__init__(process, 1/theta, n_sticks)

    def get_T(self):
        return np.random.exponential(1/self._T, 1)[0]