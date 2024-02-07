import numpy as np
from random_process import RandomProcess

from stick_breaking_process import StickBreakingProcess

# TODO: add Exp(theta) interval
class FixedIntervalStickBreakingProcess(StickBreakingProcess):
    # work between eps and 1/eps
    def __init__(self, process: RandomProcess, T: float, n_sticks: int) -> None:
        super().__init__(process)
        self._T = T
        self._n_sticks = n_sticks
    
    def measure(self, t0: float, t1: float) -> float:
        return None # np.infty
    
    def quick_measure(self, t0: float, t1: float) -> float:
        return None # np.infty

    def sample(self, N: int) -> np.ndarray:
        s = []
        for _ in range(N):
            ls = []
            xis = []
            l = self._T
            n = 0
            while n < self._n_sticks:
                ls.append(l * np.random.uniform(0, 1))
                l = ls[-1]
                xi = self.process.sample(1, l, 0)[0]
                xis.append(xi)
                n += 1
            s.append((np.array(ls), np.array(xis)))
        return np.array(s)