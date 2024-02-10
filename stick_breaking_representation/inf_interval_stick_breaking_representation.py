import numpy as np


from settings import seed

from random_process.random_process import RandomProcess
from stick_breaking_representation.stick_breaking_representation import StickBreakingRepresentation


class InfIntervalStickBreakingRepresentation(StickBreakingRepresentation):
    # work between eps and 1/eps
    def __init__(self, process: RandomProcess, epsilon: float) -> None:
        super().__init__(process)

        self.rng = np.random.default_rng(seed=seed)

        self._eps = epsilon
        self._inv_eps = 1/epsilon
        self.M = 2 * np.log(self._inv_eps)
    
    def measure(self, t0: float, t1: float) -> float:
        if t0 < self._eps: 
            t0 = self._eps
        if t1 > self._inv_eps:
            t1 = self._inv_eps
        return self.quick_measure(t0, t1)
    
    def quick_measure(self, t0: float, t1: float) -> float: 
        m = np.log(t1) - np.log(t0)
        return m / self.M

    def sample(self, N: int) -> np.ndarray:
        P = self.rng.poisson(self.M, N)
        s = np.zeros((N, 2, np.max(P)))

        for i in range(N):
            n = P[i]
            ls = np.power(self._eps, self.rng.uniform(-1, 1, n))
            xis = []
            for l in ls:
                xi = self.process.sample(1, l, 0)[0]
                xis.append(xi)
            s[i,0,:n] = ls[:]
            s[i,1,:n] = np.array(xis)
        return s