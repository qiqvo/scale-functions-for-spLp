import numpy as np

from i_random import IRandom

from random_process.random_process import RandomProcess
from stick_breaking_representation.stick_breaking_representation import PoissonSampledStickBreakingRepresentation


class PoissonSampledInfIntervalStickBreakingRepresentation(PoissonSampledStickBreakingRepresentation, IRandom):
    # simulating (l, xi) on s<l<t 
    def __init__(self, process: RandomProcess) -> None:
        super().__init__(process)
    
    def measure(self, s: float, t: float, u: float, v: float) -> float:
        # l\in (u, v) conditioned to l\in (s, t)
        return np.log(u / v) / np.log(t / s)
    
    def sample(self, N: int, s: float, t: float) -> np.ndarray:
        ts = t / s
        M = np.log(ts)
        P = self.rng.poisson(M, N)
        sam = np.zeros((N, 2, np.max(P)))

        for i in range(N):
            n = P[i]
            ls = s * np.power(ts, self.rng.uniform(0, 1, n))
            xis = []
            for l in ls:
                xi = self.process.sample(1, l, 0)[0]
                xis.append(xi)
            sam[i,0,:n] = ls[:]
            sam[i,1,:n] = np.array(xis)
        return sam
    
class InfIntervalStickBreakingRepresentation(PoissonSampledInfIntervalStickBreakingRepresentation):
    # simulating (l, xi) on s<l<t 
    def __init__(self, process: RandomProcess, s: float, t: float) -> None:
        assert s < t
        super().__init__(process)
        self._s = s
        self._t = t
        self.M = np.log(t / s)
    
    def measure(self, s: float, t: float) -> float:
        if s < self._s: 
            s = self._s
        if t > self._t:
            t = self._t
        return self.measure_unwarranted(s, t)
    
    def measure_unwarranted(self, s: float, t: float) -> float: 
        m = np.log(t / s)
        return m / self.M

    def sample(self, N: int) -> np.ndarray:
        return super().sample(N, self._s, self._t)
    