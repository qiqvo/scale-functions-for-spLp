import numpy as np
from numpy import ndarray
from numpy.core.multiarray import array as array
from random_process.random_process import RandomProcess
from scale_function.scale_function import ScaleFunction
from stick_breaking_representation.stick_breaking_representation import StickBreakingRepresentation


class SBScaleFunction(ScaleFunction):
    """
    Scale function via Stick breaking process. 
    """
    def __init__(self, q: float, process: RandomProcess, 
                 stick_breaking_process: StickBreakingRepresentation,
                 N: int) -> None:
        super().__init__(q, process)
        
        self.stick_breaking_process = stick_breaking_process 
        self.N = N
        self.resample()

    # def _setup(self) -> None:
    #     if not (self.q > 0 or (self.m < 0 and self.q == 0)):
    #         return None
    #     self.original_process = self.process
    #     self.original_q = self.q
    #     self.original_m = self.m

    #     c = 
    #     self.process = 
    
    def resample(self):
        self.stick_breaking_samples = self.stick_breaking_process.sample(self.N)

    def value(self, x: float):
        ps = []
        for i in range(self.N):
            xis = self.stick_breaking_samples[i][1]
            x_i = np.sum(xis, where=xis < 0)
            ps.append(x_i > -x)
        
        p = np.sum(ps) / self.N
        return p / self.m

    def profile(self, range_x: np.array) -> ndarray:
        ps = []
        for i in range(self.N):
            xis = self.stick_breaking_samples[i][1]
            x_i = np.sum(xis, where=xis < 0)
            ps.append(x_i)
        ps = np.array(ps)

        p = []
        for x in range_x:
            p.append(np.sum(ps>-x) / self.N / self.m)
        return np.array(p)