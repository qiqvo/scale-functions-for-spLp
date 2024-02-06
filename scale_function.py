

import numpy as np
from inf_interval_stick_breaking_process import InfIntervalStickBreakingProcess
from random_process import RandomProcess
from stick_breaking_process import StickBreakingProcess


class ScaleFunction():
    def __init__(self, process: RandomProcess) -> None:
        # self.q = q
        self.process = process 
        self.m = self.process.mean(1, 0)
        self._setup()

    def _setup(self) -> StickBreakingProcess:
        if self.m <= 0:
            self.measure_change()

class SBScaleFunction(ScaleFunction):
    """
    Scale function via Stick breaking process. 
    """
    def __init__(self, process: RandomProcess) -> None:
        super().__init__(process)

    def sampled_value(self, x: float, N: int):
        assert self.m > 0

        sbs = self.stick_breaking.sample(N)
        ps = []
        for i in range(N):
            xis = sbs[i][1]
            x_i = np.sum(xis, where=xis < 0)
            ps.append(x_i > -x)
        
        p = np.sum(ps) / N
        return p / self.m
        


class IISBScaleFunction(SBScaleFunction):
    """
    Scale function on the Inf Interval Stick breaking process. 
    """
    def __init__(self, process: RandomProcess, epsilon: float) -> None:
        self.epsilon = epsilon 
        super().__init__(process)

    def _setup(self) -> InfIntervalStickBreakingProcess:
        super()._setup()
        self.stick_breaking = InfIntervalStickBreakingProcess(self.process, self.epsilon)
    

class FISBScaleFunction(SBScaleFunction):
    """
    Scale function on the Fixed Interval Stick breaking process. 
    """
    def __init__(self, process: RandomProcess, epsilon: float) -> None:
        self.epsilon = epsilon 
        super().__init__(process)

    def _setup(self) -> InfIntervalStickBreakingProcess:
        super()._setup()
        self.stick_breaking = InfIntervalStickBreakingProcess(self.process, self.epsilon)
    