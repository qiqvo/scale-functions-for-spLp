import numpy as np

from random_process.brownian_motion_random_process import DriftBrownianMotionRandomProcess
from scale_function.scale_function import AnalyticalScaleFunction


class BrownianMotionScaleFunction(AnalyticalScaleFunction):
    def __init__(self, q: float, process: DriftBrownianMotionRandomProcess) -> None:
        super().__init__(q, process)

    def value(self, x: float) -> float:
        x = x / self.process.sigma2
        if self.q == 0 and self.m == 0:
            return 2 * x

        a = np.sqrt(2 * self.q * self.process.sigma2 + self.process.drift)
        res = np.sinh(x * a)
        res *= np.exp(-self.process.drift * x)
        res *= 2 / a
        return res