import numfracpy

from scale_function.scale_function import AnalyticalScaleFunction
from random_process.skewed_stable_random_process import TotallySkewedStableRandomProcess


class TotallySkewedStableScaleFunction(AnalyticalScaleFunction):
    def __init__(self, q: float, process: TotallySkewedStableRandomProcess) -> None:
        super().__init__(q, process)

    def value(self, x: float) -> float:
        a = self.process.alpha
        delta = 0.0001
        dE = numfracpy.Mittag_Leffler_one(self.q * x**(a)+delta, a) - numfracpy.Mittag_Leffler_one(self.q * x**(a), a)
        dE /= delta
        return a * x**(a - 1) * dE
    