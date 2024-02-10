import numfracpy
import scipy
from random_process.drift_random_process import DriftRandomProcess
from random_process.skewed_stable_random_process import TotallySkewedStableRandomProcess

from scale_function.scale_function import AnalyticalScaleFunction


class PosDriftTotallySkewedStableScaleFunction(AnalyticalScaleFunction):
    def __init__(self, q: float, process: DriftRandomProcess) -> None:
        assert type(process.process) == TotallySkewedStableRandomProcess
        assert process.drift >= 0
        assert q == 0
        super().__init__(q, process)
    
    def value(self, x: float) -> float:
        c = self.process.drift 
        a = self.process.process.alpha - 1
        if c == 0:
            return x ** (a) / scipy.special.gamma(a + 1)
        
        E = numfracpy.Mittag_Leffler_one(-c * x**(a), a)
        return (1 - E) / c
    