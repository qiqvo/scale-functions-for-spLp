import numfracpy

from random_process.pos_drift_skewed_stable_random_process import PosDriftTotallySkewedStableRandomProcess
from scale_function.scale_function import AnalyticalScaleFunction


class PosDriftTotallySkewedStableScaleFunction(AnalyticalScaleFunction):
    def __init__(self, q: float, process: PosDriftTotallySkewedStableRandomProcess) -> None:
        assert process.drift > 0
        assert q == 0
        super().__init__(q, process)
    
    def value(self, x: float) -> float:
        c = self.process.drift 
        a = self.process.alpha - 1
        E = numfracpy.Mittag_Leffler_one(-c * x**(a), a)
        return (1 - E) / c
    