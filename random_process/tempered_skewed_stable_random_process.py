from typing import Callable
import numpy as np

from random_process.tempered_spectrally_negative_levy_random_process import TemperedSpectrallyNegativeLevyRandomProcess
from random_variable.tempered_stable_random_variable import TemperedTotallySkewedStableRandomVariable

class TemperedTotallySkewedStableRandomProcess(TemperedSpectrallyNegativeLevyRandomProcess):
    def __init__(self, c: float, alpha: float) -> None:
        self.alpha = alpha
        self.c = c
        xi = self.get_underlying_xi_for_time(1)
        super().__init__(c, xi.mu, xi.sigma, xi.nu, 
                         xi.nu_unwarranted, xi.max_jump_cutoff)

    def get_underlying_xi_for_time(self, time: float) -> TemperedTotallySkewedStableRandomVariable:
        return TemperedTotallySkewedStableRandomVariable(self.c * time**(1/self.alpha), 
                                                         self.alpha, 1, time**(1/self.alpha),
                                                         min_jump_cutoff=2**(-8),
                                                         max_jump_cutoff=2**5)
    