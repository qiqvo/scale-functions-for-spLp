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

    def is_infinite_activity(self) -> bool:
        return True
    
    def get_underlying_xi_for_time(self, time: float) -> TemperedTotallySkewedStableRandomVariable:
        min_jump_cutoff = 2**(-5)
        max_jump_cutoff = 2**5
        c_timed = self.c * time**(1/self.alpha)
        if c_timed > 10:
            max_jump_cutoff = 2**2
        if c_timed < 0.1:
            max_jump_cutoff = 2**10
            min_jump_cutoff = 2**-2

        return TemperedTotallySkewedStableRandomVariable(self.c * time**(1/self.alpha), 
                                                         self.alpha, 1, time**(1/self.alpha),
                                                         min_jump_cutoff=min_jump_cutoff,
                                                         max_jump_cutoff=max_jump_cutoff)
    