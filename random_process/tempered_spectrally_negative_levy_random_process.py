from typing import Any, Callable
import numpy as np

from random_process.spectrally_negative_levy_random_process import SpectrallyNegativeLevyRandomProcess
from random_variable.spectrally_negative_levy_random_variable import SpectrallyNegativeLevyRandomVariable
from random_variable.tempered_spectrally_negative_levy_random_variable import TemperedSpectrallyNegativeLevyRandomVariable

class TemperedSpectrallyNegativeLevyRandomProcess(SpectrallyNegativeLevyRandomProcess):
    def __init__(self, c: float, mu: float, sigma: float, nu: Callable[..., Any], nu_unwarranted: Callable[..., Any] = None, max_jump_cutoff: float = 2 ** 12) -> None:
        super().__init__(mu, sigma, nu, nu_unwarranted, max_jump_cutoff)
        self.c = c

    def _get_underlying_xi_for_time(self, time: float) -> SpectrallyNegativeLevyRandomVariable:
        return TemperedSpectrallyNegativeLevyRandomVariable(self.c, 
                                                    time * self.mu, 
                                                    np.sqrt(time) * self.sigma, 
                                                    lambda x: time * self.nu(x), 
                                                    lambda x: time * self.nu_unwarranted(x) if self.nu_unwarranted is not None else None, 
                                                    self.max_jump_cutoff)
