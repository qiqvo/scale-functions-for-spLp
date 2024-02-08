from typing import Any, Callable
import numpy as np
import scipy

from random_variable.spectrally_negative_levy_random_variable import DecreasingDensitySpectrallyNegativeLevyRandomVariable
from random_variable.tempered_sn_levy_random_variable import TemperedSpectrallyNegativeLevyRandomVariable


class TemperedTotallySkewedStableRandomVariable(TemperedSpectrallyNegativeLevyRandomVariable, DecreasingDensitySpectrallyNegativeLevyRandomVariable):
    def __init__(self, alpha: float, c: float, min_jump_cutoff: float = 2**(-5), max_jump_cutoff: float = 2 ** 12) -> None:
        self.alpha = alpha
        self._min_jump_cutoff = min_jump_cutoff

        # this _const shows up in Proof of Thm 14.10 in Sato:
        self._const = scipy.special.gamma(-self.alpha)

        mu = 1/(self.alpha - 1) / self._const
        nu_unwarranted = lambda x: 1 / (-x)**(1 + self.alpha) / self._const
        nu = lambda x: nu_unwarranted(x) if x <= -self._min_jump_cutoff else 0

        super().__init__(c, mu, 0, nu, nu_unwarranted, max_jump_cutoff=max_jump_cutoff)

    def get_tempered_mu(self, mu: float, nu: Callable[..., Any]):
        # by computing the mean
        mu = scipy.integrate.quad(lambda x: np.exp(-self.c / x) * (x)**(self.alpha-2), 0, 1)[0] / self._const
        mu += self.mean()
        return mu

    def mean(self):
        return self.alpha * self.c ** (self.alpha - 1)

    def laplace_transform(self, t: np.float64) -> np.float64:
        return np.exp((t + self.c)**self.alpha - (self.c)**self.alpha)

    def get_min_jump_size(self):
        return self._min_jump_cutoff
    
    def get_shifted_sigma(self):
        return np.sqrt(scipy.integrate.quad(lambda x: self.nu_unwarranted(-x) * x * x, 0, self._min_jump_cutoff)[0])


class UntemperedTotallySkewedStableRandomVariable(TemperedTotallySkewedStableRandomVariable):
    def __init__(self, alpha: float, min_jump_cutoff: float = 2 ** (-5), max_jump_cutoff: float = 2 ** 12) -> None:
        super().__init__(alpha, 0, min_jump_cutoff, max_jump_cutoff)

    def get_tempered_mu(self, mu: float, nu: Callable[..., Any]):
        return mu
    
    def get_shifted_sigma(self):
        return np.sqrt(self._min_jump_cutoff ** (2 - self.alpha) / (2 - self.alpha) / self._const)
    
    def get_min_jump_size(self):
        return self._min_jump_cutoff
    
    def get_nu_compensation(self):
        nu_compensation = (self._min_jump_cutoff ** (1 - self.alpha) - 1) / (1 - self.alpha) / self._const
        return nu_compensation
    