from typing import Any, Callable
import numpy as np
import scipy

from random_variable.spectrally_negative_levy_random_variable import DecreasingDensitySpectrallyNegativeLevyRandomVariable
from random_variable.tempered_spectrally_negative_levy_random_variable import TemperedSpectrallyNegativeLevyRandomVariable


class TemperedTotallySkewedStableRandomVariable(TemperedSpectrallyNegativeLevyRandomVariable, DecreasingDensitySpectrallyNegativeLevyRandomVariable):
    def __init__(self, c: float, alpha: float, char_multiplier: float=1, 
                 amplitude_multiplier: float=1, 
                 min_jump_cutoff: float = 2**(-5), max_jump_cutoff: float = 2 ** 12) -> None:
        self.alpha = alpha
        self._min_jump_cutoff = min_jump_cutoff

        # this _const shows up in Proof of Thm 14.10 in Sato:
        self._const = scipy.special.gamma(-self.alpha)

        mu = 1/(self.alpha - 1) / self._const
        nu_unwarranted = lambda x: 1 / (-x)**(1 + self.alpha) / self._const
        nu = lambda x: nu_unwarranted(x) if x <= -self._min_jump_cutoff else 0

        super().__init__(c, mu, 0, nu, nu_unwarranted, char_multiplier, amplitude_multiplier, max_jump_cutoff=max_jump_cutoff)

    def get_tempered_mu(self, mu: float, nu: Callable[..., Any]):
        # by computing the mean
        mu = scipy.integrate.quad(lambda x: np.exp(-self.c / x) * (x)**(self.alpha-2), 0, 1)[0] / self._const
        mu += self.mean() / self._char_multiplier 
        return mu

    def mean(self):
        return self._char_multiplier * self.alpha * self.c ** (self.alpha - 1)

    def psi(self, t: np.float64) -> np.float64:
        t *= self.amplitude_multiplier
        return self._char_multiplier * ((t + self.c)**self.alpha - (self.c)**self.alpha)
    
    def phi(self, q:np.float64, a:np.float64=0, b:np.float64=2**10) -> np.float64:
        res = np.power((self.c)**self.alpha + q/self._char_multiplier, 1/self.alpha) - self.c
        res /= self.amplitude_multiplier
        return res

    def get_min_jump_size(self):
        return self._min_jump_cutoff
    
    def get_shifted_sigma(self):
        return np.sqrt(scipy.integrate.quad(lambda x: self.nu_unwarranted(-x) * x * x, 0, self._min_jump_cutoff)[0])


class UntemperedTotallySkewedStableRandomVariable(TemperedTotallySkewedStableRandomVariable):
    def __init__(self, alpha: float, char_multiplier: float=1, 
                 amplitude_multiplier:float = 1, 
                 min_jump_cutoff: float = 2 ** (-5), max_jump_cutoff: float = 2 ** 12) -> None:
        super().__init__(0, alpha, char_multiplier, amplitude_multiplier, min_jump_cutoff, max_jump_cutoff)

    def get_tempered_mu(self, mu: float, nu: Callable[..., Any]):
        return mu
    
    def get_shifted_sigma(self):
        return np.sqrt(self._min_jump_cutoff ** (2 - self.alpha) / (2 - self.alpha) / self._const)
    
    def get_min_jump_size(self):
        return self._min_jump_cutoff
    
    def get_nu_compensation(self, a:float=-1, b:float=0):
        if b > -self._min_jump_cutoff:
            b = -self._min_jump_cutoff
        nu_compensation = ((-b) ** (1 - self.alpha) - (-a) ** (1 - self.alpha)) / (1 - self.alpha) / self._const
        return nu_compensation
    