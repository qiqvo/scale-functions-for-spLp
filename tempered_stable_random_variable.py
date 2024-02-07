import numpy as np
import scipy
from levy_random_variable import DecreasingDensitySpectrallyNegativeLevyRandomVariable


class TemperedTotallySkewedStableRandomVariable(DecreasingDensitySpectrallyNegativeLevyRandomVariable):
    """
    no gaussian approx is made here. 
    """
    def __init__(self, alpha: float, c: float, min_jump_cutoff: float = 2**(-5), max_jump_cutoff: float = 2 ** 12) -> None:
        self.alpha = alpha
        self.c = c
        self._min_jump_cutoff = min_jump_cutoff

        # this _const shows up in Proof of Thm 14.10 in Sato:
        self._const = scipy.special.gamma(-self.alpha)

        nu_unwarranted = lambda x: np.exp(c * x) / (-x)**(1 + alpha) / self._const
        nu = lambda x: nu_unwarranted(x) if x <= -min_jump_cutoff else 0

        self._O2 = self._min_jump_cutoff ** (2 - self.alpha) / (2 - self.alpha) / self._const
        mu = 1/(alpha - 1) / self._const

        super().__init__(mu, np.sqrt(self._O2), nu, nu_unwarranted=nu_unwarranted, max_jump_cutoff=max_jump_cutoff)

    def laplace_transform(self, t: np.float64) -> np.float64:
        return np.exp((t + self.c)**self.alpha - (self.c)**self.alpha)

    def get_min_jump_size(self):
        return self._min_jump_cutoff
    
    # def get_shifted_sigma(self):
    #     return np.sqrt(self._O2)

    def _calculate_integrals(self):
        # self._O2 = self._min_jump_cutoff ** (2 - self.alpha) / (2 - self.alpha)

        self._I1 = (self._min_jump_cutoff ** (1 - self.alpha) - 1) / (1 - self.alpha) / self._const
        self._I2 = (1 - self._min_jump_cutoff ** (2 - self.alpha)) / (2 - self.alpha) / self._const

        # TODO: check:
        self._II2 = scipy.integrate.quad(lambda x: x * x * self.nu(x), -self._max_jump_cutoff, -1)[0] / self._const
        self._II2 -= (scipy.integrate.quad(lambda x: x * self.nu(x), -self._max_jump_cutoff, -1))[0]**2 / self._const**2
