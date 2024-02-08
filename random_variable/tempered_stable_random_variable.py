import numpy as np
import scipy
from random_variable.levy_random_variable import DecreasingDensitySpectrallyNegativeLevyRandomVariable


class TemperedTotallySkewedStableRandomVariable(DecreasingDensitySpectrallyNegativeLevyRandomVariable):
    def __init__(self, alpha: float, c: float, min_jump_cutoff: float = 2**(-5), max_jump_cutoff: float = 2 ** 12) -> None:
        self.alpha = alpha
        self.c = c
        self._min_jump_cutoff = min_jump_cutoff

        # this _const shows up in Proof of Thm 14.10 in Sato:
        self._const = scipy.special.gamma(-self.alpha)

        nu = lambda x: self._inner_nu(x) if x <= -min_jump_cutoff else 0

        sigma = self.get_shifted_sigma()
        mu = self.get_mu()

        super().__init__(mu, sigma, nu, nu_unwarranted=self._inner_nu, max_jump_cutoff=max_jump_cutoff)

    def _inner_nu(self, x):
        return np.exp(self.c * x) / (-x)**(1 + self.alpha) / self._const

    # def get_mu(self):
    #     # see for example Ch 9.5 of Financial Modelling with Jump Processes By Rama Cont, Peter Tankov
    #     mu = 1/(self.alpha - 1) 
    #     mu -= scipy.integrate.quad(lambda x: (np.exp(self.c * x) - 1) / (-x)**(self.alpha), -1, 0)[0] 
    #     mu /= self._const
    #     print(mu)
    #     return mu

    def get_mu(self):
        # by computing the mean
        mu = scipy.integrate.quad(lambda x: np.exp(-self.c / x) * (x)**(self.alpha-2), 0, 1)[0] / self._const
        mu += self.mean()
        print(mu)
        return mu

    def mean(self):
        return self.alpha * self.c ** (self.alpha - 1)

    def laplace_transform(self, t: np.float64) -> np.float64:
        return np.exp((t + self.c)**self.alpha - (self.c)**self.alpha)

    def get_min_jump_size(self):
        return self._min_jump_cutoff
    
    def get_shifted_sigma(self):
        return np.sqrt(scipy.integrate.quad(lambda x: self._inner_nu(-x) * x * x, 0, self._min_jump_cutoff)[0])


class UntemperedTotallySkewedStableRandomVariable(TemperedTotallySkewedStableRandomVariable):
    def __init__(self, alpha: float, min_jump_cutoff: float = 2 ** (-5), max_jump_cutoff: float = 2 ** 12) -> None:
        super().__init__(alpha, 0, min_jump_cutoff, max_jump_cutoff)

    def get_mu(self):
        return 1/(self.alpha - 1) / self._const
    
    def get_shifted_sigma(self):
        return np.sqrt(self._min_jump_cutoff ** (2 - self.alpha) / (2 - self.alpha) / self._const)
    
    def get_min_jump_size(self):
        return self._min_jump_cutoff
    
    def get_nu_compensation(self):
        return (self._min_jump_cutoff ** (1 - self.alpha) - 1) / (1 - self.alpha) / self._const
