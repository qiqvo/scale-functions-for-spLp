from typing import Any, Callable
import numpy as np
import scipy

from random_variable.spectrally_negative_levy_random_variable import SpectrallyNegativeLevyRandomVariable


class TemperedSpectrallyNegativeLevyRandomVariable(SpectrallyNegativeLevyRandomVariable):
    def __init__(self, c: float, mu: float, sigma: float, nu: Callable[..., Any], nu_unwarranted: Callable[..., Any] = None, max_jump_cutoff: float = 2 ** 12) -> None:
        assert sigma == 0

        self.c = c
        tempered_mu = self.get_tempered_mu(mu, nu)
        if nu_unwarranted is not None:
            tempered_nu_unwarranted = lambda x: np.exp(c * x) * nu_unwarranted(x)
        else:
            tempered_nu_unwarranted = None
        tempered_nu = lambda x: np.exp(c * x) * nu(x)
        super().__init__(tempered_mu, sigma, tempered_nu, tempered_nu_unwarranted, max_jump_cutoff=max_jump_cutoff)

    def get_tempered_mu(self, mu: float, nu: Callable[..., Any]):
        # see for example Ch 9.5 of Financial Modelling with Jump Processes By Rama Cont, Peter Tankov
        tempered_mu = mu + scipy.integrate.quad(lambda x: x * (np.exp(self.c * x) - 1) * nu(x), -1, 0)[0] 
        return tempered_mu

    def mean(self):
        return self.alpha * self.c ** (self.alpha - 1)
    

def create_from_snl_rv(x: SpectrallyNegativeLevyRandomVariable, c: float) -> TemperedSpectrallyNegativeLevyRandomVariable:
    return TemperedSpectrallyNegativeLevyRandomVariable(c, x.mu, x.sigma, x.nu, x.nu_unwarranted, x._max_jump_cutoff)