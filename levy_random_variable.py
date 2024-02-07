from typing import Any, Callable
import numpy as np
import scipy

from random_variable import RandomVariable

class SpectrallyNegativeLevyRandomVariable(RandomVariable):    
    # assuming the form (mu, sigma, nu)
    # where 
    # E exp(lambda \xi) 
    #  = exp(- lambda mu 
    #             + lambda^2 sigma^2/2 
    #             + \int_-infty^0 (e^{lambda x} - 1 - lambda x 1_{x>-1}) nu(dx))
    # and nu(dx) is finite!
    def __init__(self, mu: float, sigma: float, nu: Callable, nu_unwarranted: Callable=None, max_jump_cutoff:float=2**12) -> None:
        self.mu = mu
        self.sigma = sigma
        self.nu = nu
        self.nu_unwarranted = nu_unwarranted

        self._max_jump_cutoff = max_jump_cutoff
        self._calculate_integrals()
        
    def _calculate_integrals(self):
        self._I1 = scipy.integrate.quad(lambda x: x * self.nu(x), -1, 0)[0]
        self._I2 = scipy.integrate.quad(lambda x: x * x * self.nu(x), -1, 0)[0]

        self._II1 = scipy.integrate.quad(lambda x: x * self.nu(x), -self._max_jump_cutoff, -1)[0]
        self._II2 = scipy.integrate.quad(lambda x: x * x * self.nu(x), -self._max_jump_cutoff, -1)[0]
        self._II2 -= (scipy.integrate.quad(lambda x: x * self.nu(x), -self._max_jump_cutoff, -1))[0] **2

    # TODO: finish up
    def characteristic_function(self, t: np.complex64) -> np.complex64: 
        return None
        return np.exp(- (1j * self.mu * t + self.sigma**2 / 2 * t * t + ...))
    
    # TODO: finish up
    def laplace_transform(self, t: np.float64) -> np.float64:
        return None
    
    def pdf(self, x: np.float64) -> np.float64:
        return None

    def cdf(self, x: np.float64) -> np.float64:
        return None

    # TODO: test
    def mean(self) -> np.float64:
        return self.mu + self._II1

    # TODO: test 
    def variance(self) -> np.float64:
        return self.sigma*self.sigma + self._II2 + self._I2
        
    def max_abs_nu_on_interval(self, a: float, b: float):
        return -scipy.optimize.fmin(lambda x: -self.nu(x) if a < x < b else 0).fopt
    
    def get_min_jump_size(self):
        return 0
    
    # def get_shifted_sigma(self):
    #     return self.sigma

    def sample(self, N: int) -> np.ndarray[float]:
        # shifted_sigma = self.get_shifted_sigma()
        s = np.random.normal(self.mu, self.sigma, N)

        a, b = self.get_min_jump_size(), 1
        while a < self._max_jump_cutoff:
            if b > self._max_jump_cutoff:
                b = self._max_jump_cutoff

            # simulate Pois with intensity nu on [-b, -a]
            interval_max = self.max_abs_nu_on_interval(-b, -a) 
            P = np.random.poisson(interval_max * (b - a), N)
            
            for i in range(N):
                Ts = np.random.uniform(0, interval_max, P[i])
                js = np.random.uniform(-b, -a, P[i])
                if self.nu_unwarranted is not None:
                    rs = self.nu_unwarranted(js) > Ts
                else:
                    rs = np.array(np.map(self.nu, js)) > Ts
                js = js[rs]
                
                s[i] += np.sum(js)
                # compensated measure
                if b == 1:
                    s[i] -= self._I1
            a, b = b, b*2
        return s
        

class DecreasingDensitySpectrallyNegativeLevyRandomVariable(SpectrallyNegativeLevyRandomVariable):
    def max_abs_nu_on_interval(self, a: float, b: float):
        return self.nu(b)


class TemperedTotallySkewedStableRandomVariable(DecreasingDensitySpectrallyNegativeLevyRandomVariable):
    """
    no gaussian approx is made here. 
    """
    def __init__(self, alpha: float, c: float, min_jump_cutoff: float = 2**(-5), max_jump_cutoff: float = 2 ** 12) -> None:
        self.alpha = alpha
        self.c = c
        self._min_jump_cutoff = min_jump_cutoff

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

        self._II2 = scipy.integrate.quad(lambda x: x * x * self.nu(x), -self._max_jump_cutoff, -1)[0] / self._const
        self._II2 -= (scipy.integrate.quad(lambda x: x * self.nu(x), -self._max_jump_cutoff, -1))[0]**2 / self._const**2
