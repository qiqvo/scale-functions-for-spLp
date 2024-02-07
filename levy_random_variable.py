from typing import Any, Callable
import numpy as np
import scipy

from random_variable import RandomVariable

class SpectrallyNegativeLevyRandomVariable(RandomVariable):
    """
    assuming the form (mu, sigma, nu)
    where 
    E exp(lambda \xi) 
     = exp(- lambda mu 
                + lambda^2 sigma^2/2 
                + \int_-infty^0 (e^{lambda x} - 1 - lambda x 1_{x>-1}) nu(dx))

    and nu(dx) is finite!
    """
    def __init__(self, mu: float, sigma: float, nu: Callable, max_jump_cutoff:float=2**12) -> None:
        self.mu = mu
        self.sigma = sigma
        self.nu = nu

        self._max_jump_cutoff = max_jump_cutoff
        
        self._I1 = scipy.integrate.quad(lambda x: x * self.nu(x), -1, 0)
        self._I2 = scipy.integrate.quad(lambda x: x * x * self.nu(x), -1, 0)

        self._II2 = scipy.integrate.quad(lambda x: x * x * self.nu(x), -self._max_jump_cutoff, -1)
        self._II2 -= (scipy.integrate.quad(lambda x: x * self.nu(x), -self._max_jump_cutoff, -1))**2

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
        return self.mu + self._I

    # TODO: test 
    def variance(self) -> np.float64:
        return self.sigma*self.sigma + self._II2 + self._I2
        
    def max_nu_on_interval(self, a: float, b: float):
        return -scipy.optimize.fmin(lambda x: -self.nu(x) if a < x < b else 0).fopt

    def sample(self, N: int) -> np.ndarray[float]:
        s = np.random.normal(self.mu, self.sigma, N)

        a, b = 0, 1
        while a < self._max_jump_cutoff:
            # simulate Pois with intensity nu on [-b, -a]
            interval_max = self.max_nu_on_interval(-b, -a)
            P = np.random.poisson(interval_max, N)
            
            for i in range(N):
                Ts = np.random.uniform(0, interval_max, P[i])
                js = np.random.uniform(-b, -a, P[i])
                js = js[self.nu(js) > Ts]
            
                # compensated measure
                if a == 0 and b == 1:
                    js -= self._I1
                
                s[i] += np.sum(js)
            a, b = b, b*2
        return s
        

class DecreasingDensitySpectrallyNegativeLevyRandomVariable(SpectrallyNegativeLevyRandomVariable):
    def max_nu_on_interval(self, a: float, b: float):
        return self.nu(a)


class TemperedTotallySkewedStableRandomVariable(DecreasingDensitySpectrallyNegativeLevyRandomVariable):
    """
    no gaussian approx is made here. 
    """
    def __init__(self, alpha: float, c: float, min_jump_cutoff: float = 2**(-10), max_jump_cutoff: float = 2 ** 12) -> None:
        self.alpha = alpha
        self.c = c
        nu = lambda x: np.exp(c * x) / (x)**(1 + alpha) if x < -min_jump_cutoff else 0
        super().__init__(0, 0, lambda x: np.exp(c * x) * nu(x), max_jump_cutoff)

    def laplace_transform(self, t: np.float64) -> np.float64:
        return np.exp((t + self.c)**self.alpha - (self.c)**self.alpha)
    