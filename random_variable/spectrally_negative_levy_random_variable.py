from typing import Any, Callable
import numpy as np
import scipy

from settings import seed

from random_variable.random_variable import RandomVariable

class SpectrallyNegativeLevyRandomVariable(RandomVariable):    
    # assuming the form (mu, sigma, nu)
    # where 
    # E exp(lambda \xi) 
    #  = exp(- lambda mu 
    #             + lambda^2 sigma^2/2 
    #             + \int_-infty^0 (e^{lambda x} - 1 - lambda x 1_{x>-1}) nu(dx))
    # and nu(dx) is finite!
    def __init__(self, mu: float, sigma: float, nu: Callable, 
                 nu_unwarranted: Callable=None, 
                 multiplier: float=1, 
                 max_jump_cutoff:float=2**12) -> None:
        self.rng = np.random.default_rng(seed=seed)

        self._C = multiplier
        self.mu = mu
        self.sigma = sigma
        self.nu = nu
        self.nu_unwarranted = nu_unwarranted 

        self._max_jump_cutoff = max_jump_cutoff
        self._max_intensity_over_ab = 1000
        self.nu_compensation = self.get_nu_compensation()
        
    def get_nu_compensation(self, a:float=-1, b:float=0):
        nu_compensation = scipy.integrate.quad(lambda x: x * self.nu(x), a, b)[0]
        return nu_compensation

    # TODO: finish up
    def characteristic_function(self, t: np.complex64) -> np.complex64:
        return None
    
    # TODO: finish up
    def laplace_transform(self, t: np.float64) -> np.float64:
        return np.exp(self.psi(t))
    
    def psi(self, t: np.float64) -> np.float64:
        res = - t * self.mu + t*t*self.sigma*self.sigma / 2
        res += scipy.integrate.quad(lambda x: (np.exp(t*x) - 1) * self.nu(x), -np.infty, -1)[0]
        res += scipy.integrate.quad(lambda x: (np.exp(t*x) - 1 - t*x) * self.nu(x), -1, 0)[0]
        return res * self._C
    
    def phi(self, q:np.float64, a:np.float64=0, b:np.float64=2**10) -> np.float64:
        # if find_min:
        #     a = scipy.optimize.minimize_scalar(self.psi)
        if q == 0 and self.mean() < 0:
            a = 1
            while self.psi(a) > 0:
                b = a
                a = a // 2
        else:
            a = 0

        res = scipy.optimize.brentq(lambda t: self.psi(t) - q, a, b)
        return res
    
    def pdf(self, x: np.float64) -> np.float64:
        return None

    def cdf(self, x: np.float64) -> np.float64:
        return None

    # TODO: 
    def mean(self) -> np.float64:
        return None # self.mu + self._II1

    # TODO: 
    def variance(self) -> np.float64:
        return None #self.sigma*self.sigma + self._II2 + self._I2
        
    def max_abs_nu_on_interval(self, a: float, b: float):
        return -scipy.optimize.fmin(lambda x: -self.nu(x) if a < x < b else 0).fopt
    
    def get_min_jump_size(self):
        return 0
    
    def get_shifted_sigma(self):
        return self.sigma

    def sample(self, N: int) -> np.ndarray[float]:
        shifted_sigma = self.get_shifted_sigma()
        s = self.rng.normal(self.mu * self._C, shifted_sigma * np.sqrt(self._C), N)

        a, b = self.get_min_jump_size(), 1
        while a < self._max_jump_cutoff:
            if b > self._max_jump_cutoff:
                b = self._max_jump_cutoff

            # simulate Pois with intensity nu on [-b, -a]
            # adjusting the level b so that Pois intensity is smaller than self._max_intensity_over_ab 
            interval_max = self.max_abs_nu_on_interval(-b, -a)
            nu_ab = interval_max * (b - a) * self._C
            while nu_ab > self._max_intensity_over_ab:
                b = (b + a) / 2
                interval_max = self.max_abs_nu_on_interval(-b, -a)
                nu_ab = interval_max * (b - a) * self._C
            P = self.rng.poisson(nu_ab, N)
            
            for i in range(N):
                Ts = self.rng.uniform(0, interval_max, P[i])
                js = self.rng.uniform(-b, -a, P[i])
                if self.nu_unwarranted is not None:
                    rs = self.nu_unwarranted(js) > Ts
                else:
                    rs = np.array(np.map(self.nu, js)) > Ts
                js = js[rs]
                
                s[i] += np.sum(js)
                # compensated measure
                if a == self.get_min_jump_size():
                    s[i] -= self.nu_compensation * self._C 
            a, b = b, b*2
        return s
        

class DecreasingDensitySpectrallyNegativeLevyRandomVariable(SpectrallyNegativeLevyRandomVariable):
    def max_abs_nu_on_interval(self, a: float, b: float):
        return self.nu(b)