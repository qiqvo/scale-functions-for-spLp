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
                 char_multiplier: float=1, 
                 amplitude_multiplier: float=1,
                 max_jump_cutoff:float=2**12) -> None:
        self.rng = np.random.default_rng(seed=seed)

        self._char_multiplier = char_multiplier
        self.amplitude_multiplier = amplitude_multiplier
        self.mu = mu
        self.sigma = sigma
        self.nu = nu
        self.nu_unwarranted = nu_unwarranted 

        self.max_jump_cutoff = max_jump_cutoff
        self._max_intensity_over_ab = 1000
        self._min_intensity_over_ab = 0.1
        self.nu_compensation = self.get_nu_compensation()
        
    _precomputed_nu_compensation = dict()
    def _get_key_for_precomputed_nu_compensation(self):
        # TODO: nu(-1) is not a characteristic....
        return (self.mu, self.sigma, self.nu(-1))

    def get_nu_compensation(self, a:float=-1, b:float=0):
        key = self._get_key_for_precomputed_nu_compensation()
        if key in self._precomputed_nu_compensation:
            return self._precomputed_nu_compensation[key]
        
        nu_compensation = scipy.integrate.quad(lambda x: x * self.nu(x), a, b)[0]
        self._precomputed_nu_compensation[key] = nu_compensation
        return nu_compensation

    # TODO: finish up
    def characteristic_function(self, t: np.complex64) -> np.complex64:
        return None
    
    # TODO: finish up
    def laplace_transform(self, t: np.float64) -> np.float64:
        return np.exp(self.psi(t))
    
    _precomputed_psi = dict()
    def _get_key_for_precomputed_psi(self, t):
        # TODO: nu(-1) is not a characteristic....
        return (self.mu, self.sigma, self.nu(-1), t)

    def psi(self, t: np.float64) -> np.float64:
        t *= self.amplitude_multiplier
        key = self._get_key_for_precomputed_psi(t)
        if key in self._precomputed_psi:
            res = self._precomputed_psi[key]
        else:
            res = - t * self.mu + t*t*self.sigma*self.sigma / 2
            res += scipy.integrate.quad(lambda x: (np.exp(t*x) - 1) * self.nu(x), -np.infty, -1)[0]
            res += scipy.integrate.quad(lambda x: (np.exp(t*x) - 1 - t*x) * self.nu(x), -1, 0)[0]
            self._precomputed_psi[key] = res
        return res * self._char_multiplier
    
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
        s = self.rng.normal(self.mu * self._char_multiplier, shifted_sigma * np.sqrt(self._char_multiplier), N)

        a, b = self.get_min_jump_size(), 1
        while a < self.max_jump_cutoff:
            if b > self.max_jump_cutoff:
                b = self.max_jump_cutoff

            # simulate Pois with intensity nu on [-b, -a]
            # adjusting the level b so that Pois intensity is smaller than self._max_intensity_over_ab 
            interval_max = self.max_abs_nu_on_interval(-b, -a)
            nu_ab = interval_max * (b - a) * self._char_multiplier
            while nu_ab > self._max_intensity_over_ab:
                b = (b + a) / 2
                interval_max = self.max_abs_nu_on_interval(-b, -a)
                nu_ab = interval_max * (b - a) * self._char_multiplier

            while nu_ab < self._min_intensity_over_ab:
                b *= 2
                interval_max = self.max_abs_nu_on_interval(-b, -a)
                nu_ab = interval_max * (b - a) * self._char_multiplier
                if b > self.max_jump_cutoff:
                    b = self.max_jump_cutoff
                    break
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
                    s[i] -= self.nu_compensation * self._char_multiplier 
            a, b = b, b*2
        return s * self.amplitude_multiplier
        

class DecreasingDensitySpectrallyNegativeLevyRandomVariable(SpectrallyNegativeLevyRandomVariable):
    def max_abs_nu_on_interval(self, a: float, b: float):
        return self.nu(b)