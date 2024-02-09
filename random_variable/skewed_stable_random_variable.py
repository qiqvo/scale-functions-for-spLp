import numpy as np

from settings import seed

from random_variable.random_variable import RandomVariable
from random_variable.tempered_stable_random_variable import UntemperedTotallySkewedStableRandomVariable

class SkewedStableRandomVariable(RandomVariable):
    def __init__(self, alpha: float, beta: float, multiplier:float=1) -> None:
        self.rng = np.random.default_rng(seed=seed)
        
        self._C = multiplier
        self.alpha = alpha
        self.beta = beta
        self._k = 1 - abs(1 - self.alpha)

    def characteristic_function(self, t: np.complex64) -> np.complex64:
        return np.exp(-self._C  * np.abs(t)**self.alpha * np.exp(- 1j * np.pi/2 * self.beta * self._k * np.sign(t)))
    
    def laplace_transform(self, t: np.float64) -> np.float64:
        return None
    
    def pdf(self, x: np.float64) -> np.float64:
        return None

    def cdf(self, x: np.float64) -> np.float64:
        return None

    def mean(self) -> np.float64:
        if self.alpha < 1:
            return np.infty
        if 1 < self.alpha < 2:
            return 0

    def variance(self) -> np.float64:
        if self.alpha < 2:
            return np.infty
        
    def sample(self, N: int) -> np.ndarray[float]:
        w = -np.log(self.rng.uniform(0, 1, N))
        Phi = self.rng.uniform(-1, 1, N) * np.pi / 2
        Phi0 = - np.pi / 2 * self.beta * self._k / self.alpha
        dPhi = Phi - Phi0

        a = np.cos(Phi - self.alpha * dPhi)
        s = np.sin(self.alpha * dPhi) / np.power(np.cos(Phi), (1/self.alpha))
        s *= np.power(a / w, (1 - self.alpha) / self.alpha)
        s *= (self._C)**(1/self.alpha)
        return s
    

class TotallySkewedStableRandomVariable(SkewedStableRandomVariable, UntemperedTotallySkewedStableRandomVariable):
    def __init__(self, alpha: float, multiplier:float=1) -> None:
        super().__init__(alpha, -1 if alpha < 1 else 1, multiplier)
        super(SkewedStableRandomVariable, self).__init__(alpha, multiplier)

    def laplace_transform(self, t: np.float64) -> np.float64:
        return np.exp(self._C * np.sign(self.alpha - 1) * (t)**self.alpha)
    

class SymmetricStableRandomVariable(SkewedStableRandomVariable):
    def __init__(self, alpha: float, multiplier:float = 1) -> None:
        super().__init__(alpha, 0, multiplier)
