import numpy as np
from positive_stable_random_variable import PositiveStableRandomVariable

from random_variable import RandomVariable


class SymmetricStableRandomVariable(RandomVariable):
    # alpha < 1
    def __init__(self, alpha: float) -> None:
        self.alpha = alpha
        self.positive_rv = PositiveStableRandomVariable(alpha)

    def characteristic_function(self, t: np.complex64) -> np.complex64:
        return np.exp(- self.d * np.power(t, self.alpha))
    
    def laplace_transform(self, t: np.float64) -> np.float64:
        return None

    def pdf(self, x: np.float64) -> np.float64:
        return None

    def cdf(self, x: np.float64) -> np.float64:
        return None

    def mean(self) -> np.float64:
        if self.alpha < 1:
            return None

    def variance(self) -> np.float64:
        if self.alpha < 2:
            return np.infty

    def sample(self, N: int) -> np.ndarray[float]:
        s = self.positive_rv.sample(N) * (np.random.randint(2, size=N) * 2 - 1)
    