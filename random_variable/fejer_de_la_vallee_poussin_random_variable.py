import numpy as np

from random_variable.random_variable import RandomVariable

# Fejer-de la Vallee Poussin
class FejerDeLaValleePoussinRandomVariable(RandomVariable):
    def __init__(self) -> None:
        return None

    def characteristic_function(self, t: np.complex64) -> np.complex64:
        return np.max(1 - np.abs(t), 0)

    def laplace_transform(self, t: np.float64) -> np.float64:
        return np.real(self.characteristic_function(t))

    def pdf(self, x: np.float64) -> np.float64:
        return 1 / 2 / np.pi * (np.sin(x / 2) / (x/2))**2

    def cdf(self, x: np.float64) -> np.float64:
        return None
    
    #TODO: check
    def mean(self) -> np.float64:
        return 1

    #TODO: check
    def variance(self) -> np.float64:
        return 1

    def sample(self, N: int) -> np.ndarray[float]:
        s = []
        i = 0
        while i < N:
            u = np.random.uniform(-1, 1)
            v = np.random.uniform(-1, 1)
            if u < 0:
                u, v = - u* v* v, 1/v
            if u < np.sin(1/v)**2:
                s.append(v)
                i += 1
        return 2 / np.array(s)