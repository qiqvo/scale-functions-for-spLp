from abc import abstractmethod
import numpy as np
from numpy import ndarray
import scipy

from random_process.spectrally_negative_levy_random_process import SpectrallyNegativeLevyRandomProcess
from random_process.tempered_random_process_factory import create_tempered
from scale_function.scale_function import ScaleFunction


class TemperedScaleFunction(ScaleFunction):
    def __init__(self, q:float, process: SpectrallyNegativeLevyRandomProcess) -> None:
        super().__init__(q, process)
        if (self.q > 0 or (self.m < 0 and self.q == 0)):
            self.original_process = self.process
            self.c = self.compute_c(self.q)
            self.process = create_tempered(self.c, self.process)
            self.m = self.process.mean(1, 0)
        else:
            self.c = None
        
    def compute_c(self, q):
        return self.process.get_underlying_xi_for_time(1).phi(q)
    
    @abstractmethod
    def _inner_profile(self, a: float, b: float) -> tuple[ndarray, ndarray]:
        return None
    
    def profile(self, a: float, b: float) -> tuple[ndarray, ndarray]:
        xs, vs = self._inner_profile(a, b)

        if self.c is not None: 
            vs *= np.exp(self.c * xs)

        return xs, vs