from abc import abstractmethod
import numpy as np

from random_process.spectrally_negative_levy_random_process import SpectrallyNegativeLevyRandomProcess
from random_process.tempered_random_process_factory import create_tempered
from scale_function.scale_function import ScaleFunction
from scale_function.tempered_scale_function import TemperedScaleFunction
from stick_breaking_representation.stick_breaking_representation_factory import StickBreakingRepresentationFactory


class SBScaleFunction(TemperedScaleFunction):
    """
    Scale function via Stick breaking process. 
    """
    def __init__(self, q: float, process: SpectrallyNegativeLevyRandomProcess, 
                 stick_breaking_representation_factory: StickBreakingRepresentationFactory,
                 N: int) -> None:
        super().__init__(q, process)
        self.stick_breaking_representation_factory = stick_breaking_representation_factory
        self.N = N

    def get_stick_breaking_representation(self):
        return self.stick_breaking_representation_factory.create(self.process) 

    def value(self, x: float):
        return None

    def _inner_profile(self, a: float, b: float) -> tuple[np.ndarray, np.ndarray]:
        ps = self._compute_ps()
        ps = np.sort(-ps)

        k = 0
        if self.process.is_infinite_activity():
            while ps[k] == 0:
                k += 1
            if k > 0:
                ps = ps[k-1:]
                k -= 1

        N = self.N - k
        values = np.arange(1, N + 1) / N / self.m

        rs = (ps <= b) & (ps >= a)
        ps, values = ps[rs], values[rs]

        return ps, values
    
    @abstractmethod
    def _compute_ps(self):
        return None
    

class SimpleSBScaleFunction(SBScaleFunction):
    def __init__(self, q: float, process: SpectrallyNegativeLevyRandomProcess, 
                 stick_breaking_representation_factory: StickBreakingRepresentationFactory, 
                 N: int, resample_at_init:bool=False) -> None:
        super().__init__(q, process, stick_breaking_representation_factory, N)
        self.stick_breaking_samples = None
        if resample_at_init:
            self.get_samples()

    def get_samples(self):
        if self.stick_breaking_samples is None:
            self.stick_breaking_samples = self.get_stick_breaking_representation().sample(self.N)
        return self.stick_breaking_samples

    def value(self, x: float):
        stick_breaking_samples = self.get_samples()

        ps = []
        for i in range(self.N):
            xis = stick_breaking_samples[i][1]
            x_i = np.sum(xis, where=xis < 0)
            ps.append(x_i > -x)
        
        p = np.sum(ps) / self.N
        if self.c is not None: 
            p *= np.exp(self.c * x)
        return p / self.m
    

    def _compute_ps(self) -> tuple[np.ndarray, np.ndarray]:
        stick_breaking_samples = self.get_samples()
        xis = stick_breaking_samples[:,1]
        ps = np.sum(xis, where=xis<0, axis=1)
        
        return ps