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
                 N: int, resample_at_init:bool=False) -> None:
        super().__init__(q, process)
        self.stick_breaking_representation = stick_breaking_representation_factory.create(self.process) 
        self.N = N
        if resample_at_init:
            self.resample()
        else: 
            self.stick_breaking_samples = None

    def resample(self):
        self.stick_breaking_samples = self.stick_breaking_representation.sample(self.N)

    def value(self, x: float):
        if self.stick_breaking_samples is None:
            self.resample()

        ps = []
        for i in range(self.N):
            xis = self.stick_breaking_samples[i][1]
            x_i = np.sum(xis, where=xis < 0)
            ps.append(x_i > -x)
        
        p = np.sum(ps) / self.N
        if self.c is not None: 
            p *= np.exp(self.c * x)
        return p / self.m

    def _inner_profile(self, a: float, b: float) -> tuple[np.ndarray, np.ndarray]:
        if self.stick_breaking_samples is None:
            self.resample()

        xis = self.stick_breaking_samples[:,1]
        ps = np.sum(xis, where=xis<0, axis=1)
        ps = np.sort(-np.array(ps))
        rs = (ps <= b) & (ps >= a)
        values = np.arange(1, self.N + 1) / self.N / self.m

        ps, values = ps[rs], values[rs]
        
        return ps, values