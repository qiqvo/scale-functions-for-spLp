import numpy as np
from numpy import ndarray

from random_process.spectrally_negative_levy_random_process import SpectrallyNegativeLevyRandomProcess
from random_process.tempered_random_process_factory import create_tempered
from scale_function.scale_function import ScaleFunction
from stick_breaking_representation.stick_breaking_representation_factory import StickBreakingRepresentationFactory


class SBScaleFunction(ScaleFunction):
    """
    Scale function via Stick breaking process. 
    """
    def __init__(self, q: float, process: SpectrallyNegativeLevyRandomProcess, 
                 stick_breaking_representation_factory: StickBreakingRepresentationFactory,
                 N: int, resample_at_init:bool=False) -> None:
        super().__init__(q, process)

        if (self.q > 0 or (self.m < 0 and self.q == 0)):
            self.original_process = self.process
            self.c = self.compute_c(self.q)
            self.process = create_tempered(self.process, self.c)
            self.m = self.process.mean(1, 0)

        self.stick_breaking_representation = stick_breaking_representation_factory.create(self.process) 
        self.N = N
        if resample_at_init:
            self.resample()
        else: 
            self.stick_breaking_samples = None

    def compute_c(self, q):
        return self.process.get_underlying_xi_for_time(1).phi(q)
    
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
        return p / self.m

    def profile(self) -> ndarray:
        if self.stick_breaking_samples is None:
            self.resample()

        ps = [0]
        for i in range(self.N):
            xis = self.stick_breaking_samples[i][1]
            x_i = np.sum(xis, where=xis < 0)
            ps.append(x_i)
        ps = np.sort(-np.array(ps))
        values = np.arange(self.N + 1) / self.N
        return ps, values