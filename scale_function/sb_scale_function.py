import numpy as np
from numpy import ndarray

from random_process.spectrally_negative_levy_random_process import SpectrallyNegativeLevyRandomProcess
from random_process.tempered_spectrally_negative_levy_random_process import create_from_snl_process
from scale_function.scale_function import ScaleFunction
from stick_breaking_representation.stick_breaking_representation_factory import StickBreakingRepresentationFactory


class SBScaleFunction(ScaleFunction):
    """
    Scale function via Stick breaking process. 
    """
    def __init__(self, q: float, process: SpectrallyNegativeLevyRandomProcess, 
                 stick_breaking_representation_factory: StickBreakingRepresentationFactory,
                 N: int) -> None:
        super().__init__(q, process)

        if (self.q > 0 or (self.m < 0 and self.q == 0)):
            self.original_process = self.process
            self.c = self.get_c(self.q)
            self.process = create_from_snl_process(self.process, self.c)

        self.stick_breaking_representation = stick_breaking_representation_factory.create(self.process) 
        self.N = N
        self.resample()

    def get_c(self, q):
        return self.process.get_underlying_xi_for_time(1).phi(q)
    
    def resample(self):
        self.stick_breaking_samples = self.stick_breaking_process.sample(self.N)

    def value(self, x: float):
        ps = []
        for i in range(self.N):
            xis = self.stick_breaking_samples[i][1]
            x_i = np.sum(xis, where=xis < 0)
            ps.append(x_i > -x)
        
        p = np.sum(ps) / self.N
        return p / self.m

    def profile(self, range_x: np.array) -> ndarray:
        ps = []
        for i in range(self.N):
            xis = self.stick_breaking_samples[i][1]
            x_i = np.sum(xis, where=xis < 0)
            ps.append(x_i)
        ps = np.array(ps)

        p = []
        for x in range_x:
            p.append(np.sum(ps>-x) / self.N / self.m)
        return np.array(p)