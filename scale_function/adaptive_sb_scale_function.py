import numpy as np

from random_process.spectrally_negative_levy_random_process import SpectrallyNegativeLevyRandomProcess
from scale_function.sb_scale_function import SBScaleFunction
from stick_breaking_representation.stick_breaking_representation_factory import StickBreakingRepresentationFactory

class AdaptiveSBScaleFunction(SBScaleFunction):
    """
    Scale function via Stick breaking process. 
    Adaptive by the means of using independent sampling of Pois PP in the SB. 
    """
    def __init__(self, q: float, process: SpectrallyNegativeLevyRandomProcess, 
                 stick_breaking_representation_factory: StickBreakingRepresentationFactory,
                 N: int,
                 eps_conv:float=1/2**8, max_iter:int=5,
                 initial_horizon:float=2**10) -> None:
        super().__init__(q, process, stick_breaking_representation_factory, N)
        self._eps_conv = eps_conv
        self._max_iter = max_iter
        self._initial_horizon = initial_horizon

    def get_samples(self, s: float, t: float):
        return self.get_stick_breaking_representation().sample(self.N, s, t)

    def value(self, x: float):
        # if self.stick_breaking_samples is None:
        #     self.resample()

        # ps = []
        # for i in range(self.N):
        #     xis = self.stick_breaking_samples[i][1]
        #     x_i = np.sum(xis, where=xis < 0)
        #     ps.append(x_i > -x)
        
        # p = np.sum(ps) / self.N
        # if self.c is not None: 
        #     p *= np.exp(self.c * x)
        # return p / self.m
        return None

    def _compute_ps(self):
        stick_breaking_samples = self.get_samples(1/self._initial_horizon, self._initial_horizon)
        xis = stick_breaking_samples[:,1]
        ps = np.sum(xis, where=xis<0, axis=1)
        B, A = 1/self._initial_horizon, self._initial_horizon

        converged = False
        iter = 0
        while not converged and iter < self._max_iter:
            stick_breaking_samples = self.get_samples(B/10, B)
            xis = stick_breaking_samples[:,1]
            ps_update = np.sum(xis, where=xis<0, axis=1)
            B /= 10

            stick_breaking_samples = self.get_samples(A, A*10)
            xis = stick_breaking_samples[:,1]
            ps_update += np.sum(xis, where=xis<0, axis=1)
            A *= 10

            converged = np.sum(-ps_update)/self.N < self._eps_conv
            iter += 1
            ps += ps_update

        self.converged = converged
        self.last_iteration = iter

        return ps