import time
from matplotlib import pyplot as plt
import numpy as np

from random_process.skewed_stable_random_process import TotallySkewedStableRandomProcess
from scale_function.mc_scale_function import MCScaleFunction
from scale_function.sb_scale_function import SBScaleFunction
from scale_function.skewed_stable_scale_function import TotallySkewedStableScaleFunction
from stick_breaking_representation.fixed_interval_stick_breaking_representation import FixedIntervalStickBreakingRepresentation
from stick_breaking_representation.inf_interval_stick_breaking_representation import InfIntervalStickBreakingRepresentation
from stick_breaking_representation.stick_breaking_representation_factory import StickBreakingRepresentationFactory


def test_comparison_sb_mc_scale_function():
    a, b = 0, 1
    alpha = 1.6
    q = 1
    # drift = 0.3

    Y = TotallySkewedStableRandomProcess(alpha)
    W = TotallySkewedStableScaleFunction(q, Y)
    # X = DriftRandomProcess(drift, Y)
    # W = PosDriftTotallySkewedStableScaleFunction(q, X)
    Wf = np.vectorize(W.value)

    N = 1000
    for k in [6, 9, 12]:
        T = 2**k
        epsilon = 2**(-8)
        n_sticks = int(np.floor(np.log(T / epsilon) / np.log(2)))
        P2 = StickBreakingRepresentationFactory(InfIntervalStickBreakingRepresentation, s=epsilon, t=T)
        # P2 = StickBreakingRepresentationFactory(FixedIntervalStickBreakingRepresentation, T=T, n_sticks=n_sticks)
        V1 = SBScaleFunction(q, Y, P2, N)
        xs, vs = V1.profile(a, b)
        diffs = np.abs(Wf(xs) - vs)
        # plt.plot(xs, vs, label=f'SB, k={k}')
        plt.plot(xs, diffs, label=f'SB, k={k}')

    for k in [10, 16, 20]:
        h = 2**(-k)
        upper_cutoff = 1/h
        V2 = MCScaleFunction(q, Y, h, upper_cutoff)
        xs, vs = V2.profile(a, b)
        diffs = np.abs(Wf(xs) - vs)
        # plt.plot(xs, vs, linestyle='dashed', label=f'MC, k={k}')
        plt.plot(xs, diffs, linestyle='dashed', label=f'MC, k={k}')

    plt.plot([a, b], [0, 0], c='r')
    plt.legend()
    plt.show()
