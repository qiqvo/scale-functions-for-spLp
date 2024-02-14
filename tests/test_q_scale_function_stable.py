from matplotlib import pyplot as plt
import numpy as np
from random_process.skewed_stable_random_process import TotallySkewedStableRandomProcess

from scale_function.sb_scale_function import SBScaleFunction
from scale_function.skewed_stable_scale_function import TotallySkewedStableScaleFunction
from stick_breaking_representation.fixed_interval_stick_breaking_representation import ExpIntervalStickBreakingRepresentation, FixedIntervalStickBreakingRepresentation
from stick_breaking_representation.inf_interval_stick_breaking_representation import InfIntervalStickBreakingRepresentation
from stick_breaking_representation.stick_breaking_representation_factory import StickBreakingRepresentationFactory


def test_q_scale_function_stable():
    alpha = 1.7
    T = 1e6
    epsilon = 1e-4
    n_sticks = int(np.floor(np.log(T / epsilon) / np.log(2)))
    N = 500
    plot_results = True

    X = TotallySkewedStableRandomProcess(alpha)

    P1 = StickBreakingRepresentationFactory(InfIntervalStickBreakingRepresentation, epsilon=epsilon, T=T)
    P2 = StickBreakingRepresentationFactory(FixedIntervalStickBreakingRepresentation, T=T, n_sticks=n_sticks)
    P3 = StickBreakingRepresentationFactory(ExpIntervalStickBreakingRepresentation, theta=1/T, n_sticks=n_sticks)
    # P4 = StickBreakingRepresentationFactory(FixedIntervalStickBreakingRepresentation, T=T, n_sticks=n_sticks)


    # assert xs[12] == 1
    # assert x1s[12] == 1
    # assert x2s[12] == 1
    # assert x3s[12] == 1
    # assert ws[12] == 1
    # assert v1s[12] == 1
    # assert v2s[12] == 1
    # assert v3s[12] == 1
    
    a, b = 0, 1
    qs = [1e-3, 0.1, 0.5, 1]
    for q in qs:
        W = TotallySkewedStableScaleFunction(q, X)
        V1 = SBScaleFunction(q, X, P1, N)
        V2 = SBScaleFunction(q, X, P2, N)
        V3 = SBScaleFunction(q, X, P3, N)

        xs, ws = W.profile(a, b)
        x1s, v1s = V1.profile(a, b)
        x2s, v2s = V2.profile(a, b)
        x3s, v3s = V3.profile(a, b)

        if plot_results:
            plt.plot(xs, ws, label=f'W^q')
            # plt.plot(x1s, v1s, label='sampled W, inf interval')
            plt.plot(x2s, v2s, label=f'sampled W^q')
            # plt.plot(x3s, v3s, label='sampled W, exp interval')

            plt.title(f'Comparison of sampled W with the real W, q = {'%.2E' % q}')
            plt.legend()
            plt.show()
