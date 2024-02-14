from matplotlib import pyplot as plt
import numpy as np
from random_process.skewed_stable_random_process import TotallySkewedStableRandomProcess
from random_variable.skewed_stable_random_variable import TotallySkewedStableRandomVariable

from scale_function.sb_scale_function import SBScaleFunction
from scale_function.skewed_stable_scale_function import TotallySkewedStableScaleFunction
from stick_breaking_representation.fixed_interval_stick_breaking_representation import ExpIntervalStickBreakingRepresentation, FixedIntervalStickBreakingRepresentation
from stick_breaking_representation.inf_interval_stick_breaking_representation import InfIntervalStickBreakingRepresentation
from stick_breaking_representation.stick_breaking_representation_factory import StickBreakingRepresentationFactory


def test_approx_to_integrals():
    alpha = 1.7
    q = 0.1
    T = 1e6
    epsilon = 1e-2
    n_sticks = np.floor(np.log(T / epsilon) / np.log(2))
    N = 50

    X = TotallySkewedStableRandomProcess(alpha)
    W = TotallySkewedStableScaleFunction(q, X)

    P1 = StickBreakingRepresentationFactory(InfIntervalStickBreakingRepresentation, epsilon=epsilon, T=T)
    P2 = StickBreakingRepresentationFactory(FixedIntervalStickBreakingRepresentation, T=T, n_sticks=n_sticks)
    P3 = StickBreakingRepresentationFactory(ExpIntervalStickBreakingRepresentation, theta=1/T, n_sticks=n_sticks)
    # P4 = StickBreakingRepresentationFactory(FixedIntervalStickBreakingRepresentation, T=T, n_sticks=n_sticks)

    V1 = SBScaleFunction(q, X, P1, N)
    V2 = SBScaleFunction(q, X, P2, N)
    V3 = SBScaleFunction(q, X, P3, N)

    a, b = 0, 1
    xs, ws = W.profile(a, b)
    x1s, v1s = V1.profile(a, b)
    x2s, v2s = V2.profile(a, b)
    x3s, v3s = V3.profile(a, b)

    for d in [TotallySkewedStableRandomVariable._precomputed_nu_compensation,
              TotallySkewedStableRandomVariable._precomputed_shifted_sigma,
              TotallySkewedStableRandomVariable._precomputed_tempered_mu]:
        d_items = sorted(d.items())
        d_keys, d_values = zip(*d_items)
        d_keys = np.array(d_keys)[:, 0]
        plt.plot(d_keys, d_values)
        plt.xlabel('c')
        plt.show()
