from matplotlib import pyplot as plt
import numpy as np
from random_process.skewed_stable_random_process import TotallySkewedStableRandomProcess

from scale_function.sb_scale_function import SBScaleFunction
from scale_function.skewed_stable_scale_function import TotallySkewedStableScaleFunction
from stick_breaking_representation.fixed_interval_stick_breaking_representation import ExpIntervalStickBreakingRepresentation, FixedIntervalStickBreakingRepresentation
from stick_breaking_representation.inf_interval_stick_breaking_representation import InfIntervalStickBreakingRepresentation
from stick_breaking_representation.stick_breaking_representation_factory import StickBreakingRepresentationFactory


def main():
    alpha = 1.6
    q = 0.2
    T = 1e5
    epsilon = 1/T
    n_sticks = 30
    N = 1000

    X = TotallySkewedStableRandomProcess(alpha)
    W = TotallySkewedStableScaleFunction(q, X)

    P1 = StickBreakingRepresentationFactory(InfIntervalStickBreakingRepresentation, epsilon=epsilon)
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
    plt.plot(xs, ws, label='W')
    plt.plot(x1s, v1s, label='sampled W, inf interval')
    plt.plot(x2s, v2s, label='sampled W, fixed interval')
    plt.plot(x3s, v3s, label='sampled W, exp interval')
    plt.title('Comparison of sampled W with the real W')

    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
