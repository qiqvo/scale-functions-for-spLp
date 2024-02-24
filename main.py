import time
from matplotlib import pyplot as plt
import numpy as np

from random_process.skewed_stable_random_process import TotallySkewedStableRandomProcess
from scale_function.adaptive_sb_scale_function import AdaptiveSBScaleFunction
from scale_function.sb_scale_function import SimpleSBScaleFunction
from scale_function.skewed_stable_scale_function import TotallySkewedStableScaleFunction
from stick_breaking_representation.inf_interval_stick_breaking_representation import InfIntervalStickBreakingRepresentation, PoissonSampledInfIntervalStickBreakingRepresentation
from stick_breaking_representation.stick_breaking_representation_factory import StickBreakingRepresentationFactory


def main():
    a, b = 0, 1
    alpha = 1.6
    q = 0.01
    # drift = 0.3

    Y = TotallySkewedStableRandomProcess(alpha)
    W = TotallySkewedStableScaleFunction(q, Y)
    # X = DriftRandomProcess(drift, Y)
    # W = PosDriftTotallySkewedStableScaleFunction(q, X)
    Wf = np.vectorize(W.value)
    # xs, ws = W.profile(a, b)
    # plt.plot(xs, ws, label=f'formula')

    N = 300
    # n_sticks = int(np.floor(np.log(T / epsilon) / np.log(2)))
    T = 2**10
    P1 = StickBreakingRepresentationFactory(InfIntervalStickBreakingRepresentation, s=1/T, t=T)
    # P2 = StickBreakingRepresentationFactory(FixedIntervalStickBreakingRepresentation, T=T, n_sticks=n_sticks)
    V1 = SimpleSBScaleFunction(q, Y, P1, N)
    xs, vs = V1.profile(a, b)
    diffs = np.abs(Wf(xs) - vs)
    # plt.plot(xs, vs, label=f'SB')
    plt.plot(xs, diffs, label=f'SB')

    P2 = StickBreakingRepresentationFactory(PoissonSampledInfIntervalStickBreakingRepresentation)
    V2 = AdaptiveSBScaleFunction(q, Y, P2, N, eps_conv=1/2**18, max_iter=10)
    xs, vs = V2.profile(a, b)
    diffs = np.abs(Wf(xs) - vs)
    print(V2.converged, V2.last_iteration)
    # plt.plot(xs, vs, label=f'adaptive SB')
    plt.plot(xs, diffs, label=f'adaptive SB')

    plt.plot([a, b], [0, 0], c='r')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()