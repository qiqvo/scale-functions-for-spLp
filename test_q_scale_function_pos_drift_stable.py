from matplotlib import pyplot as plt
import numpy as np

from random_process.drift_random_process import DriftRandomProcess
from random_process.skewed_stable_random_process import TotallySkewedStableRandomProcess
from scale_function.pos_drift_stable_scale_function import PosDriftTotallySkewedStableScaleFunction
from scale_function.sb_scale_function import SBScaleFunction
from stick_breaking_representation.fixed_interval_stick_breaking_representation import ExpIntervalStickBreakingRepresentation, FixedIntervalStickBreakingRepresentation
from stick_breaking_representation.inf_interval_stick_breaking_representation import InfIntervalStickBreakingRepresentation
from stick_breaking_representation.stick_breaking_representation_factory import StickBreakingRepresentationFactory


def main():
    alpha = 1.7
    drift = 0.1
    T = 1e5
    epsilon = 1e-2
    n_sticks = int(np.floor(np.log(T / epsilon) / np.log(2)))
    N = 1000

    Y = TotallySkewedStableRandomProcess(alpha)
    X = DriftRandomProcess(drift, Y)
    P2 = StickBreakingRepresentationFactory(FixedIntervalStickBreakingRepresentation, T=T, n_sticks=n_sticks)

    W = PosDriftTotallySkewedStableScaleFunction(0, X)

    a, b = 0, 1
    xs, ws = W.profile(a, b)

    plt.plot(xs, ws, label='W')
    for q in [0, 1e-10, 1e-7, 1e-4, 1e-2, 0.05, 0.1, 0.15]:
        V2 = SBScaleFunction(q, X, P2, N)
        xs, vs = V2.profile(a, b)
        print('q:', q)
        print('xs')
        print(xs)
        print('vs')
        print(vs)
        plt.plot(xs, vs, label=f'sampled W^(q), q={'%.2E' % q}')
    plt.title('Comparison for different q, Pos Drift Stable')

    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
