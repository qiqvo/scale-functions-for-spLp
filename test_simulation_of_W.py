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
    alpha = 1.6
    drift = 0.1
    q = 0
    a = 10

    Y = TotallySkewedStableRandomProcess(alpha)
    X = DriftRandomProcess(drift, Y)
    W = PosDriftTotallySkewedStableScaleFunction(q, X)

    R = range(100,1100,100)
    for j in range(2, 7):
        epsilon = 0.1**j
        T = 1/epsilon
        n_sticks = np.floor(np.log(T / epsilon) / np.log(2))

        P1 = StickBreakingRepresentationFactory(InfIntervalStickBreakingRepresentation, epsilon=epsilon)
        P2 = StickBreakingRepresentationFactory(FixedIntervalStickBreakingRepresentation, T=T, n_sticks=n_sticks)
        P3 = StickBreakingRepresentationFactory(ExpIntervalStickBreakingRepresentation, theta=epsilon, n_sticks=n_sticks)
        # P4 = StickBreakingRepresentationFactory(FixedIntervalStickBreakingRepresentation, T=T, n_sticks=n_sticks)

        w1,w2,w3 = [],[],[]
        for i in R:
            W1 = SBScaleFunction(q, X, P1, i)
            W2 = SBScaleFunction(q, X, P2, i)
            W3 = SBScaleFunction(q, X, P3, i)
            # np.random.seed(0)
            # w1.append(W1.value(a))
            w2.append(W2.value(a))
            w3.append(W3.value(a))
        # plt.plot(R, w1, label=f'eps={'%.2E' % epsilon}')
        plt.plot(R, w2, label=f'eps={'%.2E' % epsilon}')
        plt.plot(R, w3, label=f'eps={'%.2E' % epsilon}')

    v = W.value(a)
    print(v)
    plt.plot([R[0], R[-1]], [v, v], c='r')
    plt.legend()
    plt.show()
    # print(w1[-1])


if __name__ == '__main__':
    main()
