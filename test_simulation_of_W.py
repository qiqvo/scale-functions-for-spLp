from matplotlib import pyplot as plt
import numpy as np
from random_process.pos_drift_skewed_stable_random_process import PosDriftTotallySkewedStableRandomProcess
from scale_function.sb_scale_function import SBScaleFunction
from stick_breaking_representation.fixed_interval_stick_breaking_representation import ExpIntervalStickBreakingRepresentation, FixedIntervalStickBreakingRepresentation
from stick_breaking_representation.inf_interval_stick_breaking_representation import InfIntervalStickBreakingRepresentation
from stick_breaking_representation.stick_breaking_representation_factory import StickBreakingRepresentationFactory


def main():
    alpha = 1.6
    drift = 0.1
    T = 1000
    epsilon = 0.001
    n_sticks = 20
    q = 0
    N = 1000

    X = PosDriftTotallySkewedStableRandomProcess(alpha, drift)

    P1 = StickBreakingRepresentationFactory(InfIntervalStickBreakingRepresentation, epsilon=epsilon)
    P2 = StickBreakingRepresentationFactory(FixedIntervalStickBreakingRepresentation, T=T, n_sticks=n_sticks)
    P3 = StickBreakingRepresentationFactory(ExpIntervalStickBreakingRepresentation, theta=1/T, n_sticks=n_sticks)
    # P4 = StickBreakingRepresentationFactory(FixedIntervalStickBreakingRepresentation, T=T, n_sticks=n_sticks)

    R = range(100,1100,100)
    for j in range(10, 15):
        epsilon = 0.1**j
        W1 = SBScaleFunction(q, X, P1, N)
        W2 = SBScaleFunction(q, X, P2, N)
        W3 = SBScaleFunction(q, X, P3, N)
        
        w1,w2,w3 = [],[],[]
        for i in R:
            # np.random.seed(0)
            # w1.append(W1.value(10))
            np.random.seed(0)
            # w2.append(W2.value(10))
            w3.append(W3.value(10))
        # plt.plot(R, w1, label=f'eps={'%.2E' % epsilon}')
        # plt.plot(R, w2, label=f'eps={'%.2E' % epsilon}')
        plt.plot(R, w3, label=f'eps={'%.2E' % epsilon}')
    plt.plot([R[0], R[-1]], [3.3133, 3.3133], c='r')
    plt.legend()
    plt.show()
    # print(w1[-1])


if __name__ == '__main__':
    main()
