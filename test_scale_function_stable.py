from matplotlib import pyplot as plt
import numpy as np

from random_process.pos_drift_skewed_stable_random_process import PosDriftTotallySkewedStableRandomProcess
from scale_function.pos_drift_stable_scale_function import PosDriftTotallySkewedStableScaleFunction
from scale_function.sb_scale_function import SBScaleFunction
from stick_breaking_representation.fixed_interval_stick_breaking_representation import FixedIntervalStickBreakingRepresentation
from stick_breaking_representation.inf_interval_stick_breaking_representation import InfIntervalStickBreakingRepresentation


def main():
    alpha = 1.6
    q = 0
    drift = 0.1
    T = 500
    epsilon = 0.001
    n_sticks = 50
    N = 1000

    X = PosDriftTotallySkewedStableRandomProcess(alpha, drift)
    W = PosDriftTotallySkewedStableScaleFunction(q, X)

    P1 = InfIntervalStickBreakingRepresentation(X, epsilon)
    P2 = FixedIntervalStickBreakingRepresentation(X, T, n_sticks)
    # P4 = FixedIntervalStickBreakingProcess(X, T, n_sticks * 2)
    # P3 = ExpIntervalStickBreakingProcess(X, 1/T, n_sticks)

    V1 = SBScaleFunction(q, X, P1, N)
    V2 = SBScaleFunction(q, X, P2, N)

    R = np.linspace(0.1,10,20)
    ws = W.profile(R)
    v1s = V1.profile(R)
    v2s = V2.profile(R)
    plt.plot(R, ws, label='W')
    plt.plot(R, v1s, label='sampled W, inf interval')
    plt.plot(R, v2s, label='sampled W, fixed interval')
    plt.title('Fixed interval sampling')

    plt.legend()
    plt.show()

    # R = range(100,1100,100)
    # for j in range(10, 15):
    #     epsilon = 0.1**j
    #     W1 = SBScaleFunction(X, P1)
    #     W2 = SBScaleFunction(X, P2)
    #     W3 = SBScaleFunction(X, P3)
        
    #     w1,w2,w3 = [],[],[]
    #     for i in R:
    #         # np.random.seed(0)
    #         # w1.append(W1.sampled_value(10, i+2))
    #         np.random.seed(0)
    #         # w2.append(W2.sampled_value(10, i+2))
    #         w3.append(W3.sampled_value(10, i+2))
    #     # plt.plot(R, w1, label=f'eps={'%.2E' % epsilon}')
    #     # plt.plot(R, w2, label=f'eps={'%.2E' % epsilon}')
    #     plt.plot(R, w3, label=f'eps={'%.2E' % epsilon}')
    # plt.plot([R[0], R[-1]], [3.3133, 3.3133], c='r')
    # plt.legend()
    # plt.show()


if __name__ == '__main__':
    main()
