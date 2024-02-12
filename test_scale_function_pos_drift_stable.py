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
    q = 0
    drift = 0.1
    T = 1e4
    epsilon = 1e-3
    n_sticks = np.floor(np.log(T / epsilon) / np.log(2))
    N = 500

    Y = TotallySkewedStableRandomProcess(alpha)
    X = DriftRandomProcess(drift, Y)
    W = PosDriftTotallySkewedStableScaleFunction(q, X)

    P1 = StickBreakingRepresentationFactory(InfIntervalStickBreakingRepresentation, epsilon=epsilon, T=T)
    P2 = StickBreakingRepresentationFactory(FixedIntervalStickBreakingRepresentation, T=T, n_sticks=n_sticks)
    P3 = StickBreakingRepresentationFactory(ExpIntervalStickBreakingRepresentation, theta=1/T, n_sticks=n_sticks)

    V1 = SBScaleFunction(q, X, P1, N)
    V2 = SBScaleFunction(q, X, P2, N)
    V3 = SBScaleFunction(q, X, P3, N)

    a, b = 0, 3
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
