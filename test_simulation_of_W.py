from matplotlib import pyplot as plt
import numpy as np
from fixed_interval_stick_breaking_process import ExpIntervalStickBreakingProcess, FixedIntervalStickBreakingProcess
from inf_interval_stick_breaking_process import InfIntervalStickBreakingProcess
from pos_drift_skewed_stable_random_process import PosDriftTotallySkewedStableRandomProcess
from scale_function import SBScaleFunction


def main():
    alpha = 1.6
    drift = 0.1
    T = 1000
    epsilon = 0.001
    n_sticks = 20

    X = PosDriftTotallySkewedStableRandomProcess(alpha, drift)

    P1 = InfIntervalStickBreakingProcess(X, epsilon)
    P2 = FixedIntervalStickBreakingProcess(X, T, n_sticks)
    P3 = ExpIntervalStickBreakingProcess(X, 1/T, n_sticks)

    R = range(100,1100,100)
    for j in range(10, 15):
        epsilon = 0.1**j
        W1 = SBScaleFunction(X, P1)
        W2 = SBScaleFunction(X, P2)
        W3 = SBScaleFunction(X, P3)
        
        w1,w2,w3 = [],[],[]
        for i in R:
            # np.random.seed(0)
            # w1.append(W1.sampled_value(10, i+2))
            np.random.seed(0)
            # w2.append(W2.sampled_value(10, i+2))
            w3.append(W3.sampled_value(10, i+2))
        # plt.plot(R, w1, label=f'eps={'%.2E' % epsilon}')
        # plt.plot(R, w2, label=f'eps={'%.2E' % epsilon}')
        plt.plot(R, w3, label=f'eps={'%.2E' % epsilon}')
    plt.plot([R[0], R[-1]], [3.3133, 3.3133], c='r')
    plt.legend()
    plt.show()
    # print(w1[-1])


if __name__ == '__main__':
    main()
