from matplotlib import pyplot as plt
import numpy as np
from inf_interval_stick_breaking_process import InfIntervalStickBreakingProcess
from pos_drift_skewed_stable_random_process import PosDriftTotallySkewedStableRandomProcess
from scale_function import FISBScaleFunction, IISBScaleFunction


def main():
    alpha = 1.6
    drift = 0.1
    X = PosDriftTotallySkewedStableRandomProcess(alpha, drift)

    # P = InfIntervalStickBreakingProcess(X, 0.001)
    # P.sample(10)
    # plt.scatter

    T = 1000
    n_sticks = 51

    R = range(100,1100,100)
    for j in range(10, 15):
        epsilon = 0.1**j
        # W_1 = IISBScaleFunction(X, epsilon)
        W_2 = FISBScaleFunction(X, 1/epsilon, n_sticks)
        # w1 = []
        w2 = []
        for i in R:
            # np.random.seed(0)
            # w1.append(W_1.sampled_value(10, i+2))
            np.random.seed(0)
            w2.append(W_2.sampled_value(10, i+2))
        # plt.plot(R, w1, label=f'eps={'%.2E' % epsilon}')
        plt.plot(R, w2, label=f'eps={'%.2E' % epsilon}')
    plt.plot([R[0], R[-1]], [3.3133, 3.3133], c='r')
    plt.legend()
    plt.show()
    # print(w1[-1])


if __name__ == '__main__':
    main()
