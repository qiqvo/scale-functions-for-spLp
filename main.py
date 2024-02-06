from matplotlib import pyplot as plt
import numpy as np
from inf_interval_stick_breaking_process import InfIntervalStickBreakingProcess
from pos_drift_skewed_stable_random_process import PosDriftTotallySkewedStableRandomProcess
from scale_function import IISBScaleFunction
from skewed_stable_random_process import TotallySkewedStableRandomProcess

from skewed_stable_random_variable import TotallySkewedStableRandomVariable


def main():
    alpha = 1.9
    X = PosDriftTotallySkewedStableRandomProcess(alpha, 0.1)
    s = X.sample(100, 1, 0)
    print(s)
    plt.plot(s)
    plt.show()

    epsilon = 0.001
    sbr = InfIntervalStickBreakingProcess(X, epsilon)
    m = 3
    s = sbr.sample(m)
    for i in range(m):
        plt.scatter(s[i][0], s[i][1], marker='x')
    plt.show()

    W = IISBScaleFunction(X, epsilon)
    for i in range(10):
        print(W.sampled_value(10, i+2))

if __name__ == '__main__':
    main()
