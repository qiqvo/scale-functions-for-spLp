from matplotlib import pyplot as plt
from inf_interval_stick_breaking_process import InfIntervalStickBreakingProcess
from pos_drift_skewed_stable_random_process import PosDriftTotallySkewedStableRandomProcess
from scale_function import FISBScaleFunction, IISBScaleFunction


def main():
    alpha = 1.2
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

    W_1 = IISBScaleFunction(X, epsilon)
    W_2 = FISBScaleFunction(X, epsilon)
    for i in range(1,1000,100):
        print(W_1.sampled_value(10, i+2))
        print(W_2.sampled_value(10, i+2))


if __name__ == '__main__':
    main()
