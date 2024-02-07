from matplotlib import pyplot as plt
from inf_interval_stick_breaking_process import InfIntervalStickBreakingProcess
from pos_drift_skewed_stable_random_process import PosDriftTotallySkewedStableRandomProcess


def main():
    alpha = 1.2
    epsilon = 0.001

    X = PosDriftTotallySkewedStableRandomProcess(alpha, 0.1)
    sbr = InfIntervalStickBreakingProcess(X, epsilon)

    m = 3
    s = sbr.sample(m)
    for i in range(m):
        plt.scatter(s[i][0], s[i][1], marker='x')
    plt.show()


if __name__ == '__main__':
    main()
