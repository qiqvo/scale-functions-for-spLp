from matplotlib import pyplot as plt
from random_process.pos_drift_skewed_stable_random_process import PosDriftTotallySkewedStableRandomProcess
from stick_breaking_representation.inf_interval_stick_breaking_representation import InfIntervalStickBreakingRepresentation


def main():
    alpha = 1.2
    epsilon = 0.001

    X = PosDriftTotallySkewedStableRandomProcess(alpha, 0.1)
    sbr = InfIntervalStickBreakingRepresentation(X, epsilon)

    m = 3
    s = sbr.sample(m)
    for i in range(m):
        plt.scatter(s[i][0], s[i][1], marker='x')
    plt.show()


if __name__ == '__main__':
    main()
