from matplotlib import pyplot as plt
from random_process.drift_random_process import DriftRandomProcess
from random_process.skewed_stable_random_process import TotallySkewedStableRandomProcess
from stick_breaking_representation.inf_interval_stick_breaking_representation import InfIntervalStickBreakingRepresentation


def main():
    alpha = 1.2
    epsilon = 0.001
    drift = 0.1
    Y = TotallySkewedStableRandomProcess(alpha)
    X = DriftRandomProcess(drift, Y)
    sbr = InfIntervalStickBreakingRepresentation(X, epsilon)

    m = 3
    s = sbr.sample(m)
    for i in range(m):
        plt.scatter(s[i][0], s[i][1], marker='x')
    plt.show()


if __name__ == '__main__':
    main()
