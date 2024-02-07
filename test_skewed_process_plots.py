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


if __name__ == '__main__':
    main()
