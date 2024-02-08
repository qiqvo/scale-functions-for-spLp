from matplotlib import pyplot as plt

from random_process.pos_drift_skewed_stable_random_process import PosDriftTotallySkewedStableRandomProcess


def main():
    alpha = 1.2
    X = PosDriftTotallySkewedStableRandomProcess(alpha, 0.1)
    s = X.sample(100, 1, 0)
    print(s)
    plt.plot(s)
    plt.show()


if __name__ == '__main__':
    main()
