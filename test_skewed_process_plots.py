from matplotlib import pyplot as plt
import numpy as np
from random_process.drift_random_process import DriftRandomProcess

from random_process.skewed_stable_random_process import TotallySkewedStableRandomProcess



def main():
    alpha = 1.2
    drift = 10
    Y = TotallySkewedStableRandomProcess(alpha)
    X = DriftRandomProcess(drift, Y)

    u = Y.sample(100, 1, 0)
    s = X.sample(100, 1, 0)
    plt.plot(np.sort(u), label='Y')
    plt.plot(np.sort(s), label='Y+drift')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
