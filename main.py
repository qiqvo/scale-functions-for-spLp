from matplotlib import pyplot as plt
import numpy as np

from skewed_stable_random_variable import TotallySkewedStableRandomVariable


def main():
    alpha = 1.1
    X = TotallySkewedStableRandomVariable(alpha)
    s = X.sample(100)
    print(s)
    plt.plot(s)
    plt.show()

if __name__ == '__main__':
    main()
