from matplotlib import pyplot as plt
import numpy as np
from inf_interval_stick_breaking_process import InfIntervalStickBreakingProcess
from pos_drift_skewed_stable_random_process import PosDriftTotallySkewedStableRandomProcess
from scale_function import FISBScaleFunction, IISBScaleFunction
from skewed_stable_random_variable import TotallySkewedStableRandomVariable
from tempered_stable_random_variable import TemperedTotallySkewedStableRandomVariable, UntemperedTotallySkewedStableRandomVariable


def main():
    alpha = 1.5
    N = 1000
    mult = 1
    c = 10
    
    X = TotallySkewedStableRandomVariable(alpha)
    s1 = X.sample(N)
    Y = TemperedTotallySkewedStableRandomVariable(alpha, 0)
    s2 = Y.sample(N*mult)
    Z = UntemperedTotallySkewedStableRandomVariable(alpha)
    s3 = Z.sample(N*mult)

    W = TemperedTotallySkewedStableRandomVariable(alpha, c)
    s4 = W.sample(N*mult)

    plt.plot(range(N), np.sort(s1), label='X')
    plt.plot(np.array(range(N * mult))/mult, np.sort(s2), label='Y')
    plt.plot(np.array(range(N * mult))/mult, np.sort(s3), label='Z')
    plt.plot(np.array(range(N * mult))/mult, np.sort(s4), label='W')
    plt.legend()
    plt.show()

    zs = np.transpose(np.linspace(0, 3, 100))
    lt = lambda s: np.log(np.mean(np.exp(np.outer(s, zs)), axis=0))
    lt_theor = np.log(W.laplace_transform(zs))
    lt1 = lt(s1)
    lt2 = lt(s2)
    lt3 = lt(s3)
    lt4 = lt(s4)

    plt.plot(zs, lt1, label='empirical X')
    plt.plot(zs, lt2, label='empirical Y')
    plt.plot(zs, lt3, label='empirical Z')
    plt.plot(zs, lt4, label='empirical W')
    plt.plot(zs, lt_theor, label='theoretical W')
    plt.title('Log scaled comparison of Laplace transforms')
    plt.legend()
    plt.show()



if __name__ == '__main__':
    main()
