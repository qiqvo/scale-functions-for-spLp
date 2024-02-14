from matplotlib import pyplot as plt
import numpy as np
from random_variable.skewed_stable_random_variable import TotallySkewedStableRandomVariable
from random_variable.tempered_stable_random_variable import TemperedTotallySkewedStableRandomVariable, UntemperedTotallySkewedStableRandomVariable


def test_small_c_perturbation_stable_simulation():
    alpha = 1.5
    N = 1000
    mult = 1
    c = 1e-6
    
    X = TotallySkewedStableRandomVariable(alpha)
    s1 = X.sample(N)
    Y = UntemperedTotallySkewedStableRandomVariable(alpha)
    s2 = Y.sample(N)
    Z = TemperedTotallySkewedStableRandomVariable(c, alpha)
    s3 = Z.sample(N)
    # s3 -= np.mean(s3)

    plt.plot(np.arange(N)/N, np.sort(s1), label='X')
    plt.plot(np.arange(N)/N, np.sort(s2), label='Y')
    plt.plot(np.arange(N)/N, np.sort(s3), label='Z')
    plt.legend()
    plt.show()

    zs = np.transpose(np.linspace(0, 3, 100))
    lt = lambda s: np.log(np.mean(np.exp(np.outer(s, zs)), axis=0))
    lt_theor = X.psi(zs)
    lt_theor_c = Z.psi(zs)
    lt1 = lt(s1)
    lt2 = lt(s2)
    lt3 = lt(s3)

    plt.plot(zs, lt1, label='empirical X')
    plt.plot(zs, lt2, label='empirical Y')
    plt.plot(zs, lt3, label='empirical Z')
    plt.plot(zs, lt_theor, label='theoretical X')
    plt.plot(zs, lt_theor_c, label='theoretical Z')
    plt.title('Log scaled comparison of Laplace transforms')
    plt.legend()
    plt.show()
