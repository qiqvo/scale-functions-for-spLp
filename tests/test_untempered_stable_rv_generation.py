from matplotlib import pyplot as plt
import numpy as np
from random_variable.skewed_stable_random_variable import TotallySkewedStableRandomVariable
from random_variable.tempered_stable_random_variable import TemperedTotallySkewedStableRandomVariable, UntemperedTotallySkewedStableRandomVariable


def test_untempered_stable_rv_generation():
    alpha = 1.5
    N = 1000
    mult = 1
    c = 0
    
    X = TotallySkewedStableRandomVariable(alpha)
    s1 = X.sample(N)
    Y = UntemperedTotallySkewedStableRandomVariable(alpha)
    s2 = Y.sample(N*mult)
    Z = TemperedTotallySkewedStableRandomVariable(c, alpha)
    s3 = Z.sample(N*mult)

    plt.plot(range(N), np.sort(s1), label='X')
    plt.plot(np.array(range(N * mult))/mult, np.sort(s2), label='Y')
    plt.plot(np.array(range(N * mult))/mult, np.sort(s3), label='Z')
    plt.legend()
    plt.show()

    zs = np.transpose(np.linspace(0, 3, 100))
    lt = lambda s: np.log(np.mean(np.exp(np.outer(s, zs)), axis=0))
    lt_theor = (Y.psi(zs))
    lt1 = lt(s1)
    lt2 = lt(s2)
    lt3 = lt(s3)

    # cf = lambda s: (np.mean(np.exp(1j * np.outer(s, zs)), axis=0))
    # cf1 = cf(s1)
    # cf2 = cf(s2)
    # cf_theor = (Y.characteristic_function(zs))

    plt.plot(zs, lt1, label='empirical X')
    plt.plot(zs, lt2, label='empirical Y')
    plt.plot(zs, lt3, label='empirical Z')
    plt.plot(zs, lt_theor, label='theoretical Y')
    # plt.ylim(-1, np.max(lt_theor))
    plt.title('Log scaled comparison of Laplace transforms')
    plt.legend()
    plt.show()
