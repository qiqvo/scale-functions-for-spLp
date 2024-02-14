from matplotlib import pyplot as plt
import numpy as np
from random_variable.skewed_stable_random_variable import TotallySkewedStableRandomVariable
from random_variable.tempered_stable_random_variable import UntemperedTotallySkewedStableRandomVariable


def test_generic_levy_rv_generation():
    alpha = 1.5
    N = 10
    mult = 1
    plot_results = False
    
    X = TotallySkewedStableRandomVariable(alpha)
    s1 = X.sample(N)
    Y = UntemperedTotallySkewedStableRandomVariable(alpha)
    s2 = Y.sample(N*mult)

    if plot_results:
        plt.plot(range(N), np.sort(s1), label='X')
        plt.plot(np.array(range(N * mult))/mult, np.sort(s2), label='Y')
        plt.show()

    zs = np.transpose(np.linspace(0.1, 3, 100))
    lt = lambda s: np.log(np.mean(np.exp(np.outer(s, zs)), axis=0))
    lt_theor = np.log(X.laplace_transform(zs))
    lt1 = lt(s1)
    lt2 = lt(s2)

    cf = lambda s: (np.mean(np.exp(1j * np.outer(s, zs)), axis=0))
    cf1 = cf(s1)
    cf2 = cf(s2)
    cf_theor = (X.characteristic_function(zs))

    if plot_results:
        plt.plot(zs, lt1, label='empirical X')
        plt.plot(zs, lt2, label='empirical Y')
        plt.plot(zs, lt_theor, label='theoretical X')
        # plt.ylim(-1, np.max(lt_theor))
        plt.title('Log scaled comparison of Laplace transforms')
        plt.legend()
        plt.show()

        plt.plot(zs, np.log(np.real(cf1)), label='real empirical X')
        plt.plot(zs, np.log(np.real(cf2)), label='real empirical Y')
        plt.plot(zs, np.log(np.real(cf_theor)), label='real theoretical X')    
        plt.plot(zs, np.log(np.imag(cf1)), label='imag empirical X')
        plt.plot(zs, np.log(np.imag(cf2)), label='imag empirical Y')
        plt.plot(zs, np.log(np.imag(cf_theor)), label='imag theoretical X')
        # plt.plot(zs, X.laplace_transform(zs), label='theoretical')
        # plt.ylim(-1, np.max(lt_theor))
        plt.title('Log scaled comparison of CF')
        plt.legend()
        plt.show()

    assert (np.log(np.abs(np.real(cf2)))[-1]) == -1.1461180779624083
    assert np.abs((np.log(np.abs(np.real(cf1)))[-1]) - (-2.659630474728008)) == 0
    assert (np.log(np.abs(np.real(cf_theor)))[-1]) == -3.823351446412928
    assert (np.log(np.abs(np.imag(cf1)))[-1]) == -1.043393876176342
    assert (np.log(np.abs(np.imag(cf2)))[-1]) == -2.8419661504231524
    assert (np.log(np.abs(np.imag(cf_theor)))[-1]) == -4.351880411350226
    assert (lt1[-1]) == 8.370860017980773
    assert (lt2[-1]) == 5.95916662670356
    assert (lt_theor[-1]) == -5.196152422706632
    