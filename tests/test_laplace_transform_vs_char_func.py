from matplotlib import pyplot as plt
import numpy as np
from random_variable.skewed_stable_random_variable import TotallySkewedStableRandomVariable


def test_laplace_transform_vs_char_func():
    alpha = 1.6
    N = 10000
    plot_results = False
    
    X = TotallySkewedStableRandomVariable(alpha)
    s = X.sample(N)
    if plot_results:
        plt.plot(s)
        plt.show()

    zs = np.transpose(np.linspace(0.1, 3, 100))
    lt = np.log(np.mean(np.exp(np.outer(s, zs)), axis=0))
    lt_theor = np.log(X.laplace_transform(zs))

    cf = (np.mean(np.exp(1j * np.outer(s, zs)), axis=0))
    cf_theor = (X.characteristic_function(zs))

    if plot_results:
        plt.plot(zs, lt, label='empirical')
        plt.plot(zs, lt_theor, label='theoretical')
        # plt.ylim(-1, np.max(lt_theor))
        plt.title('Log scaled comparison of Laplace transforms')
        plt.legend()
        plt.show()

        print(X.laplace_transform(zs))
        plt.plot(zs, np.log(np.real(cf)), label='real empirical')
        plt.plot(zs, np.log(np.real(cf_theor)), label='real theoretical')    
        plt.plot(zs, np.log(np.imag(cf)), label='imag empirical')
        plt.plot(zs, np.log(np.imag(cf_theor)), label='imag theoretical')
        # plt.plot(zs, X.laplace_transform(zs), label='theoretical')
        # plt.ylim(-1, np.max(lt_theor))
        plt.title('Log scaled comparison of CF')
        plt.legend()
        plt.show()

    assert (np.log(np.abs(np.real(cf)))[-1]) == -4.089705140753553
    assert (np.log(np.abs(np.real(cf_theor)))[-1]) == -4.728088370147886
    assert (np.log(np.abs(np.imag(cf)))[-1]) == -7.103018861487568
    assert (np.log(np.abs(np.imag(cf_theor)))[-1]) == -6.0232698763980945
    assert (lt[-1]) == 6.122031931607335
    assert (lt_theor[-1]) == -5.799546134795289
    
