from matplotlib import pyplot as plt
import numpy as np
from random_variable.skewed_stable_random_variable import TotallySkewedStableRandomVariable
from random_variable.tempered_stable_random_variable import TemperedTotallySkewedStableRandomVariable, UntemperedTotallySkewedStableRandomVariable


def main():
    alpha = 1.9999
    N = 10000
    mult = 1
    c = 0
    multiplier = 1000
    
    X = TotallySkewedStableRandomVariable(alpha, multiplier=multiplier)
    s1 = X.sample(N)
    Y = UntemperedTotallySkewedStableRandomVariable(alpha, multiplier=multiplier)
    s2 = Y.sample(N*mult)
    Z = TemperedTotallySkewedStableRandomVariable(c, alpha, multiplier=multiplier)
    s3 = Z.sample(N*mult)

    plt.plot(range(N), np.sort(s1), label='X')
    plt.plot(np.array(range(N * mult))/mult, np.sort(s2), label='Y')
    plt.plot(np.array(range(N * mult))/mult, np.sort(s3), label='Z')
    plt.legend()
    plt.show()

    zs = np.transpose(np.linspace(0, 3, 100))
    lt = lambda s: np.log(np.mean(np.exp(np.outer(s, zs)), axis=0))
    lt_theor = np.log(Y.laplace_transform(zs))
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

    # print(X.laplace_transform(zs))
    # plt.plot(zs, np.log(np.real(cf1)), label='real empirical X')
    # plt.plot(zs, np.log(np.real(cf2)), label='real empirical Y')
    # plt.plot(zs, np.log(np.real(cf_theor)), label='real theoretical X')    
    # plt.plot(zs, np.log(np.imag(cf1)), label='imag empirical X')
    # plt.plot(zs, np.log(np.imag(cf2)), label='imag empirical Y')
    # plt.plot(zs, np.log(np.imag(cf_theor)), label='imag theoretical X')
    # # plt.plot(zs, X.laplace_transform(zs), label='theoretical')
    # # plt.ylim(-1, np.max(lt_theor))
    # plt.title('Log scaled comparison of CF')
    # plt.legend()
    # plt.show()



if __name__ == '__main__':
    main()
