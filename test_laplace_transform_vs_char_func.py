from matplotlib import pyplot as plt
import numpy as np
from random_variable.skewed_stable_random_variable import TotallySkewedStableRandomVariable


def main():
    alpha = 1.6
    N = 10000
    
    X = TotallySkewedStableRandomVariable(alpha)
    s = X.sample(N)
    plt.plot(s)
    plt.show()

    zs = np.transpose(np.linspace(0.1, 3, 100))
    lt = np.log(np.mean(np.exp(np.outer(s, zs)), axis=0))
    lt_theor = np.log(X.laplace_transform(zs))

    cf = (np.mean(np.exp(1j * np.outer(s, zs)), axis=0))
    cf_theor = (X.characteristic_function(zs))

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



if __name__ == '__main__':
    main()
