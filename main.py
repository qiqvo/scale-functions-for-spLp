from matplotlib import pyplot as plt
import numpy as np
import time

from fejer_de_la_vallee_poussin_random_variable import FejerDeLaValleePoussinRandomVariable
from immigration_const_rv import ImmigrationConstRandomVariable
from linnik import Linnik
from reproduction_const_rv import ReproductionConstRandomVariable
from stable_random_variable import StableRandomVariable
from xi import Xi


def main():
    alpha = 0.4
    c = 0.4
    d = 0.2
    delta = d / (alpha * c)

    N = 100

    # x = StableRandomVariable(alpha)
    # x = FejerDeLaValleePoussinRandomVariable()
    # x = Xi(alpha)
    # x = Linnik(alpha, delta)
    x = ImmigrationConstRandomVariable(alpha, d)
    # x = ReproductionConstRandomVariable(alpha, c)

    et = []
    for _ in range(30):
        start_time = time.time()
        s = x.sample(N)
        end_time = time.time()
        et.append(end_time - start_time)
        print('elapsed time: ', et[-1])

    print('mean elapsed time: ', np.mean(et))
    print('sqrt var elapsed time: ', np.sqrt(np.var(et)))
    # print(s)
    # plt.hist(s)
    # plt.show()

if __name__ == '__main__':
    main()
