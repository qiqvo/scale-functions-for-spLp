from matplotlib import pyplot as plt
import numpy as np
from random_process.drift_random_process import DriftRandomProcess

from random_process.skewed_stable_random_process import TotallySkewedStableRandomProcess
from scale_function.mc_scale_function import MCScaleFunction
from scale_function.pos_drift_stable_scale_function import PosDriftTotallySkewedStableScaleFunction
from scale_function.skewed_stable_scale_function import TotallySkewedStableScaleFunction


def test_mc_scale_function():
    a, b = 0, 1
    alpha = 1.5
    q = 1
    h = 1/2**9
    upper_cutoff = 1000
    # drift = 0.3

    Y = TotallySkewedStableRandomProcess(alpha)
    W = TotallySkewedStableScaleFunction(q, Y)
    # X = DriftRandomProcess(drift, Y)
    # W = PosDriftTotallySkewedStableScaleFunction(q, X)

    V = MCScaleFunction(q, Y, h, upper_cutoff)
    # V = MCScaleFunction(q, X, h, upper_cutoff)

    xs, ws = W.profile(a, b)
    plt.plot(xs, ws, label='W')
    xs, vs = V.profile(a, b)
    plt.plot(xs, vs, label='sampled W')
    plt.title('Comparison of sampled W with the real W, Matija alg')

    plt.legend()
    plt.show()
    
    assert ws[43] == 1.501119221139845
    assert vs[43] == 0.29646400934019485
