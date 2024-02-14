from matplotlib import pyplot as plt
import numpy as np
import scipy

from random_process.drift_random_process import DriftRandomProcess
from random_process.skewed_stable_random_process import TotallySkewedStableRandomProcess
from scale_function.pos_drift_stable_scale_function import PosDriftTotallySkewedStableScaleFunction
from scale_function.sb_scale_function import SBScaleFunction
from stick_breaking_representation.fixed_interval_stick_breaking_representation import ExpIntervalStickBreakingRepresentation, FixedIntervalStickBreakingRepresentation
from stick_breaking_representation.inf_interval_stick_breaking_representation import InfIntervalStickBreakingRepresentation
from stick_breaking_representation.stick_breaking_representation_factory import StickBreakingRepresentationFactory


A = np.array(
    [[0.12999999999999998, 0.12, 0.42, 0.04805288420993828, 0.046374270056901805, 0.2454159863799698],
     [0.014289607895597274, 0.013190403479237053, 0.04629453867221415, 0.0013129739980652986, 0.0013121415167207254, 0.0093018658119334],
     [0.40005604381114446, 0.3770161177259379, 0.9117289147538776, 0.24489795918367346, 0.22448979591836732, 0.836734693877551,],])


def test_q_scale_function_pos_drift_stable():
    alpha = 1.7
    drift = 0.1
    T = 1e6
    epsilon = 1e-4
    n_sticks = int(np.floor(np.log(T / epsilon) / np.log(2)))
    N = 1000
    plot_results = True

    Y = TotallySkewedStableRandomProcess(alpha)
    X = DriftRandomProcess(drift, Y)
    P2 = StickBreakingRepresentationFactory(FixedIntervalStickBreakingRepresentation, T=T, n_sticks=n_sticks)

    W = PosDriftTotallySkewedStableScaleFunction(0, X)

    a, b = 0, 1
    xs, ws = W.profile(a, b)
    i = -1
    assert ws[12] == A[i, 0]
    assert ws[11] == A[i, 1]
    assert ws[41] == A[i, 2]
    assert xs[12] == A[i, 3]
    assert xs[11] == A[i, 4]
    assert xs[41] == A[i, 5]

    fig = plt.figure()
    ax1 = fig.subplots(1, 1)
    # ax1, ax2 = fig.subplots(2, 1)
    ax1.plot(xs, ws, label='W')
    # ax2.plot(xs, ws, label='W')

    qs = [0, 1e-3, 0.1, 1]
    # qs = [0, 1e-10, 1e-5, 1e-2, 0.1, 0.2]
    # qs = [0, 0.2]
    for i in range(len(qs)):
        q = qs[i]
        V2 = SBScaleFunction(q, X, P2, N)
        xs, vs = V2.profile(a, b)
        ax1.plot(xs, vs, label=f'sampled W^(q), q={'%.2E' % q}')
        # smoothed_vs = scipy.signal.savgol_filter(vs, window_length=5, polyorder=2)
        # coefs = np.polyfit(vs, xs, 5)
        # smoothed_vs = np.linspace(vs[0], vs[-1])
        # smoothed_xs = np.polyval(coefs, smoothed_vs)

        # assert vs[12] == A[i, 0]
        # assert vs[11] == A[i, 1]
        # assert vs[41] == A[i, 2]
        # assert xs[12] == A[i, 3]
        # assert xs[11] == A[i, 4]
        # assert xs[41] == A[i, 5]

        # ax2.plot(smoothed_xs, smoothed_vs, 
        #          label=f'sampled W^(q), q={'%.2E' % q}')
    ax1.set_title('Comparison for different q, Pos Drift Stable')
    ax1.legend()
    # ax2.set_title('Comparison for different q, Pos Drift Stable, smoothed')
    # ax2.legend()

    if plot_results:
        plt.show()

