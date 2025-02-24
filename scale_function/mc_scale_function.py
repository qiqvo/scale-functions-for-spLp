import numpy as np
import scipy

from random_process.spectrally_negative_levy_random_process import SpectrallyNegativeLevyRandomProcess
from random_process.tempered_random_process_factory import create_tempered
from scale_function.tempered_scale_function import TemperedScaleFunction


class MCScaleFunction(TemperedScaleFunction):
    """
    Based on the paper 
    "Markov chain approximations to scale functions of Levy processes" (2013) 
    by A. Mijatovic, M. Vidmar and S. Jacka  [arXiv:1310.1737]

    and the code provided by M. Vidmar.
    """
    def __init__(self, q:float, process: SpectrallyNegativeLevyRandomProcess, 
                 h:float, 
                 upper_cutoff:float,
                 setup_at_init:bool=False) -> None:
        super().__init__(q, process)
        self.h = h
        self.upper_cutoff = upper_cutoff

        if isinstance(self, TemperedScaleFunction):
            self.original_q = self.q
            self.q = 0
            
        self.xi = self.process.get_underlying_xi_for_time(1)
        if setup_at_init:
            self._setup()
        else: 
            self.Q = None
            self.p = None
            self.total_mass = None
    
    def _mu_h(self):
        if not self.process.is_infinite_activity():
            return 0
        
        res = 0 
        k = 1
        a = self.h / 2
        b = a + self.h
        while a < 1:
            r = self.xi.get_nu_measure(-min(b, 1), -a, unwarranted=True)
            res -= k * r
            k += 1

            a = b
            b += self.h
        res *= self.h
        return res
    
    def _c(self, k: int):
        if k == 0:
            if self.process.is_infinite_activity():
                res = self.xi.get_nu_measure(-0.5 * self.h, 0, power=2, unwarranted=True)
            else:
                res = 0
        else:
            res = self.xi.get_nu_measure(-(k + 0.5)*self.h, -(k - 0.5)*self.h, unwarranted=True)
        return res
    
    def _gamma(self): 
        res = self.xi.get_nu_measure(-self.upper_cutoff, -3/2 *self.h, unwarranted=True)
        return res 

    def _setup(self):
        upper_cutoff = int(self.upper_cutoff)
        Q = np.zeros(upper_cutoff)  
        # Initial definitions: p = \measure^h(\{h\}) and Q(i) = \measure^h(\{-ih\}), i=1,...,n.
        # Final definition divides these by total, to get the probabilities.

        mu = self.xi.mu
        h = self.h
        mu_h = self._mu_h()
        c0 = self._c(0) 
        c1 = self._c(1)
        if self.xi.sigma > 0:
            sigma2 = self.xi.sigma ** 2
            p = (sigma2 + c0) / (2 * h * h) + (mu - mu_h) / (2 * h)
            Q[0] = c1 + (sigma2 + c0) / (2 * h * h) - (mu - mu_h) / (2 * h)
        else: 
            p = (mu - mu_h) / h + c0 / (2 * h * h)
            Q[0] = c1 + c0 / (2 * h * h)

        # Is the algorithm well-defined, i.e. have we defined an upwards skip-free
        # Levy chain (see Definition 3.1 in the paper)? If not, we need a smaller h!
        assert Q[0] >= 0 and p > 0

        for k in range(2, upper_cutoff):  # compute entries for Q(2:n)
            Q[k - 1] = self._c(k)

        self.total_mass = p + Q[0] + self._gamma()  # compute total Levy mass of the approximating chain X^h

        self.p = p / self.total_mass  # normalize p and Q to become probabilities of the jump-chain.
        self.Q = Q / self.total_mass

    def value(self, x: float) -> float:
        x, W = self.profile(x, x + self.h)
        return W[0]
    
    def _inner_profile(self, a: float, b: float) -> tuple[np.ndarray, np.ndarray]:
        assert b < int(self.upper_cutoff)
        if self.Q is None:
            self._setup()

        n = int(np.ceil(b / self.h))
        xs = np.linspace(0, b, n)

        W = np.zeros(n)
        W[0] = 1 / (self.p * self.total_mass)  # W(0) is W^{(q)}(0)
        if n > 1:
            W[1] = (1 + self.q / self.total_mass) * W[0] / self.p  # W(1) is W^{(q)}(h)
            if n > 2:
                for k in range(1, n-1):  # linear recursion for W^{(q)}, cf. Eqs~(1.1) and~(3.1) in the paper.
                    s = np.dot(self.Q[:k], W[k-1::-1])  # sum Q[l]*W[k-l] for l=1 to k
                    W[k + 1] = ((1 + self.q / self.total_mass) * W[k] - s) / self.p

        rs = xs >= a
        xs, W = xs[rs], W[rs]
        if self.process.is_infinite_activity():
            W[1:] = W[:-1]
            W[0] = 0
        W = W / self.h  # divide by h, to obtain W^{(q)} for Y (rather than Y/h).
        return xs, W