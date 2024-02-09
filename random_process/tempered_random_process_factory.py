from random_process.skewed_stable_random_process import TotallySkewedStableRandomProcess
from random_process.spectrally_negative_levy_random_process import SpectrallyNegativeLevyRandomProcess
from random_process.tempered_skewed_stable_random_process import TemperedTotallySkewedStableRandomProcess
from random_process.tempered_spectrally_negative_levy_random_process import TemperedSpectrallyNegativeLevyRandomProcess

def create_tempered(x, c: float):
    if type(x) == SpectrallyNegativeLevyRandomProcess:
        return TemperedSpectrallyNegativeLevyRandomProcess(c, x.mu, x.sigma, x.nu, x.nu_unwarranted, x._max_jump_cutoff)
    if type(x) == TotallySkewedStableRandomProcess:
        return TemperedTotallySkewedStableRandomProcess(c, x.alpha)