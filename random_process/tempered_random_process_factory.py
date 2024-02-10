from random_process.drift_random_process import DriftRandomProcess
from random_process.skewed_stable_random_process import TotallySkewedStableRandomProcess
from random_process.spectrally_negative_levy_random_process import SpectrallyNegativeLevyRandomProcess
from random_process.tempered_skewed_stable_random_process import TemperedTotallySkewedStableRandomProcess
from random_process.tempered_spectrally_negative_levy_random_process import TemperedSpectrallyNegativeLevyRandomProcess

def create_tempered(x: SpectrallyNegativeLevyRandomProcess, c: float):
    if isinstance(x, DriftRandomProcess) and isinstance(x.process, TotallySkewedStableRandomProcess):
        process = create_tempered(x.process)
        return DriftRandomProcess(x.drift, process)
    if isinstance(x, TotallySkewedStableRandomProcess):
        return TemperedTotallySkewedStableRandomProcess(c, x.alpha)
    if isinstance(x, SpectrallyNegativeLevyRandomProcess):
        return TemperedSpectrallyNegativeLevyRandomProcess(c, x.mu, x.sigma, x.nu, x.nu_unwarranted, x._max_jump_cutoff)
