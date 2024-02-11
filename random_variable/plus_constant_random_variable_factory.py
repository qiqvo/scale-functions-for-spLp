import numpy as np

from random_variable.const_random_variable import ConstRandomVariable
from random_variable.multiple_of_random_variable import MultipleOfRandomVariable
from random_variable.random_variable import RandomVariable
from random_variable.spectrally_negative_levy_random_variable import SpectrallyNegativeLevyRandomVariable
from random_variable.sum_of_independent_random_variable import SumOfIndependentRandomVariable
from random_variable.tempered_spectrally_negative_levy_random_variable import TemperedSpectrallyNegativeLevyRandomVariable
from random_variable.tempered_stable_random_variable import TemperedTotallySkewedStableRandomVariable

def create_random_variable_plus_constant(const: float, rv: RandomVariable) -> RandomVariable:
    if isinstance(rv, MultipleOfRandomVariable):
        return MultipleOfRandomVariable(rv.multiplier, create_random_variable_plus_constant(const / rv.multiplier, rv.random_variable))
    if isinstance(rv, TemperedSpectrallyNegativeLevyRandomVariable):
        return TemperedSpectrallyNegativeLevyRandomVariable(rv.c, rv.mu + const, rv.sigma, rv.nu, rv.nu_unwarranted, rv.char_multiplier, rv.amplitude_multiplier, rv.max_jump_cutoff)
    if isinstance(rv, SpectrallyNegativeLevyRandomVariable):
        return SpectrallyNegativeLevyRandomVariable(rv.mu + const, rv.sigma, rv.nu, rv.nu_unwarranted, rv.char_multiplier, rv.amplitude_multiplier, rv.max_jump_cutoff)
    
    new_rv = SumOfIndependentRandomVariable(ConstRandomVariable(const), rv)
    return new_rv
