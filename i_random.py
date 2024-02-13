from abc import ABC

import numpy as np

from settings import bit_generator

class IRandom(ABC):
    rng = np.random.default_rng(bit_generator)