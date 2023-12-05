"""
title : he_uniform.py
create : @tarickali 23/11/26
update : @tarickali 23/12/05
"""

import numpy as np

from core import Initializer
from core.types import Shape


class HeUniform(Initializer):
    """HeUniform Initializer

    Initializes an array of a given shape with values sampled drawn from
    `U(-limit, limit)`, where `limit = sqrt(6.0 / fan_in)` and `fan_in` is
    the number of input units in the weight array.

    """

    def init(self, shape: Shape = None) -> np.ndarray:
        limit = np.sqrt(6.0 / shape[0])
        return np.random.uniform(-limit, limit, shape)
