"""
title : xavier_normal.py
create : @tarickali 23/11/26
update : @tarickali 23/12/05
"""

import numpy as np

from core import Initializer
from core.types import Shape


class XavierNormal(Initializer):
    """XavierNormal Initializer

    Initializes an array of a given shape with values sampled drawn from
    `N(0.0, std)`, where `std = sqrt(2.0 / fan_in + fan_out)` and
    `fan_in` and `fan_out` are the number of input and output units in the
    weight array, respectively.

    """

    def init(self, shape: Shape = None) -> np.ndarray:
        std = np.sqrt(2.0 / (shape[0] + shape[1]))
        return np.random.normal(0.0, std, shape)
