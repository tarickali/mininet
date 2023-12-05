"""
title : lecun_normal.py
create : @tarickali 23/11/26
update : @tarickali 23/12/05
"""

import numpy as np

from core import Initializer
from core.types import Shape


class LecunNormal(Initializer):
    """LecunNormal Initializer

    Initializes an array of a given shape with values sampled drawn from
    `N(0.0, std)`, where `std = sqrt(1.0 / fan_in)` and `fan_in` is the
    number of input units in the weight array.

    """

    def init(self, shape: Shape = None) -> np.ndarray:
        std = np.sqrt(1.0 / shape[0])
        return np.random.normal(0.0, std, shape)
