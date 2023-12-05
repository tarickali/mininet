"""
title : ones.py
create : @tarickali 23/11/26
update : @tarickali 23/12/05
"""

import numpy as np

from core import Initializer
from core.types import Shape


class Ones(Initializer):
    """Ones Initializer

    Initializes an array of a given shape with ones.

    """

    def init(self, shape: Shape = None) -> np.ndarray:
        return np.ones(shape)
