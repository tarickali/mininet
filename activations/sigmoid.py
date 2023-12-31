"""
title : sigmoid.py
create : @tarickali 23/11/26
update : @tarickali 23/12/05
"""

import numpy as np
from core.activation import Activation


class Sigmoid(Activation):
    """Sigmoid Activation

    Computes the function `f(x) = 1 / (1 + exp(-x))`.

    """

    def func(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    def grad(self, x: np.ndarray) -> np.ndarray:
        s = self.func(x)
        return s * (1 - s)
