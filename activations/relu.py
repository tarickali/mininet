"""
title : relu.py
create : @tarickali 23/11/26
update : @tarickali 23/12/05
"""

import numpy as np
from core.activation import Activation


class ReLU(Activation):
    """ReLU Activation

    Computes the elementwise function `f(x) = max(x, 0)`.

    """

    def func(self, x: np.ndarray) -> np.ndarray:
        """ """

        return np.maximum(x, 0)

    def grad(self, x: np.ndarray) -> np.ndarray:
        """ """

        return (x > 0).astype(x.dtype)
