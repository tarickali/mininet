"""
title : relu.py
create : @tarickali 23/11/26
update : @tarickali 23/11/26
"""

import numpy as np
from core.activation import Activation


class ReLU(Activation):
    def func(self, x: np.ndarray) -> np.ndarray:
        """ """

        return np.maximum(x, 0)

    def grad(self, x: np.ndarray) -> np.ndarray:
        """ """

        return (x > 0).astype(x.dtype)
