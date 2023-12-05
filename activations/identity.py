"""
title : identity.py
create : @tarickali 23/11/26
update : @tarickali 23/12/05
"""

import numpy as np
from core.activation import Activation


class Identity(Activation):
    """Identity Activation

    Computes the elementwise function `f(x) = x`.

    """

    def func(self, x: np.ndarray) -> np.ndarray:
        return x

    def grad(self, x: np.ndarray) -> np.ndarray:
        return np.ones_like(x)
