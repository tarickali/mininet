"""
title : tanh.py
create : @tarickali 23/11/26
update : @tarickali 23/12/05
"""

import numpy as np
from core.activation import Activation


class Tanh(Activation):
    """Tanh Activation

    Computes the function `f(x) = tanh(x)`.

    """

    def func(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    def grad(self, x: np.ndarray) -> np.ndarray:
        return 1 - np.tanh(x) ** 2
