"""
title : softplus.py
create : @tarickali 23/11/26
update : @tarickali 23/11/26
"""

import numpy as np
from core.activation import Activation


class SoftPlus(Activation):
    def func(self, x: np.ndarray) -> np.ndarray:
        """ """

        return np.log(1 + np.exp(x))

    def grad(self, x: np.ndarray) -> np.ndarray:
        """ """

        e = np.exp(x)
        return e / (1 + e)
