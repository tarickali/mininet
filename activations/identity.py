"""
title : identity.py
create : @tarickali 23/11/26
update : @tarickali 23/11/26
"""

import numpy as np
from core.activation import Activation


class Identity(Activation):
    def func(self, x: np.ndarray) -> np.ndarray:
        """ """

        return x

    def grad(self, x: np.ndarray) -> np.ndarray:
        """ """

        return np.ones_like(x)
