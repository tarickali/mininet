"""
title : selu.py
create : @tarickali 23/11/26
update : @tarickali 23/11/26
"""

import numpy as np
from core.activation import Activation

__all__ = ["SELU"]

ALPHA = 1.6732632423543772848170429916717
SCALE = 1.0507009873554804934193349852946


class SELU(Activation):
    def func(self, x: np.ndarray) -> np.ndarray:
        """ """

        return SCALE * (np.maximum(0, x) + np.minimum(0, ALPHA * (np.exp(x) - 1)))

    def grad(self, x: np.ndarray) -> np.ndarray:
        """ """

        return np.where(x >= 0, SCALE * np.ones_like(x), ALPHA * SCALE * np.exp(x))
