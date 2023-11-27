"""
title : elu.py
create : @tarickali 23/11/26
update : @tarickali 23/11/26
"""

import numpy as np
from core.activation import Activation


class ELU(Activation):
    def __init__(self, alpha: float) -> None:
        super().__init__()
        self.alpha = alpha

    def func(self, x: np.ndarray) -> np.ndarray:
        """ """

        return np.where(x >= 0, x, self.alpha * (np.exp(x) - 1))

    def grad(self, x: np.ndarray) -> np.ndarray:
        """ """

        return np.where(x >= 0, np.ones_like(x), self.alpha * np.exp(x))
