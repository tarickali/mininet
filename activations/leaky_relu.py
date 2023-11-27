"""
title : leaky_relu.py
create : @tarickali 23/11/26
update : @tarickali 23/11/26
"""

import numpy as np
from core.activation import Activation


class LeakyReLU(Activation):
    def __init__(self, alpha: float = 0.0) -> None:
        super().__init__()
        self.alpha = alpha

    def func(self, x: np.ndarray) -> np.ndarray:
        """ """

        return np.maximum(0, x) + np.minimum(0, x) * self.alpha

    def grad(self, x: np.ndarray) -> np.ndarray:
        """ """

        return (x > 0).astype(x.dtype) + (x <= 0).astype(x.dtype) * self.alpha
