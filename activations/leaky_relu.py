"""
title : leaky_relu.py
create : @tarickali 23/11/26
update : @tarickali 23/12/05
"""

import numpy as np
from core.activation import Activation


class LeakyReLU(Activation):
    """LeakyReLU Activation

    Parameterized by `alpha` [float], computes the function
    ```
    f(x) = {
        x : x >= 0,
        alpha * x : x < 0
    }
    ```.

    """

    def __init__(self, alpha: float = 0.0) -> None:
        super().__init__()
        self.alpha = alpha

    def func(self, x: np.ndarray) -> np.ndarray:
        """ """

        return np.maximum(0, x) + np.minimum(0, x) * self.alpha

    def grad(self, x: np.ndarray) -> np.ndarray:
        """ """

        return (x > 0).astype(x.dtype) + (x <= 0).astype(x.dtype) * self.alpha
