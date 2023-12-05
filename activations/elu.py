"""
title : elu.py
create : @tarickali 23/11/26
update : @tarickali 23/12/05
"""

import numpy as np
from core.activation import Activation


class ELU(Activation):
    """ELU Activation

    Parameterized by `alpha` [float], computes the function
    ```
    f(x) = {
        x : x >= 0,
        alpha * (exp(x) - 1) : x < 0
    }
    ```.

    """

    def __init__(self, alpha: float) -> None:
        super().__init__()
        self.alpha = alpha

    def func(self, x: np.ndarray) -> np.ndarray:
        """ """

        return np.where(x >= 0, x, self.alpha * (np.exp(x) - 1))

    def grad(self, x: np.ndarray) -> np.ndarray:
        """ """

        return np.where(x >= 0, np.ones_like(x), self.alpha * np.exp(x))
