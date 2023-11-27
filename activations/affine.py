"""
title : affine.py
create : @tarickali 23/11/26
update : @tarickali 23/11/26
"""

import numpy as np
from core.activation import Activation


class Affine(Activation):
    def __init__(self, slope: float, intercept: float) -> None:
        super().__init__()
        self.slope = slope
        self.intercept = intercept

    def func(self, x: np.ndarray) -> np.ndarray:
        """ """

        return self.slope * x + self.intercept

    def grad(self, x: np.ndarray) -> np.ndarray:
        """ """

        return np.full_like(x, self.slope)
