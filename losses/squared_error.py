"""
title : squared_error.py
create : @tarickali 23/11/27
update : @tarickali 23/11/27
"""

import numpy as np
from core import Loss


class SquaredError(Loss):
    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """ """

        return np.linalg.norm(y_true - y_pred) / 2.0

    def grad(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """ """

        return y_pred - y_true
