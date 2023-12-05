"""
title : squared_error.py
create : @tarickali 23/11/27
update : @tarickali 23/12/05
"""

import numpy as np
from core import Loss


class SquaredError(Loss):
    """SquaredError Loss

    Computes the squared error between y_true and y_pred given by:
    `sum((y_true - y_pred)**2) / 2.0`

    NOTE: This `Loss` computes the sum of the individual example errors
    and not the mean.

    """

    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.linalg.norm(y_true - y_pred) / 2.0

    def grad(self, y_true: np.ndarray, y_pred: np.ndarray, **params) -> np.ndarray:
        return y_pred - y_true
