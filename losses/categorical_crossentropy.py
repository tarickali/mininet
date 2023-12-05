"""
title : categorical_crossentropy.py
create : @tarickali 23/11/27
update : @tarickali 23/12/01
"""

import numpy as np
from core import Loss


class CategoricalCrossentropy(Loss):
    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """ """

        eps = np.finfo(float).eps

        return -np.sum(y_true * np.log(y_pred + eps))

    def grad(self, y_true: np.ndarray, y_pred: np.ndarray, **params) -> np.ndarray:
        """ """

        return y_pred - y_true
