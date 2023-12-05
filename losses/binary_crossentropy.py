"""
title : binary_crossentropy.py
create : @tarickali 23/12/01
update : @tarickali 23/12/01
"""

import numpy as np
from core import Loss
from activations import Sigmoid


class BinaryCrossentropy(Loss):
    def __init__(self, logits: bool = True) -> None:
        super().__init__()
        self.logits = logits

    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """ """

        eps = np.finfo(float).eps
        if self.logits:
            y_pred = Sigmoid()(y_pred)

        return -np.sum(
            y_true * np.log(y_pred + eps) + (1 - y_true) * np.log(1 - y_pred)
        )

    def grad(self, y_true: np.ndarray, y_pred: np.ndarray, **params) -> np.ndarray:
        """ """

        if not self.logits:
            s = Sigmoid()(y_pred)
            return (y_pred - y_true) / (s * (1 - s))
        else:
            return y_pred - y_true
