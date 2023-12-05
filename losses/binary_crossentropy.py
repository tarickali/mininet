"""
title : binary_crossentropy.py
create : @tarickali 23/12/01
update : @tarickali 23/12/05
"""

import numpy as np
from core import Loss
from activations import Sigmoid


class BinaryCrossentropy(Loss):
    """BinaryCrossentropy Loss

    Computes the crossentropy loss between binary arrays y_true and y_pred
    given by: `-sum(y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred))`.

    NOTE: This `Loss` can be used when y_pred are unactivated (logits) or
    are activated.

    NOTE: This `Loss` computes the sum of the individual example errors
    and not the mean.

    """

    def __init__(self, logits: bool = True) -> None:
        super().__init__()
        self.logits = logits

    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        eps = np.finfo(float).eps
        if self.logits:
            y_pred = Sigmoid()(y_pred)

        return -np.sum(
            y_true * np.log(y_pred + eps) + (1 - y_true) * np.log(1 - y_pred)
        )

    def grad(self, y_true: np.ndarray, y_pred: np.ndarray, **params) -> np.ndarray:
        if not self.logits:
            s = Sigmoid()(y_pred)
            return (y_pred - y_true) / (s * (1 - s))
        else:
            return y_pred - y_true
