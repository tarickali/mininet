"""
title : categorical_crossentropy.py
create : @tarickali 23/11/27
update : @tarickali 23/12/05
"""

import numpy as np
from core import Loss
from activations import Softmax


class CategoricalCrossentropy(Loss):
    """BinaryCrossentropy Loss

    Computes the crossentropy loss between multiclass arrays y_true and y_pred
    given by: `-sum(y_true * log(y_pred))`.

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
            y_pred = Softmax()(y_pred)
        return -np.sum(y_true * np.log(y_pred + eps))

    def grad(self, y_true: np.ndarray, y_pred: np.ndarray, **params) -> np.ndarray:
        if self.logits:
            y_pred = Softmax()(y_pred)
        return y_pred - y_true
