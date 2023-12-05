"""
title : loss.py
create : @tarickali 23/11/26
update : @tarickali 23/11/27
"""

from abc import ABC, abstractmethod
import numpy as np


class Loss(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """ """

        raise NotImplementedError

    @abstractmethod
    def grad(self, y_true: np.ndarray, y_pred: np.ndarray, **params) -> np.ndarray:
        """ """

        raise NotImplementedError

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return self.loss(y_true, y_pred)
