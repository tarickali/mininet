"""
title : loss.py
create : @tarickali 23/11/26
update : @tarickali 23/12/05
"""

from abc import ABC, abstractmethod
import numpy as np


class Loss(ABC):
    @abstractmethod
    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Computes the loss value between the ground truth and predicted arrays.

        Parameters
        ----------
        y_true : np.ndarray
        y_pred : np.ndarray

        Returns
        -------
        float

        """

        raise NotImplementedError

    @abstractmethod
    def grad(self, y_true: np.ndarray, y_pred: np.ndarray, **params) -> np.ndarray:
        """Computes the gradient of the loss function w.r.t the predicted array.

        Parameters
        ----------
        y_true : np.ndarray
        y_pred : np.ndarray
        params : **dict[str, Any]

        Returns
        -------
        np.ndarray

        """

        raise NotImplementedError

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Computes the loss value between the ground truth and predicted arrays.

        Parameters
        ----------
        y_true : np.ndarray
        y_pred : np.ndarray

        Returns
        -------
        float

        """

        return self.loss(y_true, y_pred)
