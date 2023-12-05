"""
title : activation.py
create : @tarickali 23/11/26
update : @tarickali 23/12/05
"""

from abc import ABC, abstractmethod
import numpy as np


class Activation(ABC):
    @abstractmethod
    def func(self, x: np.ndarray) -> np.ndarray:
        """Compute the output of the activation function of the input x.

        Parameters
        ----------
        x : np.ndarray

        Returns
        -------
        np.ndarray

        """

        raise NotImplementedError

    @abstractmethod
    def grad(self, x: np.ndarray) -> np.ndarray:
        """Compute the gradient of the activation function w.r.t the input x.

        Parameters
        ----------
        x : np.ndarray

        Returns
        -------
        np.ndarray

        """

        raise NotImplementedError

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Compute the output of the activation function of the input x.

        Parameters
        ----------
        x : np.ndarray

        Returns
        -------
        np.ndarray

        """

        return self.func(x)
