"""
title : model.py
create : @tarickali 23/11/27
update : @tarickali 23/12/05
"""

from typing import Any, Container
from abc import ABC, abstractmethod
import numpy as np


class Model(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, X: np.ndarray) -> np.ndarray:
        """Computes the forward pass of the Model on input X.

        Parameters
        ----------
        X : np.ndarray

        Returns
        -------
        np.ndarray

        """

        raise NotImplementedError

    @abstractmethod
    def backward(
        self, deltas: np.ndarray | list[np.ndarray]
    ) -> np.ndarray | list[np.ndarray]:
        """Computes the backward pass of the Model given upstream deltas.

        Parameters
        ----------
        deltas : np.ndarray | list[np.ndarray]

        Returns
        -------
        np.ndarray | list[np.ndarray]

        """

        raise NotImplementedError

    @abstractmethod
    def zero_gradients(self) -> None:
        """Clear the gradients for each parameter in the Model."""

        raise NotImplementedError

    @abstractmethod
    def uncache(self) -> None:
        """Remove the cached values in the Model."""

        raise NotImplementedError

    @property
    @abstractmethod
    def parameters(self) -> Container[dict[str, Any]]:
        """Get all the parameters of the Model.

        Returns
        -------
        Container[dict[str, Any]]

        """

        raise NotImplementedError

    @property
    @abstractmethod
    def gradients(self) -> Container[dict[str, Any]]:
        """Get all the gradients of the Model.

        Returns
        -------
        Container[dict[str, Any]]

        """

        raise NotImplementedError
