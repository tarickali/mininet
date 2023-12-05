"""
title : module.py
create : @tarickali 23/11/26
update : @tarickali 23/12/05
"""

from typing import Any
from collections import defaultdict
import numpy as np
from abc import ABC, abstractmethod


class Module(ABC):
    """ """

    def __init__(self) -> None:
        super().__init__()
        self.parameters: dict[str, Any] = {}
        self.gradients: dict[str, Any] = {}
        self.cache: defaultdict[str, list[Any]] = defaultdict(list)
        self.trainable: bool = True

    @abstractmethod
    def forward(self, X: np.ndarray) -> np.ndarray:
        """Computes the forward pass of the Module on input X.

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
        """Computes the backward pass of the Module given upstream deltas.

        Parameters
        ----------
        deltas : np.ndarray | list[np.ndarray]

        Returns
        -------
        np.ndarray | list[np.ndarray]

        """

        raise NotImplementedError

    def zero_gradients(self) -> None:
        """Clear the gradients for each parameter in the Module."""

        for param, grad in self.gradients.items():
            self.gradients[param] = np.zeros_like(grad)

    def uncache(self) -> None:
        """Remove the cached values in the Module."""

        self.cache.clear()

    def freeze(self) -> None:
        """Set the Module to be untrainable."""

        self.trainable = False

    def unfreeze(self) -> None:
        """Set the Module to be trainable."""

        self.trainable = True

    def summary(self) -> dict[str, Any]:
        """Get a summary of the Module.

        Returns
        -------
        dict[str, Any]

        """

        return {
            "name": self.name,
            "parameters": self.parameters,
            "hyperparameters": self.hyperparameters,
        }

    @property
    @abstractmethod
    def hyperparameters(self) -> dict[str, Any]:
        """Get the hyperparameters of the Module.

        Returns
        -------
        dict[str, Any]

        """

        raise NotImplementedError
