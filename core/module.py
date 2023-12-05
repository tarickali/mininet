"""
title : module.py
create : @tarickali 23/11/26
update : @tarickali 23/11/29
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
        """ """

        raise NotImplementedError

    @abstractmethod
    def backward(
        self, deltas: np.ndarray | list[np.ndarray]
    ) -> np.ndarray | list[np.ndarray]:
        """ """

        raise NotImplementedError

    def zero_gradients(self) -> None:
        """ """

        for param, grad in self.gradients.items():
            self.gradients[param] = np.zeros_like(grad)

    def uncache(self) -> None:
        """ """

        self.cache.clear()

    def update_parameters(self) -> None:
        """ """

    def freeze(self) -> None:
        """ """

        self.trainable = False

    def unfreeze(self) -> None:
        """ """

        self.trainable = True

    def summary(self) -> dict[str, Any]:
        """ """

        return {
            "name": self.name,
            "parameters": self.parameters,
            "hyperparameters": self.hyperparameters,
        }

    @property
    @abstractmethod
    def hyperparameters(self) -> dict[str, Any]:
        raise NotImplementedError
