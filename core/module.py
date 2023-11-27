"""
title : module.py
create : @tarickali 23/11/26
update : @tarickali 23/11/26
"""

from typing import Any
from collections import defaultdict
import numpy as np
from abc import ABC, abstractmethod


class Module(ABC):
    """ """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.parameters: dict[str, Any] = {}
        self.gradients: dict[str, Any] = {}
        self.hyperparameters: dict[str, Any] = {}
        self.cache: defaultdict[str, list[Any]] = defaultdict(list)

        self.trainable: bool = True

    @abstractmethod
    def forward(self, z: np.ndarray) -> np.ndarray:
        """ """

        raise NotImplementedError

    @abstractmethod
    def backward(self, grad: np.ndarray) -> np.ndarray:
        """ """

        raise NotImplementedError

    def freeze(self) -> None:
        """ """

        self.trainable = False

    def unfreeze(self) -> None:
        """ """

        self.trainable = True

    def update(self) -> None:
        """ """

    def summary(self) -> dict[str, Any]:
        """ """

        return {
            "name": self.name,
            "parameters": self.parameters,
            "hyperparameters": self.hyperparameters,
        }
