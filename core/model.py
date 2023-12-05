"""
title : model.py
create : @tarickali 23/11/27
update : @tarickali 23/11/27
"""

from typing import Any, Container
from abc import ABC, abstractmethod
import numpy as np


class Model(ABC):
    def __init__(self) -> None:
        super().__init__()

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

    @abstractmethod
    def zero_gradients(self) -> None:
        """ """

        raise NotImplementedError

    @abstractmethod
    def uncache(self) -> None:
        """ """

        raise NotImplementedError

    @property
    @abstractmethod
    def parameters(self) -> Container[dict[str, Any]]:
        raise NotImplementedError

    @property
    @abstractmethod
    def gradients(self) -> Container[dict[str, Any]]:
        raise NotImplementedError
