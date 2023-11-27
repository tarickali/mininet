"""
title : activation.py
create : @tarickali 23/11/26
update : @tarickali 23/11/26
"""

from abc import ABC, abstractmethod
import numpy as np


class Activation(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def func(self, x: np.ndarray) -> np.ndarray:
        """ """

        raise NotImplementedError

    @abstractmethod
    def grad(self, x: np.ndarray) -> np.ndarray:
        """ """

        raise NotImplementedError

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """ """

        return self.func(x)
