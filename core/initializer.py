"""
title : initializer.py
create : @tarickali 23/11/26
update : @tarickali 23/11/26
"""

from abc import ABC, abstractmethod
import numpy as np

from core.types import Shape


class Initializer(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def init(self, shape: Shape = None) -> np.ndarray:
        """ """

        raise NotImplementedError

    def __call__(self, shape: Shape = None) -> np.ndarray:
        return self.init(shape)
