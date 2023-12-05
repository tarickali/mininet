"""
title : initializer.py
create : @tarickali 23/11/26
update : @tarickali 23/12/05
"""

from abc import ABC, abstractmethod
import numpy as np

from core.types import Shape


class Initializer(ABC):
    @abstractmethod
    def init(self, shape: Shape) -> np.ndarray:
        """Initialize an array with given shape using initialization scheme.

        Parameters
        ----------
        shape : Shape

        Returns
        -------
        np.ndarray

        """

        raise NotImplementedError

    def __call__(self, shape: Shape) -> np.ndarray:
        """Initialize an array with given shape using initialization scheme.

        Parameters
        ----------
        shape : Shape

        Returns
        -------
        np.ndarray

        """

        return self.init(shape)
