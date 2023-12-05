"""
title : constant.py
create : @tarickali 23/11/26
update : @tarickali 23/12/05
"""

import numpy as np

from core import Initializer
from core.types import Shape


class Constant(Initializer):
    """Constant Initializer

    Initializes an array of a given shape with a constant value.

    """

    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value

    def init(self, shape: Shape = None) -> np.ndarray:
        return np.full(shape, self.value)
