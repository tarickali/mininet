"""
title : random_uniform.py
create : @tarickali 23/11/26
update : @tarickali 23/11/26
"""

import numpy as np

from core import Initializer
from core.types import Shape


class RandomUniform(Initializer):
    def __init__(self, low: float = 0.0, high: float = 1.0) -> None:
        super().__init__()
        self.low = low
        self.high = high

    def init(self, shape: Shape = None) -> np.ndarray:
        return np.random.uniform(self.low, self.high, shape)
