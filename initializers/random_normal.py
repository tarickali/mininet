"""
title : random_normal.py
create : @tarickali 23/11/26
update : @tarickali 23/12/05
"""

import numpy as np

from core import Initializer
from core.types import Shape


class RandomNormal(Initializer):
    """RandomNormal Initializer

    Initializes an array of a given shape with values sampled drawn from
    `N(mu, std)`, where `mu`, `std` are the given mean, standard deviations,
    respectively.

    """

    def __init__(self, mean: float = 0.0, std: float = 1.0) -> None:
        super().__init__()
        self.mean = mean
        self.std = std

    def init(self, shape: Shape = None) -> np.ndarray:
        return np.random.normal(self.mean, self.std, shape)
