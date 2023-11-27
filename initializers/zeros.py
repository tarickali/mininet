"""
title : zeros.py
create : @tarickali 23/11/26
update : @tarickali 23/11/26
"""

import numpy as np

from core import Initializer
from core.types import Shape


class Zeros(Initializer):
    def init(self, shape: Shape = None) -> np.ndarray:
        return np.zeros(shape)
