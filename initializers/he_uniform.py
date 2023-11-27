"""
title : he_uniform.py
create : @tarickali 23/11/26
update : @tarickali 23/11/26
"""

import numpy as np

from core import Initializer
from core.types import Shape


class HeUniform(Initializer):
    def init(self, shape: Shape = None) -> np.ndarray:
        limit = np.sqrt(6.0 / shape[0])
        return np.random.uniform(-limit, limit, shape)
