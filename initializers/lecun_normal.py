"""
title : lecun_normal.py
create : @tarickali 23/11/26
update : @tarickali 23/11/26
"""

import numpy as np

from core import Initializer
from core.types import Shape


class LecunNormal(Initializer):
    def init(self, shape: Shape = None) -> np.ndarray:
        std = np.sqrt(1.0 / shape[0])
        return np.random.normal(0.0, std, shape)