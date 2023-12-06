"""
title : optimizer.py
create : @tarickali 23/12/05
update : @tarickali 23/12/05
"""

from typing import Any
from abc import ABC, abstractmethod


class Optimizer(ABC):
    def __init__(self, parameters: list[dict[str, Any]], learning_rate: float) -> None:
        super().__init__()
        self.parameters = parameters
        self.learning_rate = learning_rate
        self.time = 0

    def increment(self) -> None:
        self.time += 1

    def reset(self) -> None:
        self.time = 0

    @abstractmethod
    def update(self, gradients: list[dict[str, Any]]) -> None:
        """ """

        raise NotImplementedError
