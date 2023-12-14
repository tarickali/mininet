"""
title : adadelta.py
create : @tarickali 23/12/06
update : @tarickali 23/12/06
"""

from typing import Any
import numpy as np
from core import Optimizer


class Adadelta(Optimizer):
    def __init__(
        self,
        parameters: list[dict[str, Any]],
        learning_rate: float = 1.0,
        rho: float = 0.9,
        weight_decay: float = 0.0,
        eps: float = 1e-10,
        maximize: bool = False,
    ) -> None:
        super().__init__(parameters, learning_rate)
        self.rho = rho
        self.weight_decay = weight_decay
        self.eps = eps
        self.maximize = maximize

        self.cache = [
            {"average": {}, "accumulator": {}} for _ in range(len(self.parameters))
        ]

    def update(self, gradients: list[dict[str, Any]]) -> None:
        if len(self.parameters) != len(gradients):
            raise ValueError(
                "The amount of gradients given do not match the parameters."
            )

        L = len(self.parameters)
        t = self.time + 1
        for l in range(L):
            if self.parameters[l].keys() != gradients[l].keys():
                raise ValueError("The gradient and parameter keys do not match.")

            grad = gradients[l]
            param = self.parameters[l]
            cache = self.cache[l]

            for key in param:
                g = grad[key] + self.weight_decay * param[key]

                if t == 1:
                    cache["average"][key] = (1 - self.rho) * g**2
                    delta = ((self.eps) / (cache["average"][key] + self.eps)) ** 0.5 * g
                    cache["accumulator"][key] = (1 - self.rho) * delta**2
                else:
                    cache["average"][key] = (
                        self.rho * cache["average"][key] + (1 - self.rho) * g**2
                    )
                    delta = (
                        (cache["accumulator"][key] + self.eps)
                        / (cache["average"][key] + self.eps)
                    ) ** 0.5 * g
                    cache["accumulator"][key] = (
                        self.rho * cache["accumulator"][key]
                        + (1 - self.rho) * delta**2
                    )

                if self.maximize:
                    param[key] = param[key] + self.learning_rate * delta
                else:
                    param[key] = param[key] - self.learning_rate * delta

        self.increment()

    def reset(self) -> None:
        super().reset()
        self.cache = [{"sum": {}} for _ in range(len(self.parameters))]
