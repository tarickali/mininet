"""
title : rmsprop.py
create : @tarickali 23/12/06
update : @tarickali 23/12/13
"""

from typing import Any
import numpy as np
from core import Optimizer


class RMSprop(Optimizer):
    def __init__(
        self,
        parameters: list[dict[str, Any]],
        learning_rate: float = 0.01,
        alpha: float = 0.99,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        centered: bool = False,
        eps: float = 1e-10,
        maximize: bool = False,
    ) -> None:
        super().__init__(parameters, learning_rate)
        self.alpha = alpha
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.centered = centered
        self.eps = eps
        self.maximize = maximize

        self.cache = [
            {"square_average": {}, "buffer": {}, "g_av": {}}
            for _ in range(len(self.parameters))
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

                # Initialize the cache values
                if t == 1:
                    cache["square_average"][key] = np.zeros_like(g)
                    cache["buffer"][key] = np.zeros_like(g)
                    cache["g_av"][key] = np.zeros_like(g)

                cache["square_average"][key] = (
                    self.alpha * cache["square_average"][key]
                    + (1 - self.alpha) * g**2
                )
                v_hat = cache["square_average"][key]

                if self.centered:
                    cache["g_av"][key] = (
                        cache["g_av"][key] * self.alpha + (1 - self.alpha) * g
                    )
                    v_hat = v_hat - cache["g_av"][key] ** 2

                if self.momentum > 0:
                    cache["buffer"][key] = self.momentum * cache["buffer"][key] + g / (
                        np.sqrt(v_hat) + self.eps
                    )
                    param[key] = param[key] - self.learning_rate * cache["buffer"][key]
                else:
                    param[key] = param[key] - self.learning_rate * g / (
                        np.sqrt(v_hat) + self.eps
                    )

        self.increment()

    def reset(self) -> None:
        super().reset()
        self.cache = [
            {"square_average": {}, "buffer": {}, "g_av": {}}
            for _ in range(len(self.parameters))
        ]
