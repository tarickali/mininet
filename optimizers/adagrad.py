"""
title : adagrad.py
create : @tarickali 23/12/06
update : @tarickali 23/12/06
"""

from typing import Any
import numpy as np
from core import Optimizer


class Adagrad(Optimizer):
    def __init__(
        self,
        parameters: list[dict[str, Any]],
        learning_rate: float = 0.01,
        learning_rate_decay: float = 0.0,
        weight_decay: float = 0.0,
        initial_accumulator_value: float = 0.0,
        eps: float = 1e-10,
        maximize: bool = False,
    ) -> None:
        super().__init__(parameters, learning_rate)
        self.learning_rate_decay = learning_rate_decay
        self.weight_decay = weight_decay
        self.initial_accumulator_value = initial_accumulator_value
        self.eps = eps
        self.maximize = maximize

        self.cache = [{"sum": {}} for _ in range(len(self.parameters))]

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
                lr = self.learning_rate / (1 + (t - 1) * self.learning_rate_decay)

                if t == 1:
                    cache["sum"][key] = np.full_like(
                        param[key], self.initial_accumulator_value
                    )
                cache["sum"][key] = cache["sum"][key] + g**2

                assert g.shape == cache["sum"][key].shape == param[key].shape

                param[key] = param[key] - lr * g / (cache["sum"][key] ** 0.5 + self.eps)

        self.increment()

    def reset(self) -> None:
        super().reset()
        self.cache = [{"sum": {}} for _ in range(len(self.parameters))]
