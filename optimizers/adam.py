"""
title : adam.py
create : @tarickali 23/12/05
update : @tarickali 23/12/05
"""

from typing import Any
import numpy as np
from core import Optimizer


class Adam(Optimizer):
    def __init__(
        self,
        parameters: list[dict[str, Any]],
        learning_rate: float = 0.001,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        amsgrad: bool = False,
        maximize: bool = False,
    ) -> None:
        super().__init__(parameters, learning_rate)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.eps = eps
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad
        self.maximize = maximize

        self.cache = [
            {"momentum": {}, "velocity": {}, "vhat_max": {}}
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
            # momentum = self.momentums[l]
            # velocity = self.velocities[l]
            # if self.amsgrad:
            #     vhat_max = self.vhat_maxes[l]

            for key in param:
                g = grad[key] if not self.maximize else -grad[key]
                g = g + self.weight_decay * param[key]

                if t > 1:
                    cache["momentum"][key] = (
                        self.beta_1 * cache["momentum"][key] + (1 - self.beta_1) * g
                    )
                    cache["velocity"][key] = (
                        self.beta_2 * cache["velocity"][key]
                        + (1 - self.beta_2) * g**2
                    )
                else:
                    cache["momentum"][key] = (1 - self.beta_1) * g
                    cache["velocity"][key] = (1 - self.beta_2) * g**2

                mhat = cache["momentum"][key] / (1 - self.beta_1**t)
                vhat = cache["velocity"][key] / (1 - self.beta_2**t)

                if self.amsgrad:
                    cache["vhat_max"][key] = np.maximum(
                        cache["vhat_max"].get(key, -np.inf), vhat
                    )  # NOTE: bug here?
                    param[key] = param[key] - self.learning_rate * mhat / (
                        np.sqrt(cache["vhat_max"][key]) + self.eps
                    )
                else:
                    param[key] = param[key] - self.learning_rate * mhat / (
                        np.sqrt(vhat) + self.eps
                    )

        self.increment()

    def reset(self) -> None:
        super().reset()
        self.momentums = [{} for _ in range(len(self.parameters))]
        self.velocities = [{} for _ in range(len(self.parameters))]
