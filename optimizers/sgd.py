"""
title : sgd.py
create : @tarickali 23/12/05
update : @tarickali 23/12/05
"""

from typing import Any
from core import Optimizer


class SGD(Optimizer):
    def __init__(
        self,
        parameters: list[dict[str, Any]],
        learning_rate: float = 0.01,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        dampening: float = 0.0,
        nesterov: bool = False,
        maximize: bool = False,
    ) -> None:
        super().__init__(parameters, learning_rate)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.dampening = dampening
        self.nesterov = nesterov
        self.maximize = maximize

        self.velocities = [{} for _ in range(len(self.parameters))]

    def update(self, gradients: list[dict[str, Any]]) -> None:
        if len(self.parameters) != len(gradients):
            raise ValueError(
                "The amount of gradients given do not match the parameters."
            )

        L = len(self.parameters)
        for l in range(L):
            if self.parameters[l].keys() != gradients[l].keys():
                raise ValueError("The gradient and parameter keys do not match.")

            param = self.parameters[l]
            grad = gradients[l]
            velocity = self.velocities[l]

            for key in param:
                g = grad[key] + self.weight_decay * param[key]

                if self.momentum != 0.0:
                    if self.time >= 1:
                        velocity[key] = (
                            self.momentum * velocity[key] + self.dampening * g
                        )
                    else:
                        velocity[key] = g

                    if self.nesterov:
                        g = g + self.momentum * velocity[key]
                    else:
                        g = velocity[key]

                if self.maximize:
                    param[key] = param[key] + self.learning_rate * g
                else:
                    param[key] = param[key] - self.learning_rate * g

        self.increment()

    def reset(self) -> None:
        super().reset()
        self.velocities = [{} for _ in range(len(self.parameters))]
