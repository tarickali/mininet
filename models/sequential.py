"""
title : sequential.py
create : @tarickali 23/11/27
update : @tarickali 23/12/05
"""

from typing import Any
import numpy as np
from core import Model, Module

__all__ = ["Sequential"]


class Sequential(Model):
    def __init__(self, modules: list[Module] = None) -> None:
        self.modules = modules

    def forward(self, X: np.ndarray) -> np.ndarray:
        for module in self.modules:
            X = module.forward(X)
        return X

    def backward(
        self, deltas: np.ndarray | list[np.ndarray]
    ) -> np.ndarray | list[np.ndarray]:
        for module in reversed(self.modules):
            deltas = module.backward(deltas)
        return deltas

    def zero_gradients(self) -> None:
        for module in self.modules:
            module.zero_gradients()

    def uncache(self) -> None:
        for module in self.modules:
            module.uncache()

    def append(self, module: Module) -> None:
        """Add a Module to the end of the Sequential Model.

        Parameters
        ----------
        module : Module

        """

        self.modules.append(module)

    @property
    def parameters(self) -> list[dict[str, Any]]:
        return [module.parameters for module in self.modules]

    @property
    def gradients(self) -> list[dict[str, Any]]:
        return [module.gradients for module in self.modules]
