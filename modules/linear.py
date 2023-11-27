"""
title : linear.py
create : @tarickali 23/11/26
update : @tarickali 23/11/26
"""

from typing import Any
import numpy as np

from core.module import Module
from factories import activation_factory


class Linear(Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        weight_initializer: str = None,
        bias_initializer: str = None,
        activation: str | dict[str, Any] = None,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.parameters = {
            "W": np.random.randn(input_dim, output_dim),
            "b": np.zeros(output_dim),
        }

        assert self.parameters["W"].shape == (input_dim, output_dim)
        assert self.parameters["b"].shape == (output_dim,)

        self.hyperparameters = {
            "input_dim": input_dim,
            "output_dim": output_dim,
            "weight_initializer": weight_initializer,
            "bias_initializer": bias_initializer,
            "activation": activation,
        }

    def init(self) -> None:
        """ """

    def forward(self, X: np.ndarray) -> np.ndarray:
        """

        Parameters
        ----------
        X : np.ndarray @ (m, n_in)

        Returns
        -------


        """

        # Get Module weights and bias
        W, b = self.parameters["W"], self.parameters["b"]

        # Compute linear transformation
        Z = X @ W + b

        # Compute activation
        # A = self.activation(Z)
        A = Z

        # Store input, linear, and non-linear arrays
        self.cache["X"].append(X)
        self.cache["Z"].append(Z)
        self.cache["A"].append(A)

        return A

    def backward(
        self, deltas: np.ndarray | list[np.ndarray]
    ) -> np.ndarray | list[np.ndarray]:
        """

        Parameters
        ----------
        deltas : np.ndarray | list[np.ndarray]

        Returns
        -------
        np.npdarray | list[np.ndarray]

        """

        assert self.trainable, "Module is not trainable."

        if not isinstance(deltas, list):
            deltas = [deltas]

        self.gradients["W"] = ...
        self.gradients["b"] = ...
