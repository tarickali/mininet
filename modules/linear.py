"""
title : linear.py
create : @tarickali 23/11/26
update : @tarickali 23/12/05
"""

from typing import Any
import numpy as np

from core.module import Module
from factories import activation_factory, initializer_factory


class Linear(Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        weight_initializer: str | dict[str, Any] = "xavier_normal",
        bias_initializer: str | dict[str, Any] = "zeros",
        activation: str | dict[str, Any] = None,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight_initializer = initializer_factory(weight_initializer)
        self.bias_initializer = initializer_factory(bias_initializer)
        self.activation = activation_factory(activation)

        # Initialize parameters
        self.init_parameters()

    def init_parameters(self) -> None:
        """Initialize the parameters of the Module."""

        self.parameters = {
            "W": self.weight_initializer((self.input_dim, self.output_dim)),
            "b": self.bias_initializer((self.output_dim,)),
        }
        self.gradients = {
            "W": np.zeros_like(self.parameters["W"]),
            "b": np.zeros_like(self.parameters["b"]),
        }

        assert self.parameters["W"].shape == (self.input_dim, self.output_dim)
        assert self.parameters["b"].shape == (self.output_dim,)

    def forward(self, X: np.ndarray) -> np.ndarray:
        # Get Module weights and bias
        W, b = self.parameters["W"], self.parameters["b"]

        # Compute linear transformation
        Z = X @ W + b
        assert Z.shape == (X.shape[0], self.output_dim)

        # Compute activation
        A = self.activation(Z)

        # Store input, linear, and non-linear arrays
        self.cache["X"].append(X)
        self.cache["Z"].append(Z)

        return A

    def backward(
        self, deltas: np.ndarray | list[np.ndarray]
    ) -> np.ndarray | list[np.ndarray]:
        assert self.trainable, "Module is not trainable."

        if not isinstance(deltas, list):
            deltas = [deltas]

        W = self.parameters["W"]
        Xs, Zs = self.cache["X"], self.cache["Z"]
        dXs = []
        m = Xs[0].shape[0]
        for delta, X, Z in zip(deltas, Xs, Zs):
            dZ = delta * self.activation.grad(Z)
            assert dZ.shape == (m, self.output_dim)

            dW = X.T @ dZ / m
            assert dW.shape == (self.input_dim, self.output_dim)

            db = np.mean(dZ, axis=0)
            assert db.shape == (self.output_dim,)

            dX = dZ @ W.T
            assert dX.shape == (m, self.input_dim)

            # NOTE: if scaling by m is done, need to rescale for the different sizes of accumulated gradients
            self.gradients["W"] += dW
            self.gradients["b"] += db
            dXs.append(dX)

        return dXs if len(dXs) > 1 else dXs[0]

    @property
    def hyperparameters(self) -> dict[str, Any]:
        return {
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "weight_initializer": self.weight_initializer,
            "bias_initializer": self.bias_initializer,
            "activation": self.activation,
        }
