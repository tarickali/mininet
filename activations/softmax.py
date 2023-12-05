"""
title : softmax.py
create : @tarickali 23/12/05
update : @tarickali 23/12/05
"""

import numpy as np
from core.activation import Activation


class Softmax(Activation):
    """Softmax Activation

    Computes the function `f(x) = exp(x) / sum(exp(x))`.

    NOTE: Returns the all ones matrix with shape of input z,
    since the categorical cross-entropy loss function L computes
    the appropriate gradient of L with respect to z.

    NOTE: The true gradient of softmax with respect to z is a Jacobian,
    and the code is given below:
    s = softmax(z)
    jacob = np.diag(s.flatten()) - np.outer(s, s)

    NOTE: It is important to note that this choice limits the use of
    the softmax activation to only the last layer of a neural network.

    """

    def func(self, x: np.ndarray) -> np.ndarray:
        e = np.exp(x)
        return e / np.sum(e, axis=1, keepdims=True)

    def grad(self, x: np.ndarray) -> np.ndarray:
        return np.ones_like(x)
