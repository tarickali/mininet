"""
title : regression.py
create : @tarickali 23/11/27
update : @tarickali 23/11/29
"""

import numpy as np
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt

from modules.linear import Linear
from models.sequential import Sequential
from losses import SquaredError


def regression():
    X, y = make_regression(n_samples=200, n_features=10)
    y = y.reshape(-1, 1)

    # Normalize data
    X = (X - X.mean()) / X.std()
    y = (y - y.mean()) / y.std()

    ALPHA = 0.1
    EPOCHS = 300
    model = Sequential(
        [
            Linear(10, 16, activation="relu"),
            Linear(16, 16, activation="relu"),
            Linear(16, 1),
        ]
    )

    loss = SquaredError()

    history = []
    for e in range(EPOCHS):
        o = model.forward(X)
        l = loss.loss(y, o) / y.shape[0]
        history.append(l)

        model.zero_gradients()
        delta = loss.grad(y, o)
        model.backward(delta)

        for param, grad in zip(model.parameters, model.gradients):
            param["W"] -= ALPHA * grad["W"]
            param["b"] -= ALPHA * grad["b"]

        print(f"Epoch {e} -- loss {l}")

    pred = model.forward(X)

    print(np.abs(pred - y))
    # plt.plot(history)
    # plt.show()
