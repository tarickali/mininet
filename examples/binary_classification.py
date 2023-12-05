"""
title : binary_classification.py
create : @tarickali 23/11/27
update : @tarickali 23/11/27
"""

import numpy as np
from sklearn.datasets import make_circles, make_moons
import matplotlib.pyplot as plt

from modules import Linear
from models.sequential import Sequential
from activations import Sigmoid
from losses import BinaryCrossentropy


def binary_classification():
    X, y = make_circles(256, random_state=42)
    y = y.reshape(-1, 1)

    # plt.scatter(X[:, 0], X[:, 1], c=y)
    # plt.show()

    ALPHA = 0.03
    EPOCHS = 1000

    model = Sequential(
        [
            Linear(2, 64, activation="relu"),
            Linear(64, 64, activation="relu"),
            Linear(64, 1, activation="sigmoid"),
        ]
    )

    loss = BinaryCrossentropy(logits=False)

    history = []
    for e in range(EPOCHS):
        o = model.forward(X)
        # o = Sigmoid()(o)
        l = loss.loss(y, o) / y.shape[0]
        history.append(l)

        model.zero_gradients()
        delta = loss.grad(y, o)
        model.backward(delta)
        model.uncache()

        for param, grad in zip(model.parameters, model.gradients):
            param["W"] -= ALPHA * grad["W"]
            param["b"] -= ALPHA * grad["b"]

        print(f"Epoch {e} -- loss {l}")

    # o = np.round(Sigmoid()(model.forward(X)))
    o = np.round(model.forward(X))
    print(np.mean(o == y))

    plt.scatter(X[:, 0], X[:, 1], c=o)
    plt.show()
    plt.plot(history)
    plt.show()
