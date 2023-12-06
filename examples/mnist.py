"""
title : mnist.py
create : @tarickali 23/12/05
update : @tarickali 23/12/05
"""

import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from models import Sequential
from modules import Linear
from optimizers import SGD
from activations import Softmax
from losses import CategoricalCrossentropy


__all__ = ["mnist_driver"]


def generate_data() -> tuple[np.ndarray, np.ndarray]:
    """Generate data for the MNIST problem.

    Returns
    -------
    tuple[np.ndarray, np.ndarray] : X @ (n, 28, 28), y @ (n, 10)

    """

    # Read in MNIST training dataframe
    train_df = pd.read_csv("examples/data/mnist/train.csv")

    # Get pixel and label information from dataframe
    pixels = train_df.drop("label", axis=1).to_numpy()
    labels = train_df["label"].to_numpy()

    # Scale pixel data to be between 0.0 - 1.0
    X = pixels.reshape((-1, 784)) / 255.0

    # Create one hot vector of labels
    y = one_hot(labels)

    return X, y


def one_hot(x: np.ndarray, k: int = 10) -> np.ndarray:
    """Create a one-hot array of input array with k classes.

    Parameters
    ----------
    x : np.ndarray @ (n, 1)
    k : int = 10
        Number of classes in the one-hot array.

    Returns
    -------
    np.ndarray @ (n, k)

    """

    n = x.shape[0]
    o = np.zeros((n, k))
    o[np.arange(n), x] = 1
    return o


def get_batches(
    X: np.ndarray, y: np.ndarray, m: int = 32
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create batches of (X, y) data.

    Parameters
    ----------
    X : np.ndarray
    y : np.ndarray
    m : int = 32

    Returns
    -------
    list[tuple[np.ndarray, np.ndarray]]

    """

    n = X.shape[0]
    batches = []

    # Loop for creating batches of size m
    for i in range(n // m):
        a, b = i * m, (i + 1) * m
        batches.append((X[a:b], y[a:b]))

    # Create an extra match of size < m for leftover data
    if b != n:
        batches.append((X[b:], y[b:]))

    return batches


def mnist_driver():
    # Get MNIST data
    X, y = generate_data()

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    ALPHA = 0.01
    EPOCHS = 5

    model = Sequential(
        [
            Linear(784, 512, activation="relu"),
            Linear(512, 512, activation="relu"),
            Linear(512, 512, activation="relu"),
            Linear(512, 10, activation="identity"),
        ]
    )

    loss_fn = CategoricalCrossentropy(logits=True)
    optimizer = SGD(model.parameters, learning_rate=ALPHA, momentum=0.9, dampening=0.1)
    softmax = Softmax()

    n = X_train.shape[0]
    m = 32
    k = 10
    history = []
    for e in range(EPOCHS):
        epoch_loss = 0.0
        epoch_acc = 0.0
        for Xb, yb in get_batches(X_train, y_train, m):
            pred = model.forward(Xb)
            loss = loss_fn(yb, pred)

            model.zero_gradients()
            delta = loss_fn.grad(yb, pred)
            model.backward(delta)
            optimizer.update(model.gradients)

            prob = one_hot(np.argmax(softmax(pred), axis=1), k=10)
            acc = np.sum(yb == prob) / k

            epoch_loss += loss
            epoch_acc += acc

        history.append({"epoch": e, "loss": epoch_loss / n, "acc": epoch_acc / n})
        print(history[-1])

    # Plot loss and accuracy curves
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 7))
    losses = [epoch["loss"] for epoch in history]
    accs = [epoch["acc"] for epoch in history]
    axes[0].plot(losses)
    axes[0].set(title="Loss Curve", xlabel="Epochs", ylabel="Loss")
    axes[1].plot(accs)
    axes[1].set(title="Accuracy Curve", xlabel="Epochs", ylabel="Accuracy")
    plt.tight_layout()
    plt.show()

    # Compute trained network's accuracy on training data
    y_train_pred = model.forward(X_train)
    y_train_pred = one_hot(np.argmax(softmax(y_train_pred), axis=1), k=10)
    train_acc = np.mean(y_train == y_train_pred)
    print(f"Train accuracy: {train_acc}")

    # Compute trained network's accuracy on test data
    y_test_pred = model.forward(X_test)
    y_test_pred = one_hot(np.argmax(softmax(y_test_pred), axis=1), k=10)
    test_acc = np.mean(y_test == y_test_pred)
    print(f"Test accuracy: {test_acc}")
