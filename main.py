"""
title : main.py
create : @tarickali 23/11/26
update : @tarickali 23/11/29
"""

from examples import binary_classification, regression


def main():
    binary_classification()
    # regression()
    # X, y = make_classification(256, n_informative=5, n_classes=5)
    # print(X.shape, y.shape)
    # y = one_hot(y, 5)
    # print(y)

    # X, y = make_regression(256, n_features=10)
    # y = y.reshape(-1, 1)

    # o = model.forward(X)
    # delta = loss.grad(y, o)
    # model.backward(delta)
    # print(model.modules[0].gradients)
    # model.modules[0].zero_gradients()
    # print(model.modules[0].gradients)

    # print(delta.shape)
    # print(l.shape)
    # print(l)

    # f = Linear(3, 6, weight_initializer="he_uniform", activation="leaky_relu")
    # A = f.forward(X)
    # dX = f.backward(A)
    # print(A)
    # print(dX)


if __name__ == "__main__":
    main()
