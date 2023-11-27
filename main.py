"""
title : main.py
create : @tarickali 23/11/26
update : @tarickali 23/11/26
"""

import numpy as np
from modules.linear import Linear
from initializers import *


def main():
    # X = np.array([[1, 2, 3], [4, 5, 6]])

    # f = Linear(3, 6)

    # Z = f.forward(X)

    # print(Z)

    # x = np.array([[0, 1, 2], [2, 3, 4], [12, 3, 4]])
    # print(x.shape)
    # std = np.sqrt(2.0 / x.shape[0])
    # he = np.random.normal(0.0, std, x.shape)
    # print(he)

    x = np.array([[0, 1, 2], [2, 3, 4], [12, 3, 4]])
    initializer = RandomUniform()
    y = initializer(x.shape)
    print(y)


if __name__ == "__main__":
    main()
