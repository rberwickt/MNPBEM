"""
Legendre-Gauss integration nodes and weights.

MATLAB: Misc/integration/lglnodes.m, Misc/integration/lgwt.m
"""

import numpy as np
from typing import Tuple


def lglnodes(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    MATLAB: Misc/integration/lglnodes.m

    Legendre-Gauss-Lobatto nodes and weights for integration.
    Adapted from Greg von Winckel.

    Parameters
    ----------
    n : int
        Number of integration points

    Returns
    -------
    x : ndarray, shape (n+1,)
        Integration points in interval [-1, 1]
    w : ndarray, shape (n+1,)
        Integration weights
    """
    n1 = n + 1
    # Chebyshev-Gauss-Lobatto nodes as initial guess
    x = np.cos(np.pi * np.arange(n + 1) / n)

    # Legendre Vandermonde Matrix
    p = np.zeros((n1, n1))

    # Newton-Raphson iteration
    xold = 2.0 * np.ones_like(x)

    while np.max(np.abs(x - xold)) > np.finfo(float).eps:
        xold = x.copy()
        p[:, 0] = 1.0
        p[:, 1] = x

        for k in range(2, n + 1):
            p[:, k] = ((2 * k - 1) * x * p[:, k - 1] - (k - 1) * p[:, k - 2]) / k

        x = xold - (x * p[:, n] - p[:, n - 1]) / (n1 * p[:, n])

    w = 2.0 / (n * n1 * p[:, n] ** 2)

    return x, w


def lgwt(n: int, a: float, b: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    MATLAB: Misc/integration/lgwt.m

    Legendre-Gauss nodes and weights on interval [a, b].
    Written by Greg von Winckel.

    Parameters
    ----------
    n : int
        Number of integration points (truncation order)
    a : float
        Lower bound
    b : float
        Upper bound

    Returns
    -------
    x : ndarray, shape (n,)
        Integration points
    w : ndarray, shape (n,)
        Integration weights
    """
    n_orig = n
    n = n - 1
    n1 = n + 1
    n2 = n + 2

    xu = np.linspace(-1, 1, n1)

    # initial guess
    y = np.cos((2 * np.arange(n1) + 1) * np.pi / (2 * n + 2)) + (0.27 / n1) * np.sin(np.pi * xu * n / n2)

    # Legendre-Gauss Vandermonde Matrix
    l_mat = np.zeros((n1, n2))

    y0 = 2.0 * np.ones_like(y)
    lp_vec = np.zeros(n1)

    while np.max(np.abs(y - y0)) > np.finfo(float).eps:
        l_mat[:, 0] = 1.0
        l_mat[:, 1] = y

        for k in range(2, n1 + 1):
            l_mat[:, k] = ((2 * k - 1) * y * l_mat[:, k - 1] - (k - 1) * l_mat[:, k - 2]) / k

        lp_vec = n2 * (l_mat[:, n1 - 1] - y * l_mat[:, n2 - 1]) / (1 - y ** 2)

        y0 = y.copy()
        y = y0 - l_mat[:, n2 - 1] / lp_vec

    # linear map from [-1,1] to [a,b]
    x = (a * (1 - y) + b * (1 + y)) / 2

    # weights
    w = (b - a) / ((1 - y ** 2) * lp_vec ** 2) * (n2 / n1) ** 2

    return x, w
