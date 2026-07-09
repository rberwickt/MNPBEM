"""
Shape functions for triangular and quadrilateral elements.

MATLAB: +shape/@tri/, +shape/@quad/
"""

import numpy as np
from typing import Optional


class Tri(object):
    """
    Triangular shape element.

    MATLAB: +shape/@tri

    Parameters
    ----------
    node : int
        Number of nodes (3 or 6)

    Methods
    -------
    eval(x, y) -> ndarray
    deriv(x, y, key) -> ndarray
    """

    def __init__(self, node: int) -> None:
        self.node = node

    def eval(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        MATLAB: +shape/@tri/eval.m

        Evaluate triangular shape function.
        """
        x = np.asarray(x).ravel()
        y = np.asarray(y).ravel()
        z = 1 - x - y

        if self.node == 3:
            s = np.column_stack([x, y, z])
        elif self.node == 6:
            s = np.column_stack([
                x * (2 * x - 1),
                y * (2 * y - 1),
                z * (2 * z - 1),
                4 * x * y,
                4 * y * z,
                4 * z * x])
        else:
            raise ValueError('[error] Invalid <node> for Tri!')

        return s

    def deriv(self, x: np.ndarray, y: np.ndarray, key: str) -> np.ndarray:
        """
        MATLAB: +shape/@tri/deriv.m

        Derivative of triangular shape function.
        key: 'x', 'y', 'xx', 'yy', 'xy'
        """
        x = np.asarray(x).ravel()
        y = np.asarray(y).ravel()
        n = len(x)

        if self.node == 3:
            return self._deriv3(x, y, key, n)
        elif self.node == 6:
            return self._deriv6(x, y, key, n)
        else:
            raise ValueError('[error] Invalid <node> for Tri!')

    def _deriv3(self, x: np.ndarray, y: np.ndarray,
            key: str, n: int) -> np.ndarray:
        if key == 'x':
            sp = np.tile([1.0, 0.0, -1.0], (n, 1))
        elif key == 'y':
            sp = np.tile([0.0, 1.0, -1.0], (n, 1))
        else:
            sp = np.tile([0.0, 0.0, 0.0], (n, 1))
        return sp

    def _deriv6(self, x: np.ndarray, y: np.ndarray,
            key: str, n: int) -> np.ndarray:
        z = 1 - x - y

        if key == 'x':
            sp = np.column_stack([
                4 * x - 1, np.zeros(n), 1 - 4 * z,
                4 * y, -4 * y, 4 * (z - x)])
        elif key == 'y':
            sp = np.column_stack([
                np.zeros(n), 4 * y - 1, 1 - 4 * z,
                4 * x, 4 * (z - y), -4 * x])
        elif key == 'xx':
            sp = np.tile([4.0, 0.0, 4.0, 0.0, 0.0, -8.0], (n, 1))
        elif key == 'yy':
            sp = np.tile([0.0, 4.0, 4.0, 0.0, -8.0, 0.0], (n, 1))
        elif key == 'xy':
            sp = np.tile([0.0, 0.0, 4.0, 4.0, -4.0, -4.0], (n, 1))
        else:
            raise ValueError('[error] Invalid <key>!')

        return sp

    def __call__(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.eval(x, y)

    def __repr__(self) -> str:
        return 'Tri(node={})'.format(self.node)


class Quad(object):
    """
    Quadrilateral shape element.

    MATLAB: +shape/@quad

    Parameters
    ----------
    node : int
        Number of nodes (4 or 9)

    Methods
    -------
    eval(x, y) -> ndarray
    deriv(x, y, key) -> ndarray
    """

    def __init__(self, node: int) -> None:
        self.node = node

    def eval(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        MATLAB: +shape/@quad/eval.m

        Evaluate quadrilateral shape function.
        """
        x = np.asarray(x).ravel()
        y = np.asarray(y).ravel()

        if self.node == 4:
            s = 0.25 * np.column_stack([
                (1 - x) * (1 - y),
                (1 + x) * (1 - y),
                (1 + x) * (1 + y),
                (1 - x) * (1 + y)])
        elif self.node == 9:
            sx = np.column_stack([
                0.5 * x * (x - 1), 1 - x ** 2, 0.5 * x * (1 + x)])
            sy = np.column_stack([
                0.5 * y * (y - 1), 1 - y ** 2, 0.5 * y * (1 + y)])
            s = self._assemble(sx, sy)
        else:
            raise ValueError('[error] Invalid <node> for Quad!')

        return s

    def deriv(self, x: np.ndarray, y: np.ndarray, key: str) -> np.ndarray:
        """
        MATLAB: +shape/@quad/deriv.m

        Derivative of quadrilateral shape function.
        key: 'x', 'y', 'xx', 'yy', 'xy'
        """
        x = np.asarray(x).ravel()
        y = np.asarray(y).ravel()
        n = len(x)

        if self.node == 4:
            return self._deriv4(x, y, key, n)
        elif self.node == 9:
            return self._deriv9(x, y, key, n)
        else:
            raise ValueError('[error] Invalid <node> for Quad!')

    def _deriv4(self, x: np.ndarray, y: np.ndarray,
            key: str, n: int) -> np.ndarray:
        if key == 'x':
            sp = 0.25 * np.column_stack([
                -(1 - y), (1 - y), (1 + y), -(1 + y)])
        elif key == 'y':
            sp = 0.25 * np.column_stack([
                -(1 - x), -(1 + x), (1 + x), (1 - x)])
        elif key == 'xy':
            sp = np.tile(0.25 * np.array([1.0, -1.0, 1.0, -1.0]), (n, 1))
        else:
            sp = np.tile([0.0, 0.0, 0.0, 0.0], (n, 1))
        return sp

    def _deriv9(self, x: np.ndarray, y: np.ndarray,
            key: str, n: int) -> np.ndarray:
        sx0 = np.column_stack([
            0.5 * x * (x - 1), 1 - x ** 2, 0.5 * x * (1 + x)])
        sy0 = np.column_stack([
            0.5 * y * (y - 1), 1 - y ** 2, 0.5 * y * (1 + y)])

        sx1 = np.column_stack([x - 0.5, -2 * x, x + 0.5])
        sy1 = np.column_stack([y - 0.5, -2 * y, y + 0.5])

        sx2 = np.tile([1.0, -2.0, 1.0], (n, 1))
        sy2 = np.tile([1.0, -2.0, 1.0], (n, 1))

        if key == 'x':
            sp = self._assemble(sx1, sy0)
        elif key == 'y':
            sp = self._assemble(sx0, sy1)
        elif key == 'xx':
            sp = self._assemble(sx2, sy0)
        elif key == 'yy':
            sp = self._assemble(sx0, sy2)
        elif key == 'xy':
            sp = self._assemble(sx1, sy1)
        else:
            raise ValueError('[error] Invalid <key>!')

        return sp

    def _assemble(self, sx: np.ndarray, sy: np.ndarray) -> np.ndarray:
        """Assembly function for 9-node quad."""
        return np.column_stack([
            sx[:, 0] * sy[:, 0],
            sx[:, 2] * sy[:, 0],
            sx[:, 2] * sy[:, 2],
            sx[:, 0] * sy[:, 2],
            sx[:, 1] * sy[:, 0],
            sx[:, 2] * sy[:, 1],
            sx[:, 1] * sy[:, 2],
            sx[:, 0] * sy[:, 1],
            sx[:, 1] * sy[:, 1]])

    def __call__(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.eval(x, y)

    def __repr__(self) -> str:
        return 'Quad(node={})'.format(self.node)
