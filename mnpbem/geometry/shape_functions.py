"""
Shape functions for boundary elements.

This module provides shape functions for:
- Triangular elements (3-node linear, 6-node quadratic)
- Quadrilateral elements (4-node bilinear, 9-node biquadratic)

MATLAB reference: /Misc/+shape/
"""

import numpy as np
from typing import Union, Literal


class TriangleShape(object):
    """
    Triangular shape element.

    Supports both linear (3-node) and quadratic (6-node) triangles.
    """

    def __init__(self, n_nodes: int = 3):
        """
        Initialize triangular shape element.

        Parameters
        ----------
        n_nodes : int
            Number of nodes (3 or 6)
            3: Linear triangle
            6: Quadratic triangle (curved boundaries)
        """
        if n_nodes not in [3, 6]:
            raise ValueError("n_nodes must be 3 or 6, got {}".format(n_nodes))
        self.n_nodes = n_nodes

    def __call__(self, x: Union[float, np.ndarray],
                 y: Union[float, np.ndarray]) -> np.ndarray:
        """
        Evaluate shape functions.

        Parameters
        ----------
        x, y : float or array_like
            Coordinates in unit triangle
            Valid range: x >= 0, y >= 0, x + y <= 1

        Returns
        -------
        s : np.ndarray, shape (n_points, n_nodes)
            Shape function values at each point
            For 3-node: [N1, N2, N3]
            For 6-node: [N1, N2, N3, N4, N5, N6]
        """
        x = np.atleast_1d(x).flatten()
        y = np.atleast_1d(y).flatten()
        z = 1 - x - y  # Third triangular coordinate

        if self.n_nodes == 3:
            # Linear shape functions
            # N1 = x, N2 = y, N3 = 1-x-y
            return np.column_stack([x, y, z])

        else:  # n_nodes == 6
            # Quadratic shape functions (Lagrange polynomials)
            # Corner nodes: N_i = ξ_i(2ξ_i - 1)
            # Mid-side nodes: N_i = 4ξ_jξ_k
            return np.column_stack([
                x * (2*x - 1),      # Node 1: corner at (1,0,0)
                y * (2*y - 1),      # Node 2: corner at (0,1,0)
                z * (2*z - 1),      # Node 3: corner at (0,0,1)
                4 * x * y,          # Node 4: midside 1-2
                4 * y * z,          # Node 5: midside 2-3
                4 * z * x           # Node 6: midside 3-1
            ])

    def x(self, xi: Union[float, np.ndarray],
          eta: Union[float, np.ndarray]) -> np.ndarray:
        """
        Derivative of shape functions with respect to x.

        Parameters
        ----------
        xi, eta : float or array_like
            Coordinates in unit triangle

        Returns
        -------
        dN_dx : np.ndarray, shape (n_points, n_nodes)
            Shape function derivatives ∂N/∂x
        """
        xi = np.atleast_1d(xi).flatten()
        eta = np.atleast_1d(eta).flatten()
        n_pts = len(xi)

        if self.n_nodes == 3:
            # Linear: derivatives are constant
            # dN1/dx = 1, dN2/dx = 0, dN3/dx = -1
            return np.tile([1.0, 0.0, -1.0], (n_pts, 1))

        else:  # n_nodes == 6
            z = 1 - xi - eta
            return np.column_stack([
                4*xi - 1,           # dN1/dx
                0*eta,              # dN2/dx = 0
                1 - 4*z,            # dN3/dx = 4*z - 1
                4*eta,              # dN4/dx
                -4*eta,             # dN5/dx
                4*(z - xi)          # dN6/dx
            ])

    def y(self, xi: Union[float, np.ndarray],
          eta: Union[float, np.ndarray]) -> np.ndarray:
        """
        Derivative of shape functions with respect to y.

        Returns
        -------
        dN_dy : np.ndarray, shape (n_points, n_nodes)
            Shape function derivatives ∂N/∂y
        """
        xi = np.atleast_1d(xi).flatten()
        eta = np.atleast_1d(eta).flatten()
        n_pts = len(xi)

        if self.n_nodes == 3:
            # dN1/dy = 0, dN2/dy = 1, dN3/dy = -1
            return np.tile([0.0, 1.0, -1.0], (n_pts, 1))

        else:  # n_nodes == 6
            z = 1 - xi - eta
            return np.column_stack([
                0*xi,               # dN1/dy = 0
                4*eta - 1,          # dN2/dy
                1 - 4*z,            # dN3/dy = 4*z - 1
                4*xi,               # dN4/dy
                4*(z - eta),        # dN5/dy
                -4*xi               # dN6/dy
            ])


class QuadShape(object):
    """
    Quadrilateral shape element.

    Supports both bilinear (4-node) and biquadratic (9-node) quads.
    """

    def __init__(self, n_nodes: int = 4):
        """
        Initialize quadrilateral shape element.

        Parameters
        ----------
        n_nodes : int
            Number of nodes (4 or 9)
            4: Bilinear quadrilateral
            9: Biquadratic quadrilateral (curved boundaries)
        """
        if n_nodes not in [4, 9]:
            raise ValueError("n_nodes must be 4 or 9, got {}".format(n_nodes))
        self.n_nodes = n_nodes

    def __call__(self, xi: Union[float, np.ndarray],
                 eta: Union[float, np.ndarray]) -> np.ndarray:
        """
        Evaluate shape functions.

        Parameters
        ----------
        xi, eta : float or array_like
            Coordinates in parametric space
            Valid range: -1 <= xi <= 1, -1 <= eta <= 1

        Returns
        -------
        s : np.ndarray, shape (n_points, n_nodes)
            Shape function values at each point
        """
        xi = np.atleast_1d(xi).flatten()
        eta = np.atleast_1d(eta).flatten()

        if self.n_nodes == 4:
            # Bilinear shape functions
            # N_i = 1/4 * (1 + ξ_i*ξ) * (1 + η_i*η)
            return 0.25 * np.column_stack([
                (1 - xi) * (1 - eta),   # Node 1: (-1,-1)
                (1 + xi) * (1 - eta),   # Node 2: (+1,-1)
                (1 + xi) * (1 + eta),   # Node 3: (+1,+1)
                (1 - xi) * (1 + eta)    # Node 4: (-1,+1)
            ])

        else:  # n_nodes == 9
            # Biquadratic shape functions (Lagrange)
            # Basis functions in each direction
            phi_xi = np.column_stack([
                0.5 * xi * (xi - 1),    # ξ = -1
                1 - xi**2,              # ξ = 0
                0.5 * xi * (1 + xi)     # ξ = +1
            ])
            phi_eta = np.column_stack([
                0.5 * eta * (eta - 1),  # η = -1
                1 - eta**2,             # η = 0
                0.5 * eta * (1 + eta)   # η = +1
            ])

            # Tensor product to get 9 shape functions
            # Node numbering:
            # 7---6---5
            # |   |   |
            # 8---9---4
            # |   |   |
            # 1---2---3
            return self._assemble(phi_xi, phi_eta)

    def x(self, xi: Union[float, np.ndarray],
          eta: Union[float, np.ndarray]) -> np.ndarray:
        """Derivative of shape functions with respect to xi."""
        xi = np.atleast_1d(xi).flatten()
        eta = np.atleast_1d(eta).flatten()

        if self.n_nodes == 4:
            return 0.25 * np.column_stack([
                -(1 - eta),
                 (1 - eta),
                 (1 + eta),
                -(1 + eta)
            ])

        else:  # n_nodes == 9
            # Derivatives of basis functions
            dphi_xi = np.column_stack([
                xi - 0.5,
                -2*xi,
                xi + 0.5
            ])
            phi_eta = np.column_stack([
                0.5 * eta * (eta - 1),
                1 - eta**2,
                0.5 * eta * (1 + eta)
            ])

            return self._assemble(dphi_xi, phi_eta)

    def y(self, xi: Union[float, np.ndarray],
          eta: Union[float, np.ndarray]) -> np.ndarray:
        """Derivative of shape functions with respect to eta."""
        xi = np.atleast_1d(xi).flatten()
        eta = np.atleast_1d(eta).flatten()

        if self.n_nodes == 4:
            return 0.25 * np.column_stack([
                -(1 - xi),
                -(1 + xi),
                 (1 + xi),
                 (1 - xi)
            ])

        else:  # n_nodes == 9
            phi_xi = np.column_stack([
                0.5 * xi * (xi - 1),
                1 - xi**2,
                0.5 * xi * (1 + xi)
            ])
            dphi_eta = np.column_stack([
                eta - 0.5,
                -2*eta,
                eta + 0.5
            ])

            return self._assemble(phi_xi, dphi_eta)

    @staticmethod
    def _assemble(phi_xi: np.ndarray, phi_eta: np.ndarray) -> np.ndarray:
        """
        Assemble tensor product shape functions.

        Maps (3x3) tensor product to 9 node numbering:
        Nodes: 1,2,3 (bottom), 4,5,6 (right), 7,8,9 (top,left,center)
        """
        return np.column_stack([
            phi_xi[:, 0] * phi_eta[:, 0],  # Node 1: (-1,-1)
            phi_xi[:, 2] * phi_eta[:, 0],  # Node 2: (+1,-1)
            phi_xi[:, 2] * phi_eta[:, 2],  # Node 3: (+1,+1)
            phi_xi[:, 0] * phi_eta[:, 2],  # Node 4: (-1,+1)
            phi_xi[:, 1] * phi_eta[:, 0],  # Node 5: ( 0,-1)
            phi_xi[:, 2] * phi_eta[:, 1],  # Node 6: (+1, 0)
            phi_xi[:, 1] * phi_eta[:, 2],  # Node 7: ( 0,+1)
            phi_xi[:, 0] * phi_eta[:, 1],  # Node 8: (-1, 0)
            phi_xi[:, 1] * phi_eta[:, 1]   # Node 9: ( 0, 0)
        ])


# Test functions
if __name__ == "__main__":
    print("Testing TriangleShape:")

    # 3-node linear triangle
    tri3 = TriangleShape(3)
    x_test = np.array([0.0, 1.0, 0.0, 1/3])
    y_test = np.array([0.0, 0.0, 1.0, 1/3])

    N = tri3(x_test, y_test)
    print("  3-node at corners and centroid:")
    print("    N = \n{}".format(N))
    print("    Row sums (should be 1): {}".format(N.sum(axis=1)))

    # Check partition of unity
    assert np.allclose(N.sum(axis=1), 1.0), "3-node partition of unity failed"

    # 6-node quadratic triangle
    tri6 = TriangleShape(6)
    N6 = tri6(x_test, y_test)
    print("\n  6-node at corners and centroid:")
    print("    Row sums (should be 1): {}".format(N6.sum(axis=1)))
    assert np.allclose(N6.sum(axis=1), 1.0), "6-node partition of unity failed"

    # Check derivatives
    dNdx = tri3.x([0.5], [0.3])
    dNdy = tri3.y([0.5], [0.3])
    print("\n  Derivatives at (0.5, 0.3):")
    print("    dN/dx: {}".format(dNdx[0]))
    print("    dN/dy: {}".format(dNdy[0]))

    print("\nTesting QuadShape:")

    # 4-node bilinear quad
    quad4 = QuadShape(4)
    xi_test = np.array([-1, 1, 1, -1, 0])
    eta_test = np.array([-1, -1, 1, 1, 0])

    N4 = quad4(xi_test, eta_test)
    print("  4-node at corners and center:")
    print("    N = \n{}".format(N4))
    print("    Row sums (should be 1): {}".format(N4.sum(axis=1)))
    assert np.allclose(N4.sum(axis=1), 1.0), "4-node partition of unity failed"

    # 9-node biquadratic quad
    quad9 = QuadShape(9)
    N9 = quad9(xi_test, eta_test)
    print("\n  9-node at corners and center:")
    print("    Row sums (should be 1): {}".format(N9.sum(axis=1)))
    assert np.allclose(N9.sum(axis=1), 1.0), "9-node partition of unity failed"

    # Check derivatives
    dNdxi = quad4.x([0.5], [0.5])
    dNdeta = quad4.y([0.5], [0.5])
    print("\n  Derivatives at (0.5, 0.5):")
    print("    dN/dxi: {}".format(dNdxi[0]))
    print("    dN/deta: {}".format(dNdeta[0]))

    print("\n✓ All shape function tests passed!")
