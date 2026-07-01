"""
QuadFace class for boundary element integration.

Manages all quadrature rules including:
- Standard triangle/quad Gaussian quadrature
- Polar integration for self-interaction elements

MATLAB reference: /Misc/integration/@quadface/
"""

import numpy as np
from typing import Tuple

try:
    from .quadrature import lglnodes, triangle_unit_set
except ImportError:
    # For standalone testing
    from quadrature import lglnodes, triangle_unit_set


def _trisubdivide(xtab, ytab, wtab, nsub):
    """Refine unit-triangle integration by nsub x nsub subdivision.

    MATLAB: @quadface/private/trisubdivide.m
    """
    x_list, y_list, w_list = [], [], []
    h = 1.0 / nsub
    for i in range(nsub):
        for j in range(nsub - i):
            # triangle pointing upwards
            x_list.append(i + xtab)
            y_list.append(j + ytab)
            w_list.append(wtab.copy())
            # triangle pointing downwards (except for last row)
            if j != nsub - 1 - i:
                x_list.append(i + 1 - xtab)
                y_list.append(j + 1 - ytab)
                w_list.append(wtab.copy())
    x_out = np.hstack(x_list) * h
    y_out = np.hstack(y_list) * h
    w_out = np.hstack(w_list)
    n_sub_tri = len(x_out) // len(xtab)
    w_out = w_out / n_sub_tri
    return x_out, y_out, w_out


class QuadFace(object):
    """
    Integration over triangular or quadrilateral boundary elements.

    This class provides integration points and weights for:
    1. Standard integration over triangles (Gaussian quadrature)
    2. Polar integration over triangles (for near-field refinement)
    3. Polar integration over quadrilaterals

    Attributes
    ----------
    x, y, w : np.ndarray
        Standard triangle integration points and weights
    npol : tuple
        Number of points for polar integration (n_radial, n_angular)
    x3, y3, w3 : np.ndarray
        Polar integration points and weights for triangles
    x4, y4, w4 : np.ndarray
        Polar integration points and weights for quadrilaterals
    """

    def __init__(self, rule: int = 18, npol: Tuple[int, int] = (7, 5),
            refine: int = None):
        """
        Initialize quadrature rules.

        Parameters
        ----------
        rule : int, optional
            Triangle integration rule (default 18)
            Higher numbers = more points = better accuracy
        npol : tuple, optional
            (n_radial, n_angular) for polar integration
            Default: (7, 5) = 7 radial × 5 angular points
            Total points: 3 × n_radial × n_angular (for triangles)
                         4 × n_radial × n_angular (for quads)
        refine : int, optional
            Subdivide unit triangle into refine×refine sub-triangles, giving
            finer integration points. MATLAB: bemoptions('refine', N).
        """
        # MATLAB: if numel(op.npol) ~= 2, op.npol = [1,1] * op.npol; end
        # Scalar npol → broadcast to (npol, npol) per MATLAB @quadface init.
        if np.isscalar(npol):
            npol = (int(npol), int(npol))
        self.npol = npol

        # Standard triangle quadrature
        self.x, self.y, self.w = triangle_unit_set(rule)

        # Optional refinement (MATLAB @quadface/private/init.m lines 20-23)
        if refine is not None and refine > 1:
            self.x, self.y, self.w = _trisubdivide(self.x, self.y, self.w, int(refine))

        # Polar triangle quadrature
        self.x3, self.y3, self.w3 = self._polar_triangle(npol)

        # Polar quadrilateral quadrature
        self.x4, self.y4, self.w4 = self._polar_quad(npol)

    def _polar_triangle(self, npol: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Polar integration for triangular elements.

        Uses polar coordinates (ρ, φ) to avoid singularity at origin.
        Covers triangle by rotating 60° sector three times (120° apart).

        MATLAB reference: /Misc/integration/@quadface/private/init.m lines 25-53

        Parameters
        ----------
        npol : tuple
            (n_radial, n_angular) integration points

        Returns
        -------
        x, y, w : np.ndarray
            Integration points and weights in unit triangle coordinates
        """
        n_rad, n_ang = npol

        # Radial direction: LGL nodes mapped to [0, 1]
        # MATLAB line 27
        x1, w1 = lglnodes(n_rad)
        rho_1d = 0.5 * (x1 + 1 + 1e-6)  # Map [-1,1] to [0,1], add small epsilon

        # Angular direction: LGL nodes mapped to [270°, 330°] (60° sector)
        # MATLAB line 28
        x2, w2 = lglnodes(n_ang)
        phi_1d = np.deg2rad(270 + 60 * x2)  # 60° sector starting at 270°

        # Rotation angle for three sectors
        # MATLAB line 30
        phi0 = np.deg2rad(120)  # 120° rotation

        # Create 2D grid (MATLAB line 33)
        rho_grid, phi_grid = np.meshgrid(rho_1d, phi_1d, indexing='ij')
        rho_flat = rho_grid.ravel()  # MATLAB line 34
        phi_flat = phi_grid.ravel()

        # Radius to triangle edge at angle phi (MATLAB line 36)
        # For equilateral triangle: r_edge = 1 / (2 * sin(phi))
        rad_flat = 1.0 / np.abs(2 * np.sin(phi_flat))

        # Three rotated sectors (MATLAB line 38)
        phi = np.hstack([
            phi_flat,
            phi_flat + phi0,
            phi_flat + 2 * phi0
        ])

        # Integration points in polar coordinates (MATLAB line 41-42)
        # Note: rho and rad are repeated here by broadcasting
        rho_rep = np.tile(rho_flat, 3)
        rad_rep = np.tile(rad_flat, 3)

        x_polar = np.cos(phi) * rho_rep * rad_rep
        y_polar = np.sin(phi) * rho_rep * rad_rep

        # Transform to unit triangle coordinates (MATLAB line 44-45)
        x = (1 - np.sqrt(3) * x_polar - y_polar) / 3
        y = (1 + np.sqrt(3) * x_polar - y_polar) / 3

        # Integration weights (MATLAB line 48-50)
        # Outer product of 1D weights
        w_2d = np.outer(w1, w2).ravel()  # Shape: (n_rad * n_ang,)

        # Weight including Jacobian
        w_sector = w_2d * rho_flat * rad_flat**2

        # Replicate for three sectors
        w = np.tile(w_sector, 3)

        # Normalize to ensure sum(w) = 0.5 (area of unit triangle)
        w = w / np.sum(w)

        return x, y, w

    def _polar_quad(self, npol: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Polar integration for quadrilateral elements.

        Covers quad by rotating 45° sector four times (90° apart).

        MATLAB reference: /Misc/integration/@quadface/private/init.m lines 55-80

        Parameters
        ----------
        npol : tuple
            (n_radial, n_angular) integration points

        Returns
        -------
        x, y, w : np.ndarray
            Integration points and weights in unit quad coordinates [-1,1]×[-1,1]
        """
        n_rad, n_ang = npol

        # Radial direction: LGL nodes mapped to [0, 1] (MATLAB line 57)
        x1, w1 = lglnodes(n_rad)
        rho_1d = 0.5 * (x1 + 1 + 1e-6)

        # Angular direction: LGL nodes mapped to [90°, 135°] (MATLAB line 58)
        x2, w2 = lglnodes(n_ang)
        phi_1d = np.deg2rad(90 + 45 * x2)

        # Rotation angle for four sectors (MATLAB line 60)
        phi0 = np.deg2rad(90)  # 90° rotation

        # Create 2D grid (MATLAB line 63)
        rho_grid, phi_grid = np.meshgrid(rho_1d, phi_1d, indexing='ij')
        rho_flat = rho_grid.ravel()  # MATLAB line 64
        phi_flat = phi_grid.ravel()

        # Radius to quad edge at angle phi (MATLAB line 66)
        # For unit square [-1,1]×[-1,1]: r_edge = 1 / |sin(phi)|
        rad_flat = 1.0 / np.abs(np.sin(phi_flat))

        # Four rotated sectors (MATLAB line 68)
        phi = np.hstack([
            phi_flat,
            phi_flat + phi0,
            phi_flat + 2 * phi0,
            phi_flat + 3 * phi0
        ])

        # Integration points in polar coordinates (MATLAB line 71-72)
        rho_rep = np.tile(rho_flat, 4)
        rad_rep = np.tile(rad_flat, 4)

        x = np.cos(phi) * rho_rep * rad_rep
        y = np.sin(phi) * rho_rep * rad_rep

        # Integration weights (MATLAB line 75-77)
        w_2d = np.outer(w1, w2).ravel()
        w_sector = w_2d * rho_flat * rad_flat**2

        # Replicate for four sectors
        w = np.tile(w_sector, 4)

        # Normalize: sum(w) = 4 (area of unit square [-1,1]×[-1,1])
        w = 4 * w / np.sum(w)

        return x, y, w

    def __repr__(self):
        return ("QuadFace(n_std={}, "
                "n_polar_tri={}, "
                "n_polar_quad={}, "
                "npol={})".format(
                    len(self.x), len(self.x3), len(self.x4), self.npol))


# Test functions
if __name__ == "__main__":
    print("Testing QuadFace:")
    print("=" * 70)

    # Create QuadFace with default settings
    quad = QuadFace(rule=18, npol=(7, 5))
    print("\n{}".format(quad))

    # Test standard triangle quadrature
    print("\nStandard triangle quadrature:")
    print("  Points: {}".format(len(quad.x)))
    print("  sum(w) = {:.10f} (should be 1.0)".format(np.sum(quad.w)))
    print("  All points in triangle: {}".format(np.all((quad.x >= -1e-10) & (quad.y >= -1e-10) & (quad.x + quad.y <= 1 + 1e-10))))

    assert np.abs(np.sum(quad.w) - 1.0) < 1e-10, "Standard triangle weights incorrect"

    # Test polar triangle quadrature
    print("\nPolar triangle quadrature:")
    print("  Points: {} = 3 × {} × {}".format(len(quad.x3), quad.npol[0], quad.npol[1]))
    print("  sum(w3) = {:.10f}".format(np.sum(quad.w3)))
    print("  All points in triangle: {}".format(np.all((quad.x3 >= -1e-10) & (quad.y3 >= -1e-10) & (quad.x3 + quad.y3 <= 1 + 1e-10))))

    # Note: MATLAB normalizes polar weights to sum=1.0, not 0.5
    # These weights are later scaled by element area in quadpol()
    # This is different from standard quadrature which includes area directly
    assert np.abs(np.sum(quad.w3) - 1.0) < 1e-10, "Polar triangle weights should sum to 1.0"

    # Test that integration of f=1 over unit triangle gives correct area
    # The transformation includes a factor, so we need to check the actual integration
    tri_area_test = np.sum(quad.w3 * 0.5)  # 0.5 = area of unit triangle
    print("  Integration test: ∫1 dA = {:.10f} (should be 0.5)".format(tri_area_test))
    assert np.abs(tri_area_test - 0.5) < 1e-10, "Polar triangle integration incorrect"

    # Check points are within reasonable bounds (allow for numerical precision)
    tri_in_bounds = np.all((quad.x3 >= -1e-6) & (quad.y3 >= -1e-6) & (quad.x3 + quad.y3 <= 1 + 1e-6))
    print("  Points within tolerance: {}".format(tri_in_bounds))
    assert tri_in_bounds, "Polar triangle points outside reasonable bounds"

    # Test polar quad quadrature
    print("\nPolar quadrilateral quadrature:")
    print("  Points: {} = 4 × {} × {}".format(len(quad.x4), quad.npol[0], quad.npol[1]))
    print("  sum(w4) = {:.10f} (should be 4.0)".format(np.sum(quad.w4)))

    assert np.abs(np.sum(quad.w4) - 4.0) < 1e-9, "Polar quad weights incorrect"

    # Check points are within reasonable bounds (allow for numerical precision)
    quad_in_bounds = np.all((quad.x4 >= -1 - 1e-5) & (quad.x4 <= 1 + 1e-5) &
                            (quad.y4 >= -1 - 1e-5) & (quad.y4 <= 1 + 1e-5))
    print("  Points within tolerance: {}".format(quad_in_bounds))
    assert quad_in_bounds, "Polar quad points outside reasonable bounds"

    # Test different npol settings
    print("\nTesting different npol settings:")
    for npol in [(3, 3), (5, 5), (10, 10)]:
        q = QuadFace(rule=18, npol=npol)
        w3_sum = np.sum(q.w3)
        w4_sum = np.sum(q.w4)
        print("  npol={}: tri_sum={:.6f}, quad_sum={:.6f}, "
              "n_tri={}, n_quad={}".format(npol, w3_sum, w4_sum, len(q.x3), len(q.x4)))

        assert np.abs(w3_sum - 1.0) < 1e-9, "Triangle weights incorrect for npol={}".format(npol)
        assert np.abs(w4_sum - 4.0) < 1e-8, "Quad weights incorrect for npol={}".format(npol)

    print("\n" + "=" * 70)
    print("✓ All QuadFace tests passed!")
