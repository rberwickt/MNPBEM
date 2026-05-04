"""
Refined retarded Green function with polar integration.

This module implements proper Green function refinement using:
1. Multi-order Taylor expansion: G = Σ g_n × (ik)^n / factorial(n)
2. Polar integration for diagonal elements
3. Boundary element integration for near-field off-diagonal elements

MATLAB reference: Greenfun/@greenret/
"""

import numpy as np
from typing import Optional, Tuple
from scipy.sparse import csr_matrix, find as sparse_find
import math

try:
    from .refine_utils import refinematrix
    from ..geometry.particle import Particle
except ImportError:
    from refine_utils import refinematrix
    # For standalone testing, we'll handle the import later
    Particle = None


class GreenRetRefined(object):
    """
    Retarded Green function with proper polar integration refinement.

    Attributes
    ----------
    p1, p2 : Particle
        Source and target particles
    deriv : str
        'norm' for normal derivative, 'cart' for Cartesian derivative
    order : int
        Order of multi-order expansion (default: 2)
    ind : ndarray
        Linear indices of refined elements
    row, col : ndarray
        Row and column indices of refined elements
    g : ndarray, shape (n_refined, order+1)
        Refined Green function expansion coefficients
    f : ndarray, shape (n_refined, order+1)
        Refined derivative expansion coefficients
    """

    def __init__(self, p1, p2, deriv='norm', order=2, **options):
        """
        Initialize refined retarded Green function.

        Parameters
        ----------
        p1, p2 : Particle
            Source and target particles
        deriv : str, optional
            'norm' (surface derivative) or 'cart' (Cartesian derivative)
            Default: 'norm'
        order : int, optional
            Order of Taylor expansion (default: 2)
            Higher order = better accuracy but more computation
        **options : dict
            AbsCutoff : float
                Absolute distance cutoff for refinement (nm)
            RelCutoff : float
                Relative distance cutoff (multiples of element radius)
                Default: 3
        """
        self.p1 = p1
        self.p2 = p2
        self.deriv = deriv
        self.order = order
        self._d_cache = None

        # Initialize refinement
        self._init_refinement(**options)

    def _init_refinement(self, **options):
        """
        Compute refined Green function elements.

        MATLAB reference: Greenfun/@greenret/private/init.m
        """
        # Refinement matrix: 0=far, 1=near, 2=diagonal
        AbsCutoff = options.get('AbsCutoff', 0)
        RelCutoff = options.get('RelCutoff', 3)

        ir = refinematrix(self.p1, self.p2, AbsCutoff=AbsCutoff, RelCutoff=RelCutoff)

        # Linear indices of refined elements (MATLAB line 24)
        self.ind = np.array(ir.nonzero()).T  # (n_refined, 2) array of (row, col)
        n_refined = len(self.ind)

        # If no refinement needed, return empty arrays
        if n_refined == 0 or self.order == 0:
            self.g = np.array([])
            self.f = np.array([])
            self.row = np.array([], dtype=int)
            self.col = np.array([], dtype=int)
            return

        # Store row and column indices separately
        self.row = self.ind[:, 0]
        self.col = self.ind[:, 1]

        # Allocate arrays for multi-order expansion (MATLAB lines 35-40)
        order = self.order
        self.g = np.zeros((n_refined, order + 1), dtype=complex)
        if self.deriv == 'cart':
            self.f = np.zeros((n_refined, 3, order + 1), dtype=complex)
        else:
            self.f = np.zeros((n_refined, order + 1), dtype=complex)

        # Refine diagonal elements (ir == 2)
        # MATLAB lines 42-95
        self._refine_diagonal(ir)

        # Refine off-diagonal elements (ir == 1)
        # MATLAB lines 98-177
        self._refine_offdiagonal(ir)

    def _refine_diagonal(self, ir):
        """
        Refine diagonal elements using polar integration.

        MATLAB reference: Greenfun/@greenret/private/init.m lines 42-95

        Algorithm:
        1. Find diagonal elements where ir == 2
        2. Use quadpol() for polar integration
        3. For each order n: g_n = ∫ r^(n-1) / n! dA
        4. For derivatives: f_n = ∫ (n-1) × (n·r) × r^(n-3) / n! dA

        v1.6.1: Vectorised across faces using a single (n_diag, n_pts_per_face)
        broadcast.  quadpol returns a uniform number of integration points per
        face on the standard particle meshes used by BEM, so the per-face
        Python loop collapses to a few numpy reductions.  Falls back to the
        per-face path when integration points are non-uniform.
        """
        # Find diagonal elements (MATLAB line 47)
        diag_mask = ir.toarray() == 2
        if not np.any(diag_mask):
            return

        face_idx, face2_idx = np.where(diag_mask)

        # Find corresponding indices in refinement array
        # Create mapping from (row, col) to refinement index
        refine_map = {(r, c): i for i, (r, c) in enumerate(zip(self.row, self.col))}
        iface = np.array([refine_map[(r, c)] for r, c in zip(face_idx, face2_idx)])

        # Polar integration points and weights (MATLAB line 51)
        # Returns: pos (n_total, 3), weight (n_total,), row (n_total,)
        # where row[i] indicates which face point i belongs to
        if Particle is None:
            # Standalone testing - import here
            import sys
            sys.path.insert(0, '/home/user/MNPBEM')
            from mnpbem.geometry.particle import Particle as Part
            pos, weight, row_indices = Part.quadpol(self.p2, face2_idx)
        else:
            pos, weight, row_indices = Particle.quadpol(self.p2, face2_idx)

        # Get positions of source faces
        pos1 = self.p1.pos[face_idx]  # (n_diag, 3)
        nvec1 = self.p1.nvec[face_idx]  # (n_diag, 3)

        n_diag = len(face_idx)

        # ---------------- v1.6.1 vectorised batched path ----------------
        # quadpol returns uniform per-face point counts on the standard
        # particle meshes used by BEM (56 / 192 points).  When that holds
        # we can reshape the (n_total, 3) integration arrays to
        # (n_diag, n_pts, 3) and replace the per-face Python loop with a
        # single numpy reduction.  When non-uniform we fall back to the
        # original per-face path.
        if n_diag > 0:
            counts = np.bincount(row_indices, minlength = n_diag)[:n_diag]
        else:
            counts = np.zeros(0, dtype = np.int64)
        uniform = counts.size > 0 and int(counts.min()) == int(counts.max())

        if uniform and counts[0] > 0:
            n_pts = int(counts[0])

            # Sort points by face so each face occupies a contiguous block.
            # quadpol's emission order is already face-by-face, but we sort
            # defensively.
            order_perm = np.argsort(row_indices, kind = 'stable')
            pos_sorted = np.ascontiguousarray(pos[order_perm], dtype = np.float64)
            w_sorted = np.ascontiguousarray(weight[order_perm], dtype = np.float64)
            pos_2d = pos_sorted.reshape(n_diag, n_pts, 3)
            w_2d = w_sorted.reshape(n_diag, n_pts)

            vec = pos1[:, np.newaxis, :] - pos_2d                # (n_diag, n_pts, 3)
            r = np.sqrt((vec * vec).sum(axis = 2))               # (n_diag, n_pts)
            r = np.maximum(r, np.finfo(float).eps)
            n_dot_r = np.einsum('ij,ikj->ik', nvec1, vec)        # (n_diag, n_pts)

            inv_r = 1.0 / r
            inv_r2 = inv_r * inv_r
            inv_r3 = inv_r * inv_r2

            # Incremental powers: r_pow_nm1 tracks r^(n-1), r_pow_nm3 tracks r^(n-3).
            # At n=0 these are r^-1 and r^-3 respectively.
            r_pow_nm1 = inv_r.copy()
            r_pow_nm3 = inv_r3.copy()

            for n in range(self.order + 1):
                self.g[iface, n] = (
                    (w_2d * r_pow_nm1).sum(axis = 1)
                ) / math.factorial(n)
                if self.deriv == 'norm':
                    self.f[iface, n] = (
                        w_2d * (n - 1) * n_dot_r * r_pow_nm3
                    ).sum(axis = 1) / math.factorial(n)
                r_pow_nm1 = r_pow_nm1 * r
                r_pow_nm3 = r_pow_nm3 * r

            if self.deriv == 'cart':
                rr_max = r.max(axis = 1)                          # (n_diag,)
                rr = np.maximum(r, 1e-4 * rr_max[:, np.newaxis])
                inv_rr3 = 1.0 / (rr * rr * rr)
                tvec1 = self.p1.tvec1[face_idx]                    # (n_diag, 3)
                tvec2 = self.p1.tvec2[face_idx]                    # (n_diag, 3)
                in1 = np.einsum('ij,ikj->ik', tvec1, vec)
                in2 = np.einsum('ij,ikj->ik', tvec2, vec)

                r_pow_nm3_c = inv_r3.copy()
                r_pow_n_c = np.ones_like(r)
                for n in range(self.order + 1):
                    f1 = (
                        w_2d * (n - 1) * n_dot_r * r_pow_nm3_c
                    ).sum(axis = 1) / math.factorial(n)
                    f2 = (
                        w_2d * (n - 1) * in1 * r_pow_n_c * inv_rr3
                    ).sum(axis = 1) / math.factorial(n)
                    f3 = (
                        w_2d * (n - 1) * in2 * r_pow_n_c * inv_rr3
                    ).sum(axis = 1) / math.factorial(n)
                    self.f[iface, :, n] = (
                        nvec1 * f1[:, np.newaxis] +
                        tvec1 * f2[:, np.newaxis] +
                        tvec2 * f3[:, np.newaxis]
                    )
                    r_pow_nm3_c = r_pow_nm3_c * r
                    r_pow_n_c = r_pow_n_c * r
            return

        # -------- Fallback: per-face Python loop (non-uniform meshes) ----
        for i, (face, face2, iref) in enumerate(zip(face_idx, face2_idx, iface)):
            # Get integration points for this face
            # row_indices contains the index into face2_idx, not face2_idx itself
            face_points = (row_indices == i)
            pos_face = pos[face_points]  # (n_points, 3)
            w_face = weight[face_points]  # (n_points,)

            # Vector from integration points to face centroid (MATLAB line 56)
            vec = pos1[i] - pos_face  # (n_points, 3)

            # Distance (MATLAB line 58)
            r = np.sqrt(np.sum(vec**2, axis=1))  # (n_points,)

            # Green function: g_n = Σ w × r^(n-1) / n! (MATLAB lines 61-63)
            for n in range(self.order + 1):
                self.g[iref, n] = np.sum(w_face * r**(n - 1)) / math.factorial(n)

            # Surface derivative (MATLAB lines 65-94)
            n_dot_r = np.sum(vec * nvec1[i], axis=1)

            if self.deriv == 'norm':
                for n in range(self.order + 1):
                    self.f[iref, n] = np.sum(w_face * (n - 1) * n_dot_r * r**(n - 3)) / math.factorial(n)
            else:
                # deriv='cart': MATLAB init.m lines 74-93
                rr = np.maximum(r, 1e-4 * np.max(r))
                in1 = np.sum(vec * self.p1.tvec1[face], axis=1)
                in2 = np.sum(vec * self.p1.tvec2[face], axis=1)
                for n in range(self.order + 1):
                    f1 = np.sum(w_face * (n - 1) * n_dot_r * r**(n - 3)) / math.factorial(n)
                    f2 = np.sum(w_face * (n - 1) * in1 * r**n / rr**3) / math.factorial(n)
                    f3 = np.sum(w_face * (n - 1) * in2 * r**n / rr**3) / math.factorial(n)
                    self.f[iref, :, n] = (nvec1[i] * f1 +
                                          self.p1.tvec1[face] * f2 +
                                          self.p1.tvec2[face] * f3)

    def _refine_offdiagonal(self, ir):
        """
        Refine off-diagonal near-field elements using boundary element integration.

        MATLAB reference: Greenfun/@greenret/private/init.m lines 98-177

        Algorithm:
        1. Find faces that have near-field neighbors (ir == 1)
        2. Use quad() for boundary element integration
        3. For each order n: g_n = ∫ (r-r0)^n / (r × n!) dA
           where r0 is distance to face centroid
        4. For derivatives: Similar but with directional factors

        v1.6.1: Per-face Python loop replaced by a single batched numpy
        reduction over a flat (pair, n_pts) layout.  When `quad()` returns
        a uniform number of integration points per face (the typical mesh
        case for tricube / sphere / shape primitives) we collapse the
        whole loop into a few numpy operations; otherwise we fall back to
        the original CSR per-face path.
        """
        # Faces to be refined (columns with any ir==1) (MATLAB line 100)
        ir_array = ir.toarray()
        reface = np.where(np.any(ir_array == 1, axis=0))[0]

        if len(reface) == 0:
            return

        # Boundary element integration (MATLAB line 102)
        # Returns: pos (n_total, 3), w_sparse (n_faces, n_points), iface (n_points,)
        # Note: self.p2.quad is an attribute, so we need to call the method via the class
        if Particle is None:
            # Standalone testing - import here
            import sys
            sys.path.insert(0, '/home/user/MNPBEM')
            from mnpbem.geometry.particle import Particle as Part
            pos_all, w_sparse, row_indices = Part.quad(self.p2, reface)
        else:
            pos_all, w_sparse, row_indices = Particle.quad(self.p2, reface)

        # Get source positions
        pos1 = self.p1.pos
        nvec1 = self.p1.nvec

        # Create mapping from (row, col) to refinement index
        refine_map = {(r, c): i for i, (r, c) in enumerate(zip(self.row, self.col))}

        # ---------------- v1.6.1 batched vectorised path ----------------
        n_face = len(reface)
        counts = np.bincount(row_indices, minlength = n_face)[:n_face]
        uniform = counts.size > 0 and int(counts.min()) == int(counts.max())
        if uniform and counts[0] > 0:
            n_pts = int(counts[0])

            # Sort points by face so each face owns a contiguous block.
            order_perm = np.argsort(row_indices, kind = 'stable')
            pos_sorted = np.ascontiguousarray(pos_all[order_perm], dtype = np.float64)
            pos_2d = pos_sorted.reshape(n_face, n_pts, 3)        # (n_face, n_pts, 3)

            # Build per-face w_face arrays from the sparse weights.
            # The sparse layout has w_sparse[i, :] non-zero exactly at the
            # integration points of face i, in the same column ordering
            # as `pos_all[row_indices == i]`.
            w_csr = w_sparse.tocsr() if hasattr(w_sparse, 'tocsr') else w_sparse
            w_2d = np.zeros((n_face, n_pts), dtype = np.float64)
            indptr = w_csr.indptr
            indices = w_csr.indices
            data = w_csr.data
            for i in range(n_face):
                rs, re = int(indptr[i]), int(indptr[i + 1])
                if re == rs:
                    continue
                w_row = data[rs:re]
                # Drop zero-weight slots to match the original w[w!=0] filter.
                nz = w_row > 0
                if not np.all(nz):
                    w_row = w_row[nz]
                # Defensive: truncate / pad to n_pts (matches sorted_pos size).
                k = min(len(w_row), n_pts)
                w_2d[i, :k] = w_row[:k]

            # Build flat (face_idx, nb) pair list across all faces.
            # Use ir_array[:, reface] to vectorise the np.where over columns.
            nb_mask = ir_array[:, reface] == 1                   # (n_total, n_face)
            nb_rows, face_cols = np.where(nb_mask)               # both 1-D
            n_pairs = nb_rows.size

            if n_pairs > 0:
                # Refinement index per (nb_row, reface[face_col]) pair.
                # Vectorise via the existing refine_map lookup.
                iref = np.empty(n_pairs, dtype = np.int64)
                for k in range(n_pairs):
                    iref[k] = refine_map[(int(nb_rows[k]), int(reface[face_cols[k]]))]

                # Per-pair source data (broadcast over n_pts).
                pos_src = pos1[nb_rows]                           # (n_pairs, 3)
                nvec_src = nvec1[nb_rows]                         # (n_pairs, 3)
                pos_face_pair = pos_2d[face_cols]                 # (n_pairs, n_pts, 3)
                w_face_pair = w_2d[face_cols]                     # (n_pairs, n_pts)

                # Difference vectors and distances (MATLAB lines 133-137)
                rvec = pos_src[:, np.newaxis, :] - pos_face_pair  # (n_pairs, n_pts, 3)
                x = rvec[..., 0]
                y = rvec[..., 1]
                z = rvec[..., 2]
                r = np.sqrt((rvec * rvec).sum(axis = 2))
                r = np.maximum(r, np.finfo(float).eps)

                # Distance from face centroids (MATLAB lines 140-142)
                vec0 = self.p2.pos[reface[face_cols]] - pos_src   # (n_pairs, 3)
                r0 = np.sqrt((vec0 * vec0).sum(axis = 1))         # (n_pairs,)
                delta = r - r0[:, np.newaxis]                     # (n_pairs, n_pts)

                inv_r = 1.0 / r
                inv_r2 = inv_r * inv_r
                inv_r3 = inv_r2 * inv_r

                n_dot_r = (
                    nvec_src[:, 0:1] * x +
                    nvec_src[:, 1:2] * y +
                    nvec_src[:, 2:3] * z
                )

                # Green function expansion (MATLAB lines 145-148)
                # Maintain delta_pow incrementally to avoid `**n` blow-up.
                delta_pow_n = np.ones_like(r)                     # delta^n at n=0
                delta_pow_nm1 = np.zeros_like(r)                  # delta^(n-1) (irrelevant at n=0)

                for n in range(self.order + 1):
                    integrand = delta_pow_n * inv_r / math.factorial(n)
                    self.g[iref, n] = (integrand * w_face_pair).sum(axis = 1)
                    delta_pow_nm1 = delta_pow_n
                    delta_pow_n = delta_pow_n * delta

                if self.deriv == 'norm':
                    # f_0 = -(n_dot_r / r^3) @ w
                    self.f[iref, 0] = -(n_dot_r * inv_r3 * w_face_pair).sum(axis = 1)
                    delta_pow_n = delta.copy()                    # delta^1 at n=1
                    delta_pow_nm1 = np.ones_like(r)               # delta^0 at n=1
                    for n in range(1, self.order + 1):
                        term1 = -delta_pow_n * inv_r3 / math.factorial(n)
                        term2 = delta_pow_nm1 * inv_r2 / math.factorial(n - 1)
                        self.f[iref, n] = (
                            n_dot_r * (term1 + term2) * w_face_pair
                        ).sum(axis = 1)
                        delta_pow_nm1 = delta_pow_n
                        delta_pow_n = delta_pow_n * delta
                else:
                    # deriv='cart': MATLAB init.m lines 165-175
                    f_scalar0 = -inv_r3
                    self.f[iref, 0, 0] = (x * f_scalar0 * w_face_pair).sum(axis = 1)
                    self.f[iref, 1, 0] = (y * f_scalar0 * w_face_pair).sum(axis = 1)
                    self.f[iref, 2, 0] = (z * f_scalar0 * w_face_pair).sum(axis = 1)
                    delta_pow_n = delta.copy()
                    delta_pow_nm1 = np.ones_like(r)
                    for n in range(1, self.order + 1):
                        term1 = -delta_pow_n * inv_r3 / math.factorial(n)
                        term2 = delta_pow_nm1 * inv_r2 / math.factorial(n - 1)
                        f_scalar = term1 + term2
                        self.f[iref, 0, n] = (x * f_scalar * w_face_pair).sum(axis = 1)
                        self.f[iref, 1, n] = (y * f_scalar * w_face_pair).sum(axis = 1)
                        self.f[iref, 2, n] = (z * f_scalar * w_face_pair).sum(axis = 1)
                        delta_pow_nm1 = delta_pow_n
                        delta_pow_n = delta_pow_n * delta
            return

        # -------- Fallback: per-face Python loop (non-uniform meshes) ----
        # Process each face to be refined (MATLAB line 116)
        for face_idx, face in enumerate(reface):
            # Find neighbor faces that need refinement for this face
            nb = np.where(ir_array[:, face] == 1)[0]

            if len(nb) == 0:
                continue

            # Indices in refinement array
            iface = np.array([refine_map[(n, face)] for n in nb])

            # Get integration points for this face
            face_mask = (row_indices == face_idx)
            pos_face = pos_all[face_mask]  # (n_points, 3)

            # Extract weights from sparse matrix for this face
            # w_sparse is (n_faces, n_points)
            w_row = w_sparse[face_idx, :].toarray().ravel()
            w_face = w_row[w_row != 0]  # Get non-zero weights

            # Difference vectors (MATLAB lines 133-137)
            # Broadcasting: (n_nb, 1, 3) - (1, n_points, 3) = (n_nb, n_points, 3)
            x = pos1[nb, 0:1] - pos_face[:, 0]  # (n_nb, n_points)
            y = pos1[nb, 1:2] - pos_face[:, 1]
            z = pos1[nb, 2:3] - pos_face[:, 2]

            # Distance from integration points to centroids (MATLAB line 137)
            r = np.sqrt(x**2 + y**2 + z**2)  # (n_nb, n_points)

            # Distance from face centroids (MATLAB lines 140-142)
            vec0 = self.p2.pos[face] - pos1[nb]  # (n_nb, 3)
            r0 = np.sqrt(np.sum(vec0**2, axis=1))  # (n_nb,)

            # Green function expansion (MATLAB lines 145-148)
            # g_n = ∫ (r-r0)^n / (r × n!) dA
            for n in range(self.order + 1):
                integrand = (r - r0[:, np.newaxis])**n / r / math.factorial(n)
                self.g[iface, n] = integrand @ w_face  # Matrix-vector product

            # Surface derivative expansion (MATLAB lines 151-164)
            # Inner product: n·(x,y,z)
            n_dot_r = (nvec1[nb, 0:1] * x +
                      nvec1[nb, 1:2] * y +
                      nvec1[nb, 2:3] * z)  # (n_nb, n_points)

            if self.deriv == 'norm':
                self.f[iface, 0] = -(n_dot_r / r**3) @ w_face
                for n in range(1, self.order + 1):
                    term1 = -(r - r0[:, np.newaxis])**n / (r**3 * math.factorial(n))
                    term2 = (r - r0[:, np.newaxis])**(n-1) / (r**2 * math.factorial(n-1))
                    self.f[iface, n] = (n_dot_r * (term1 + term2)) @ w_face
            else:
                # deriv='cart': MATLAB init.m lines 165-175
                f_scalar = -1.0 / r**3
                self.f[iface, 0, 0] = (x * f_scalar) @ w_face
                self.f[iface, 1, 0] = (y * f_scalar) @ w_face
                self.f[iface, 2, 0] = (z * f_scalar) @ w_face
                for n in range(1, self.order + 1):
                    term1 = -(r - r0[:, np.newaxis])**n / (r**3 * math.factorial(n))
                    term2 = (r - r0[:, np.newaxis])**(n-1) / (r**2 * math.factorial(n-1))
                    f_scalar = term1 + term2
                    self.f[iface, 0, n] = (x * f_scalar) @ w_face
                    self.f[iface, 1, n] = (y * f_scalar) @ w_face
                    self.f[iface, 2, n] = (z * f_scalar) @ w_face

    def _ensure_cache(self):
        """Build and cache wavelength-independent distance quantities."""
        if self._d_cache is not None:
            return
        from ._numba_ret_kernels import (
            green_ret_distances, numba_enabled,
            gpu_enabled, green_ret_distances_gpu,
        )

        pos1 = self.p1.pos
        pos2 = self.p2.pos
        area2 = self.p2.area
        nvec1 = self.p1.nvec
        same = self.p1 is self.p2

        on_gpu = gpu_enabled()
        if on_gpu:
            d, inv_d, n_dot_r, x, y, z = green_ret_distances_gpu(
                pos1, pos2, nvec1, area2, same=same, want_r=True
            )
            inv_d2 = inv_d * inv_d
        elif numba_enabled():
            d, inv_d, n_dot_r, x, y, z = green_ret_distances(
                pos1, pos2, nvec1, area2, same = same, want_r = True
            )
            inv_d2 = inv_d * inv_d
        else:
            x = pos1[:, 0:1] - pos2[:, 0]  # (n1, n2)
            y = pos1[:, 1:2] - pos2[:, 1]
            z = pos1[:, 2:3] - pos2[:, 2]
            d = np.sqrt(x**2 + y**2 + z**2)
            d = np.maximum(d, np.finfo(float).eps)
            inv_d = 1.0 / d
            inv_d2 = inv_d * inv_d
            n_dot_r = (nvec1[:, 0:1] * x + nvec1[:, 1:2] * y + nvec1[:, 2:3] * z)

        self._d_cache = {
            'x': x, 'y': y, 'z': z, 'd': d,
            'inv_d': inv_d, 'inv_d2': inv_d2,
            'area2': area2, 'n_dot_r': n_dot_r,
            'on_gpu': on_gpu,
        }

    def eval(self, k, key):
        """
        Evaluate Green function with proper refinement.

        MATLAB reference: Greenfun/@greenret/private/eval1.m

        Parameters
        ----------
        k : float
            Wavenumber (2π/λ where λ is wavelength in medium)
        key : str
            'G' - Green function
            'F' - Surface derivative
            'H1' - F + 2π (inside)
            'H2' - F - 2π (outside)

        Returns
        -------
        g : ndarray, shape (n1, n2)
            Green function matrix
        """
        self._ensure_cache()
        c = self._d_cache
        d = c['d']
        inv_d = c['inv_d']
        inv_d2 = c['inv_d2']
        area2 = c['area2']
        on_gpu = c.get('on_gpu', False)

        # GPU fast path -- assemble pre-phase matrix on device, apply
        # refinement overlay on device, multiply phase on device, then
        # bring back to host so callers (BEM matrix builders) stay
        # backward-compatible with numpy arrays. Stage 2 will keep the
        # result on device when callers are GPU-aware too.
        if on_gpu:
            return self._eval_gpu(k, key, c)

        # Numba fast path for distinct-particle (meshfield / observer) case.
        # Refinement overrides act on the *pre-phase* matrix, so the kernel
        # produces the pre-phase result, the caller overwrites refined
        # entries via numpy fancy indexing, and a final numba phase apply
        # multiplies in exp(i k d).
        use_numba = self.p1 is not self.p2
        nb = None
        if use_numba:
            from ..simulation import _meshfield_numba as _nb
            if _nb.numba_enabled():
                nb = _nb

        # Evaluate based on key
        if key == 'G':
            if nb is not None:
                G = nb.ret_G_pre(inv_d, area2)
                if len(self.ind) > 0:
                    ik_powers = np.array([(1j * k)**n for n in range(self.order + 1)])
                    G_refined = self.g @ ik_powers
                    G[self.row, self.col] = G_refined
                phase = nb.ret_phase(d, k)
                nb.apply_phase_2d(G, phase)
                return G

            G = inv_d * area2[np.newaxis, :] + 0j

            if len(self.ind) > 0:
                ik_powers = np.array([(1j * k)**n for n in range(self.order + 1)])
                G_refined = self.g @ ik_powers
                G[self.row, self.col] = G_refined

            G = G * np.exp(1j * k * d)
            return G

        elif key == 'F':
            if self.deriv == 'cart':
                x, y, z = c['x'], c['y'], c['z']
                nvec = self.p1.nvec
                if nb is not None:
                    F = nb.ret_F_cart_pre(inv_d, inv_d2, x, y, z, nvec, area2, k)
                    if len(self.ind) > 0:
                        ik_powers = np.array([(1j * k)**n for n in range(self.order + 1)])
                        nvec_ref = nvec[self.row]
                        F_refined = np.einsum('ij,ijk,k->i', nvec_ref, self.f, ik_powers)
                        F[self.row, self.col] = F_refined
                    phase = nb.ret_phase(d, k)
                    nb.apply_phase_2d(F, phase)
                    return F

                # MATLAB eval1.m lines 110-124: F via Gp inner product
                f_aux = (1j * k - inv_d) * inv_d2
                F = (nvec[:, 0:1] * (f_aux * x) +
                     nvec[:, 1:2] * (f_aux * y) +
                     nvec[:, 2:3] * (f_aux * z)) * area2[np.newaxis, :]

                if len(self.ind) > 0:
                    ik_powers = np.array([(1j * k)**n for n in range(self.order + 1)])
                    # MATLAB line 120: F(ind) = inner(nvec(i,:), f) * ik_powers
                    nvec_ref = nvec[self.row]
                    F_refined = np.einsum('ij,ijk,k->i', nvec_ref, self.f, ik_powers)
                    F[self.row, self.col] = F_refined

                F = F * np.exp(1j * k * d)
                return F
            else:
                n_dot_r = c['n_dot_r']
                if nb is not None:
                    F = nb.ret_F_norm_pre(inv_d, inv_d2, n_dot_r, area2, k)
                    if len(self.ind) > 0:
                        ik_powers = np.array([(1j * k)**n for n in range(self.order + 1)])
                        F_refined = self.f @ ik_powers
                        F[self.row, self.col] = F_refined
                    phase = nb.ret_phase(d, k)
                    nb.apply_phase_2d(F, phase)
                    return F

                F = n_dot_r * (1j * k - inv_d) * inv_d2 * area2[np.newaxis, :]

                if len(self.ind) > 0:
                    ik_powers = np.array([(1j * k)**n for n in range(self.order + 1)])
                    F_refined = self.f @ ik_powers
                    F[self.row, self.col] = F_refined

                F = F * np.exp(1j * k * d)
                return F

        elif key == 'H1':
            H1 = self.eval(k, 'F')
            if self.p1 is self.p2:
                np.fill_diagonal(H1, np.diag(H1) + 2.0 * np.pi)
            return H1

        elif key == 'H2':
            H2 = self.eval(k, 'F')
            if self.p1 is self.p2:
                np.fill_diagonal(H2, np.diag(H2) - 2.0 * np.pi)
            return H2

        elif key == 'Gp':
            x, y, z = c['x'], c['y'], c['z']
            if nb is not None:
                Gp = nb.ret_Gp_pre(inv_d, inv_d2, x, y, z, area2, k)
                if len(self.ind) > 0 and self.deriv == 'cart':
                    ik_powers = np.array([(1j * k)**n for n in range(self.order + 1)])
                    Gp_refined = np.einsum('ijk,k->ij', self.f, ik_powers)
                    Gp[self.row, 0, self.col] = Gp_refined[:, 0]
                    Gp[self.row, 1, self.col] = Gp_refined[:, 1]
                    Gp[self.row, 2, self.col] = Gp_refined[:, 2]
                phase = nb.ret_phase(d, k)
                nb.apply_phase_3d_axis02(Gp, phase)
                return Gp

            phase = np.exp(1j * k * d)
            f_aux = (1j * k - inv_d) * inv_d2
            # Gp as (n1, n2, 3) — then transpose to (n1, 3, n2)
            Gp_x = f_aux * x * area2[np.newaxis, :]
            Gp_y = f_aux * y * area2[np.newaxis, :]
            Gp_z = f_aux * z * area2[np.newaxis, :]

            # Apply refinement (MATLAB eval1.m lines 96-98)
            if len(self.ind) > 0 and self.deriv == 'cart':
                ik_powers = np.array([(1j * k)**n for n in range(self.order + 1)])
                # f is (n_ref, 3, order+1), Gp_refined = f @ ik_powers → (n_ref, 3)
                Gp_refined = np.einsum('ijk,k->ij', self.f, ik_powers)
                Gp_x[self.row, self.col] = Gp_refined[:, 0]
                Gp_y[self.row, self.col] = Gp_refined[:, 1]
                Gp_z[self.row, self.col] = Gp_refined[:, 2]

            Gp_x *= phase; Gp_y *= phase; Gp_z *= phase
            Gp = np.stack([Gp_x, Gp_y, Gp_z], axis=1)  # (n1, 3, n2)
            return Gp

        elif key == 'H1p':
            Gp = self.eval(k, 'Gp')
            if self.p1 is self.p2:
                H1p = Gp.copy()
                nvec = self.p1.nvec
                idx = np.arange(len(nvec))
                H1p[idx, :, idx] += 2.0 * np.pi * nvec.T
                return H1p
            return Gp

        elif key == 'H2p':
            Gp = self.eval(k, 'Gp')
            if self.p1 is self.p2:
                H2p = Gp.copy()
                nvec = self.p1.nvec
                idx = np.arange(len(nvec))
                H2p[idx, :, idx] -= 2.0 * np.pi * nvec.T
                return H2p
            return Gp

        else:
            raise ValueError("Unknown key: {}".format(key))

    def _eval_gpu(self, k, key, c):
        """GPU evaluation path for G/F/H1/H2/Gp/H1p/H2p.

        All arithmetic happens on the device (cupy).  When MNPBEM_GPU_NATIVE=1
        the cupy ndarray is returned directly so downstream BEM/Spectrum code
        can consume it without an unnecessary host round-trip.  Otherwise the
        result is brought back to host (numpy) for backward compatibility
        with the host-side numpy linear algebra.
        """
        import cupy as cp
        from ._numba_ret_kernels import (
            ret_G_pre_gpu, ret_F_norm_pre_gpu, ret_F_cart_pre_gpu,
            ret_Gp_pre_gpu, ret_phase_gpu,
            apply_phase_2d_gpu, apply_phase_3d_axis02_gpu,
            to_host, gpu_native_enabled,
        )

        native = gpu_native_enabled()
        _ret = (lambda x: x) if native else to_host

        d = c['d']
        inv_d = c['inv_d']
        inv_d2 = c['inv_d2']
        area2 = c['area2']

        # Cache the small device arrays needed for refinement overlay.
        if 'g_gpu' not in c and len(self.ind) > 0:
            c['g_gpu'] = cp.asarray(self.g) if self.g.size > 0 else None
            c['f_gpu'] = cp.asarray(self.f) if self.f.size > 0 else None
            c['row_gpu'] = cp.asarray(self.row)
            c['col_gpu'] = cp.asarray(self.col)
        if 'nvec_gpu' not in c:
            c['nvec_gpu'] = cp.asarray(self.p1.nvec)

        if key == 'G':
            G = ret_G_pre_gpu(inv_d, area2)
            if len(self.ind) > 0:
                ik_powers = cp.asarray(np.array([(1j * k)**n for n in range(self.order + 1)]))
                G_refined = c['g_gpu'] @ ik_powers
                G[c['row_gpu'], c['col_gpu']] = G_refined
            phase = ret_phase_gpu(d, k)
            apply_phase_2d_gpu(G, phase)
            return _ret(G)

        elif key == 'F':
            if self.deriv == 'cart':
                x, y, z = c['x'], c['y'], c['z']
                F = ret_F_cart_pre_gpu(inv_d, inv_d2, x, y, z, c['nvec_gpu'], area2, k)
                if len(self.ind) > 0:
                    ik_powers = cp.asarray(np.array([(1j * k)**n for n in range(self.order + 1)]))
                    nvec_ref = c['nvec_gpu'][c['row_gpu']]
                    f_g = c['f_gpu']
                    # F_refined[i] = sum_j sum_k nvec_ref[i,j] * f[i,j,k] * ik_powers[k]
                    F_refined = cp.einsum('ij,ijk,k->i', nvec_ref, f_g, ik_powers)
                    F[c['row_gpu'], c['col_gpu']] = F_refined
                phase = ret_phase_gpu(d, k)
                apply_phase_2d_gpu(F, phase)
                return _ret(F)
            else:
                n_dot_r = c['n_dot_r']
                F = ret_F_norm_pre_gpu(inv_d, inv_d2, n_dot_r, area2, k)
                if len(self.ind) > 0:
                    ik_powers = cp.asarray(np.array([(1j * k)**n for n in range(self.order + 1)]))
                    F_refined = c['f_gpu'] @ ik_powers
                    F[c['row_gpu'], c['col_gpu']] = F_refined
                phase = ret_phase_gpu(d, k)
                apply_phase_2d_gpu(F, phase)
                return _ret(F)

        elif key == 'H1':
            H1 = self.eval(k, 'F')
            if self.p1 is self.p2:
                if isinstance(H1, cp.ndarray):
                    n = min(H1.shape[0], H1.shape[1])
                    idx = cp.arange(n)
                    H1[idx, idx] = H1[idx, idx] + 2.0 * np.pi
                else:
                    np.fill_diagonal(H1, np.diag(H1) + 2.0 * np.pi)
            return H1

        elif key == 'H2':
            H2 = self.eval(k, 'F')
            if self.p1 is self.p2:
                if isinstance(H2, cp.ndarray):
                    n = min(H2.shape[0], H2.shape[1])
                    idx = cp.arange(n)
                    H2[idx, idx] = H2[idx, idx] - 2.0 * np.pi
                else:
                    np.fill_diagonal(H2, np.diag(H2) - 2.0 * np.pi)
            return H2

        elif key == 'Gp':
            x, y, z = c['x'], c['y'], c['z']
            Gp = ret_Gp_pre_gpu(inv_d, inv_d2, x, y, z, area2, k)
            if len(self.ind) > 0 and self.deriv == 'cart':
                ik_powers = cp.asarray(np.array([(1j * k)**n for n in range(self.order + 1)]))
                # f is (n_ref, 3, order+1) -> Gp_refined (n_ref, 3)
                Gp_refined = cp.einsum('ijk,k->ij', c['f_gpu'], ik_powers)
                Gp[c['row_gpu'], 0, c['col_gpu']] = Gp_refined[:, 0]
                Gp[c['row_gpu'], 1, c['col_gpu']] = Gp_refined[:, 1]
                Gp[c['row_gpu'], 2, c['col_gpu']] = Gp_refined[:, 2]
            phase = ret_phase_gpu(d, k)
            apply_phase_3d_axis02_gpu(Gp, phase)
            return _ret(Gp)

        elif key == 'H1p':
            Gp = self.eval(k, 'Gp')
            if self.p1 is self.p2:
                if isinstance(Gp, cp.ndarray):
                    H1p = Gp.copy()
                    nvec = c['nvec_gpu']
                    idx = cp.arange(nvec.shape[0])
                    H1p[idx, :, idx] = H1p[idx, :, idx] + 2.0 * np.pi * nvec
                    return H1p
                H1p = Gp.copy()
                nvec = self.p1.nvec
                idx = np.arange(len(nvec))
                H1p[idx, :, idx] += 2.0 * np.pi * nvec.T
                return H1p
            return Gp

        elif key == 'H2p':
            Gp = self.eval(k, 'Gp')
            if self.p1 is self.p2:
                if isinstance(Gp, cp.ndarray):
                    H2p = Gp.copy()
                    nvec = c['nvec_gpu']
                    idx = cp.arange(nvec.shape[0])
                    H2p[idx, :, idx] = H2p[idx, :, idx] - 2.0 * np.pi * nvec
                    return H2p
                H2p = Gp.copy()
                nvec = self.p1.nvec
                idx = np.arange(len(nvec))
                H2p[idx, :, idx] -= 2.0 * np.pi * nvec.T
                return H2p
            return Gp

        else:
            raise ValueError("Unknown key: {}".format(key))

    def __repr__(self):
        n_refined = len(self.ind) if hasattr(self, 'ind') else 0
        return ("GreenRetRefined(n1 = {}, n2 = {}, "
                "order = {}, n_refined = {})".format(self.p1.n, self.p2.n, self.order, n_refined))


# Test code
if __name__ == "__main__":
    print("Testing GreenRetRefined:")
    print("=" * 70)

    import sys
    sys.path.insert(0, '/home/user/MNPBEM')
    from mnpbem.geometry.particle import Particle

    # Create simple test particle
    verts = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
    ]) * 10.0

    faces = np.array([
        [0, 1, 2], [0, 2, 3],
        [4, 6, 5], [4, 7, 6],
        [0, 5, 1], [0, 4, 5],
        [2, 7, 3], [2, 6, 7],
        [0, 7, 4], [0, 3, 7],
        [1, 6, 2], [1, 5, 6],
    ])

    p = Particle(verts, faces)

    print("\nParticle: {} faces".format(p.n))

    # Create refined Green function
    print("\nCreating refined Green function...")
    g = GreenRetRefined(p, p, order=2, RelCutoff=3)

    print("{}".format(g))
    print("Refined elements: {}".format(len(g.ind)))
    print("  g shape: {}".format(g.g.shape))
    print("  f shape: {}".format(g.f.shape))

    # Test evaluation at 600 nm wavelength
    wavelength = 600.0  # nm
    k = 2 * np.pi / wavelength

    print("\nEvaluating at lambda={} nm (k={:.6f} nm^-1):".format(wavelength, k))

    G = g.eval(k, 'G')
    print("  G shape: {}".format(G.shape))
    print("  G diagonal range: [{:.2e}, {:.2e}]".format(np.min(np.abs(np.diag(G))), np.max(np.abs(np.diag(G)))))
    print("  G off-diag range: [{:.2e}, "
          "{:.2e}]".format(np.min(np.abs(G[~np.eye(p.n, dtype=bool)])),
                           np.max(np.abs(G[~np.eye(p.n, dtype=bool)]))))

    F = g.eval(k, 'F')
    print("  F shape: {}".format(F.shape))
    print("  F diagonal range: [{:.2e}, {:.2e}]".format(np.min(np.abs(np.diag(F))), np.max(np.abs(np.diag(F)))))

    H1 = g.eval(k, 'H1')
    H2 = g.eval(k, 'H2')
    print("  H1 diagonal: {} ...".format(np.diag(H1)[:3]))
    print("  H2 diagonal: {} ...".format(np.diag(H2)[:3]))
    print("  H1-H2 diagonal diff: {:.6f} (should be 4pi={:.6f})".format(np.mean(np.abs(np.diag(H1) - np.diag(H2))), 4 * np.pi))

    print("\n" + "=" * 70)
    print("✓ GreenRetRefined tests passed!")
