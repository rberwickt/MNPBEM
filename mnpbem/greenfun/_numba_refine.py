"""
Numba-accelerated kernels for Green-function refinement loops.

Targets the per-face Python loops in:
  * GreenRetRefined._refine_diagonal           (greenret_refined.py)
  * GreenRetRefined._refine_offdiagonal        (greenret_refined.py)
  * CompGreenStat._refine_greenstat (offdiag)  (compgreen_stat.py)

These loops dominate BEM assembly cost for medium / large meshes
(~70% of compgreen_init wall time at 3k+ faces) because each Python
iteration only does O(n_neighbors x n_quad_points) numpy work, so the
interpreter overhead becomes the bottleneck once the mesh has thousands
of faces.

Activation:
  - default: enabled when numba is importable
  - disable by setting MNPBEM_NUMBA_REFINE=0 (or MNPBEM_NUMBA=0)

Numerical contract:
  - fastmath = False everywhere (bit-identical to numpy reference within
    associativity differences in the per-face dot products).  The kernels
    intentionally accumulate over n_quad_points in the same order as the
    pre-existing numpy ``integrand @ w_face`` reduction so reductions
    follow MATLAB's row-major order.
"""

import os
import math
import numpy as np

try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


def numba_refine_enabled():
    """Return True iff numba refinement kernels should be used."""
    if not NUMBA_AVAILABLE:
        return False
    if os.environ.get('MNPBEM_NUMBA', '1') == '0':
        return False
    return os.environ.get('MNPBEM_NUMBA_REFINE', '1') != '0'


if NUMBA_AVAILABLE:

    @njit(cache = True, parallel = True, fastmath = False)
    def _ret_refine_offdiag_norm(
            face_starts, face_ends, nb_starts, nb_ends, nb_idx, iface_arr,
            pos_face_flat, w_face_flat,
            pos1, nvec1, pos2_face,
            order_p1, fact_inv, g_out, f_out):
        """
        Numba kernel for GreenRetRefined._refine_offdiagonal (deriv='norm').

        Layout:
          face_starts/face_ends : (n_face_refine + 1,) CSR over integration points
          nb_starts/nb_ends     : (n_face_refine + 1,) CSR over neighbors
          nb_idx[nb_starts[k]:nb_ends[k]]    : neighbor row indices for face k
          iface_arr[nb_starts[k]:nb_ends[k]] : refine-index for (nb, face2)
          pos_face_flat (n_points_total, 3) : integration positions
          w_face_flat   (n_points_total,)   : integration weights
          pos1 (n_total_faces, 3) : centroids (source mesh)
          nvec1 (n_total_faces, 3) : normals
          pos2_face (n_face_refine, 3) : centroid of refined column face
          fact_inv (order+1,) : 1.0 / n!  (precomputed)
          g_out (n_refined, order+1) complex128
          f_out (n_refined, order+1) complex128

        Reductions are performed per face in the inner serial loop so
        ordering matches numpy's @ reduction; outer prange parallelises
        across the refined faces (independent rows of g_out/f_out).
        """
        n_face = face_ends.shape[0] - 1
        order = order_p1 - 1
        for k in prange(n_face):
            ip0 = face_starts[k]
            ip1 = face_ends[k]
            nq = ip1 - ip0
            nb0 = nb_starts[k]
            nb1 = nb_ends[k]

            face_x = pos2_face[k, 0]
            face_y = pos2_face[k, 1]
            face_z = pos2_face[k, 2]

            for q in range(nb0, nb1):
                nb = nb_idx[q]
                iref = iface_arr[q]
                p1x = pos1[nb, 0]
                p1y = pos1[nb, 1]
                p1z = pos1[nb, 2]
                nx = nvec1[nb, 0]
                ny = nvec1[nb, 1]
                nz = nvec1[nb, 2]

                vx0 = face_x - p1x
                vy0 = face_y - p1y
                vz0 = face_z - p1z
                r0 = math.sqrt(vx0 * vx0 + vy0 * vy0 + vz0 * vz0)

                # Accumulators for g_n (n=0..order) and f_n
                # g_n = sum(w * (r-r0)^n / r) / n!
                # f_n (norm, n>=1) = sum(w * n_dot_r * (term1 + term2)) / 1.
                #     where term1 = -(r-r0)^n / (r^3 * n!) and
                #           term2 =  (r-r0)^(n-1) / (r^2 * (n-1)!)
                # f_0 (norm) = -sum(w * n_dot_r / r^3)
                # We unroll an explicit loop over n inside per-point so we
                # only touch each integration point once.
                # Allocate small per-thread scratch via stack tuple of len(order+1).
                # Numba supports np.zeros inside njit; use a fixed-size array.
                gn = np.zeros(order_p1, dtype = np.complex128)
                fn = np.zeros(order_p1, dtype = np.complex128)

                for ip in range(ip0, ip1):
                    px = pos_face_flat[ip, 0]
                    py = pos_face_flat[ip, 1]
                    pz = pos_face_flat[ip, 2]
                    w = w_face_flat[ip]

                    rx = p1x - px
                    ry = p1y - py
                    rz = p1z - pz
                    r = math.sqrt(rx * rx + ry * ry + rz * rz)
                    if r < 2.220446049250313e-16:
                        r = 2.220446049250313e-16
                    inv_r = 1.0 / r
                    inv_r2 = inv_r * inv_r
                    inv_r3 = inv_r2 * inv_r

                    n_dot_r = nx * rx + ny * ry + nz * rz
                    delta = r - r0

                    # Powers of delta accumulated incrementally
                    delta_pow = 1.0
                    delta_pow_prev = 0.0  # used as delta^(n-1); starts at delta^-1, handled by n loop

                    for n in range(order_p1):
                        # g_n contribution
                        gn[n] += w * delta_pow * inv_r * fact_inv[n]
                        if n == 0:
                            # f_0 = -sum(w * n_dot_r / r^3)
                            fn[0] += -w * n_dot_r * inv_r3
                        else:
                            term1 = -delta_pow * inv_r3 * fact_inv[n]
                            term2 = delta_pow_prev * inv_r2 * fact_inv[n - 1]
                            fn[n] += w * n_dot_r * (term1 + term2)
                        delta_pow_prev = delta_pow
                        delta_pow *= delta

                for n in range(order_p1):
                    g_out[iref, n] = gn[n]
                    f_out[iref, n] = fn[n]

    @njit(cache = True, parallel = True, fastmath = False)
    def _ret_refine_offdiag_cart(
            face_starts, face_ends, nb_starts, nb_ends, nb_idx, iface_arr,
            pos_face_flat, w_face_flat,
            pos1, pos2_face,
            order_p1, fact_inv, g_out, f_out):
        """
        Cart-deriv variant: f_out has shape (n_refined, 3, order+1).
        f_n_x = sum(w * x * f_scalar_n)  with f_scalar same as norm path.
        f_0 uses f_scalar = -1/r^3 (without the (r-r0)^n*term2 pieces).

        Note: g_out semantics identical to norm path (no nvec involvement).
        """
        n_face = face_ends.shape[0] - 1
        for k in prange(n_face):
            ip0 = face_starts[k]
            ip1 = face_ends[k]
            nb0 = nb_starts[k]
            nb1 = nb_ends[k]

            face_x = pos2_face[k, 0]
            face_y = pos2_face[k, 1]
            face_z = pos2_face[k, 2]

            for q in range(nb0, nb1):
                nb = nb_idx[q]
                iref = iface_arr[q]
                p1x = pos1[nb, 0]
                p1y = pos1[nb, 1]
                p1z = pos1[nb, 2]

                vx0 = face_x - p1x
                vy0 = face_y - p1y
                vz0 = face_z - p1z
                r0 = math.sqrt(vx0 * vx0 + vy0 * vy0 + vz0 * vz0)

                gn = np.zeros(order_p1, dtype = np.complex128)
                fxn = np.zeros(order_p1, dtype = np.complex128)
                fyn = np.zeros(order_p1, dtype = np.complex128)
                fzn = np.zeros(order_p1, dtype = np.complex128)

                for ip in range(ip0, ip1):
                    px = pos_face_flat[ip, 0]
                    py = pos_face_flat[ip, 1]
                    pz = pos_face_flat[ip, 2]
                    w = w_face_flat[ip]

                    rx = p1x - px
                    ry = p1y - py
                    rz = p1z - pz
                    r = math.sqrt(rx * rx + ry * ry + rz * rz)
                    if r < 2.220446049250313e-16:
                        r = 2.220446049250313e-16
                    inv_r = 1.0 / r
                    inv_r2 = inv_r * inv_r
                    inv_r3 = inv_r2 * inv_r

                    delta = r - r0

                    delta_pow = 1.0
                    delta_pow_prev = 0.0

                    for n in range(order_p1):
                        gn[n] += w * delta_pow * inv_r * fact_inv[n]
                        if n == 0:
                            fs = -inv_r3
                            fxn[0] += w * rx * fs
                            fyn[0] += w * ry * fs
                            fzn[0] += w * rz * fs
                        else:
                            term1 = -delta_pow * inv_r3 * fact_inv[n]
                            term2 = delta_pow_prev * inv_r2 * fact_inv[n - 1]
                            fs = term1 + term2
                            fxn[n] += w * rx * fs
                            fyn[n] += w * ry * fs
                            fzn[n] += w * rz * fs
                        delta_pow_prev = delta_pow
                        delta_pow *= delta

                for n in range(order_p1):
                    g_out[iref, n] = gn[n]
                    f_out[iref, 0, n] = fxn[n]
                    f_out[iref, 1, n] = fyn[n]
                    f_out[iref, 2, n] = fzn[n]

    @njit(cache = True, parallel = True, fastmath = False)
    def _ret_refine_diag_norm(
            point_starts, point_ends,
            pos1_face, nvec1_face, iref_face,
            pos_quad, w_quad,
            order_p1, fact_inv, g_out, f_out):
        """
        Numba kernel for GreenRetRefined._refine_diagonal (deriv='norm').

        Diagonal refinement integrates over a single face (polar quad).
        Per-face accumulation is independent, so prange across faces is
        safe.
        """
        n_face = point_ends.shape[0] - 1
        for k in prange(n_face):
            ip0 = point_starts[k]
            ip1 = point_ends[k]
            iref = iref_face[k]
            p1x = pos1_face[k, 0]
            p1y = pos1_face[k, 1]
            p1z = pos1_face[k, 2]
            nx = nvec1_face[k, 0]
            ny = nvec1_face[k, 1]
            nz = nvec1_face[k, 2]

            gn = np.zeros(order_p1, dtype = np.complex128)
            fn = np.zeros(order_p1, dtype = np.complex128)

            for ip in range(ip0, ip1):
                px = pos_quad[ip, 0]
                py = pos_quad[ip, 1]
                pz = pos_quad[ip, 2]
                w = w_quad[ip]
                vx = p1x - px
                vy = p1y - py
                vz = p1z - pz
                r = math.sqrt(vx * vx + vy * vy + vz * vz)
                if r < 2.220446049250313e-16:
                    r = 2.220446049250313e-16
                n_dot_r = nx * vx + ny * vy + nz * vz

                # g_n = sum(w * r^(n-1)) / n!
                # f_n = sum(w * (n-1) * n_dot_r * r^(n-3)) / n!
                # Use incremental power: r_pow tracks r^(n-1); start at r^-1
                inv_r = 1.0 / r
                r_pow_nm1 = inv_r  # r^(-1) for n=0
                inv_r3 = inv_r * inv_r * inv_r

                for n in range(order_p1):
                    gn[n] += w * r_pow_nm1 * fact_inv[n]
                    # r^(n-3) = r^(n-1) * r^-2
                    fn[n] += w * (n - 1) * n_dot_r * (r_pow_nm1 * inv_r * inv_r) * fact_inv[n]
                    r_pow_nm1 *= r

            for n in range(order_p1):
                g_out[iref, n] = gn[n]
                f_out[iref, n] = fn[n]


def build_offdiag_csr(ir_array, reface, row_indices, w_sparse, refine_map):
    """
    Build CSR-style flat arrays for the offdiagonal kernel.

    Parameters
    ----------
    ir_array : (n_faces, n_faces) ndarray
        Refinement type matrix (1 for offdiag refinement target).
    reface : (n_face_refine,) ndarray
        Column indices to be refined.
    row_indices : (n_points_total,) ndarray
        Face index (into reface) for each integration point produced by
        Particle.quad(reface).
    w_sparse : scipy.sparse.csr_matrix or compatible (n_face_refine, n_points_total)
        Sparse integration weights.
    refine_map : dict (row, col) -> int
        Mapping to pre-computed refinement output indices.

    Returns
    -------
    face_starts, face_ends, pos_face_flat, w_face_flat,
    nb_starts, nb_ends, nb_idx, iface_arr, pos2_face_arr
    """
    # Sort points by face so each face owns a contiguous block.
    order = np.argsort(row_indices, kind = 'stable')
    sorted_rows = row_indices[order]
    n_face_refine = len(reface)
    face_starts = np.zeros(n_face_refine + 1, dtype = np.int64)
    # cumulative count per face
    counts = np.bincount(sorted_rows, minlength = n_face_refine)
    np.cumsum(counts, out = face_starts[1:])
    face_ends = face_starts[1:].copy()
    # Reorder weight (CSR row corresponds to face_idx)
    return order, face_starts, face_ends


def _flatten_pos_w(reface, row_indices, pos_all, w_sparse):
    """Build per-face contiguous flat arrays for pos and w."""
    n_face_refine = len(reface)
    # Sort permutation by row
    order = np.argsort(row_indices, kind = 'stable')
    sorted_rows = row_indices[order]
    pos_flat = np.ascontiguousarray(pos_all[order], dtype = np.float64)
    counts = np.bincount(sorted_rows, minlength = n_face_refine)
    starts = np.zeros(n_face_refine + 1, dtype = np.int64)
    np.cumsum(counts, out = starts[1:])
    # Build w_face_flat by selecting w_sparse[i, :] non-zero entries.
    # Each face's row of w_sparse should agree with the same number of
    # non-zero columns as `counts[i]` and follow the same column ordering
    # as `pos_all[row_indices == i]`.
    w_csr = w_sparse.tocsr() if hasattr(w_sparse, 'tocsr') else w_sparse
    w_flat = np.zeros(starts[-1], dtype = np.float64)
    # Re-create per-face w mapping by using the sparse data directly.
    # For each face i, the i-th row's nonzero columns give weights.
    # Particle.quad layout guarantees the integration point set for
    # face i is exactly the non-zero columns of row i.
    indptr = w_csr.indptr
    indices = w_csr.indices
    data = w_csr.data
    for i in range(n_face_refine):
        rs, re = indptr[i], indptr[i + 1]
        cols_i = indices[rs:re]
        w_row = data[rs:re]
        n_pts_face = starts[i + 1] - starts[i]
        # Filter zero weights to match pre-existing semantics; pos ordering
        # is by row_indices so we need to align w_row with that.
        # The original code did
        #   pos_face = pos_all[row_indices == i]
        #   w_row = w_sparse[i,:].toarray().ravel(); w_face = w_row[w_row != 0]
        # which yields w_face in column-order (since row_csr stores cols
        # ascending).  pos_all[row_indices == i] is also in column-order
        # provided row_indices was generated by quad() in that same order
        # (which Particle.quad does).  So we can simply assign w_row.
        nz_mask = w_row > 0
        w_pos = w_row[nz_mask] if not nz_mask.all() else w_row
        if w_pos.shape[0] != n_pts_face:
            # Fall back: derive w from face_mask path to keep semantics.
            # This path should be rare; still numerically equivalent.
            face_mask = (row_indices == i)
            w_full = np.asarray(w_csr[i, :].toarray()).ravel()
            w_pos = w_full[w_full != 0]
            pos_flat[starts[i]:starts[i + 1]] = pos_all[face_mask]
        w_flat[starts[i]:starts[i + 1]] = w_pos
    return starts[:-1].copy(), starts[1:].copy(), pos_flat, w_flat


def build_offdiag_arrays(
        ir_array, reface, row_indices, pos_all, w_sparse, refine_map):
    """
    Build all CSR-flat arrays needed by `_ret_refine_offdiag_*` kernels.

    Returns
    -------
    dict with keys:
      face_starts, face_ends           int64 (n_face,)
      pos_face_flat, w_face_flat       float64
      nb_starts, nb_ends               int64
      nb_idx                           int64
      iface_arr                        int64
    """
    face_starts, face_ends, pos_face_flat, w_face_flat = _flatten_pos_w(
        reface, row_indices, pos_all, w_sparse)

    n_face = len(reface)
    # Build neighbor CSR per face from ir_array column slices.
    nb_lists = []
    iref_lists = []
    for face_idx in range(n_face):
        face = reface[face_idx]
        nb = np.where(ir_array[:, face] == 1)[0]
        nb_lists.append(nb.astype(np.int64))
        iref = np.empty(len(nb), dtype = np.int64)
        for j, nb_row in enumerate(nb):
            iref[j] = refine_map[(int(nb_row), int(face))]
        iref_lists.append(iref)

    nb_starts = np.zeros(n_face + 1, dtype = np.int64)
    nb_counts = np.array([len(x) for x in nb_lists], dtype = np.int64)
    np.cumsum(nb_counts, out = nb_starts[1:])
    nb_idx = np.concatenate(nb_lists) if nb_lists else np.zeros(0, dtype = np.int64)
    iface_arr = np.concatenate(iref_lists) if iref_lists else np.zeros(0, dtype = np.int64)

    return {
        'face_starts': face_starts,
        'face_ends': face_ends,
        'pos_face_flat': pos_face_flat,
        'w_face_flat': w_face_flat,
        'nb_starts': nb_starts[:-1].copy(),
        'nb_ends': nb_starts[1:].copy(),
        'nb_idx': nb_idx,
        'iface_arr': iface_arr,
    }
