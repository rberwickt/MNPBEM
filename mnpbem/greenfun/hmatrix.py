import os
import sys
import numpy as np
from typing import Optional, Tuple, Any, List, Callable, Dict

from .clustertree import ClusterTree

try:
    import numba
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False


if _HAS_NUMBA:
    @numba.njit(cache=True)
    def _aca_subtract_row(U_mat, V_mat, rank, pivot_row, row_vals):
        """Subtract previous approximants from a row (Numba-accelerated, real)."""
        for j in range(rank):
            scale = U_mat[pivot_row, j]
            for i in range(len(row_vals)):
                row_vals[i] -= scale * V_mat[i, j]
        return row_vals

    @numba.njit(cache=True)
    def _aca_subtract_col(U_mat, V_mat, rank, pivot_col, col_vals):
        """Subtract previous approximants from a column (Numba-accelerated, real)."""
        for j in range(rank):
            scale = V_mat[pivot_col, j]
            for i in range(len(col_vals)):
                col_vals[i] -= scale * U_mat[i, j]
        return col_vals

    @numba.njit(cache=True)
    def _aca_cross_terms(U_mat, V_mat, rank, u_new, v_new):
        """Compute cross-terms for Frobenius norm (Numba-accelerated, real)."""
        cross = 0.0
        m = len(u_new)
        n = len(v_new)
        for j in range(rank - 1):
            dot_u = 0.0
            dot_v = 0.0
            for i in range(m):
                dot_u += U_mat[i, j] * u_new[i]
            for i in range(n):
                dot_v += V_mat[i, j] * v_new[i]
            cross += 2.0 * dot_u * dot_v
        return cross

    # ----------------------------------------------------------------
    # Complex128 ACA kernels (separate compiled paths so retarded /
    # layered Green functions also benefit from numba acceleration).
    # ----------------------------------------------------------------
    @numba.njit(cache=True)
    def _aca_subtract_row_c(U_mat, V_mat, rank, pivot_row, row_vals):
        """Subtract previous approximants from a row (Numba, complex128)."""
        for j in range(rank):
            scale = U_mat[pivot_row, j]
            for i in range(len(row_vals)):
                row_vals[i] -= scale * V_mat[i, j]
        return row_vals

    @numba.njit(cache=True)
    def _aca_subtract_col_c(U_mat, V_mat, rank, pivot_col, col_vals):
        """Subtract previous approximants from a column (Numba, complex128)."""
        for j in range(rank):
            scale = V_mat[pivot_col, j]
            for i in range(len(col_vals)):
                col_vals[i] -= scale * U_mat[i, j]
        return col_vals

    @numba.njit(cache=True)
    def _aca_cross_terms_c(U_mat, V_mat, rank, u_new, v_new):
        """Cross-terms for Frobenius norm (Numba, complex128).

        For complex matrices the residual Frobenius norm uses Hermitian inner
        products: <U_j, u_new> = sum conj(U_j) * u_new.  Returning the real
        part keeps frobenius_sq real while remaining mathematically correct.
        """
        cross = 0.0 + 0.0j
        m = len(u_new)
        n = len(v_new)
        for j in range(rank - 1):
            dot_u = 0.0 + 0.0j
            dot_v = 0.0 + 0.0j
            for i in range(m):
                dot_u += np.conj(U_mat[i, j]) * u_new[i]
            for i in range(n):
                dot_v += np.conj(V_mat[i, j]) * v_new[i]
            cross += 2.0 * (dot_u * dot_v).real
        return cross.real


class HMatrix(object):

    # MATLAB: @hmatrix
    # Hierarchical matrix using cluster tree and admissibility.
    # Stores dense blocks and low-rank (UV) blocks.
    # See S. Boerm et al., Eng. Analysis with Bound. Elem. 27, 405 (2003).

    def __init__(self,
            tree: Optional[ClusterTree] = None,
            htol: float = 1e-6,
            kmax: int = 100,
            fadmiss: Optional[Callable] = None):

        self.tree = tree
        self.htol = htol
        self.kmax = kmax

        # Block indices: dense blocks (row1, col1) and low-rank blocks (row2, col2)
        self.row1 = np.array([], dtype = np.int64)  # tree node indices for dense blocks
        self.col1 = np.array([], dtype = np.int64)
        self.row2 = np.array([], dtype = np.int64)  # tree node indices for low-rank blocks
        self.col2 = np.array([], dtype = np.int64)

        # Storage: lists of arrays
        self.val = []   # dense blocks: list of 2D arrays
        self.lhs = []   # low-rank left factors: list of 2D arrays (m x k)
        self.rhs = []   # low-rank right factors: list of 2D arrays (n x k)

        if tree is not None:
            self._init(tree, fadmiss = fadmiss)

    def _init(self,
            tree: ClusterTree,
            fadmiss: Optional[Callable] = None) -> None:

        # MATLAB: @hmatrix/private/init.m
        self.tree = tree

        # Compute admissibility
        admiss = tree.admissibility(tree, fadmiss = fadmiss)

        # Separate dense (==2) and low-rank (==1) blocks
        row1_list = []
        col1_list = []
        row2_list = []
        col2_list = []

        for (i1, i2), val in admiss.items():
            if val == 2:
                row1_list.append(i1)
                col1_list.append(i2)
            elif val == 1:
                row2_list.append(i1)
                col2_list.append(i2)

        self.row1 = np.array(row1_list, dtype = np.int64)
        self.col1 = np.array(col1_list, dtype = np.int64)
        self.row2 = np.array(row2_list, dtype = np.int64)
        self.col2 = np.array(col2_list, dtype = np.int64)

        # Initialize empty storage
        self.val = [None] * len(self.row1)
        self.lhs = [None] * len(self.row2)
        self.rhs = [None] * len(self.row2)

    def aca(self, fun: Callable) -> 'HMatrix':

        # MATLAB: @hmatrix/aca.m
        # Fills the H-matrix using Adaptive Cross Approximation.
        # fun(row, col) returns matrix values for given row/col indices (0-based, particle ordering)

        tree = self.tree
        # Map from cluster to particle indices
        ind_c2p = tree.ind[:, 0]

        # Wrapped function: takes cluster indices, returns values
        def fun2(row_c: np.ndarray, col_c: np.ndarray) -> np.ndarray:
            return fun(ind_c2p[row_c], ind_c2p[col_c])

        # Compute dense blocks
        for i in range(len(self.row1)):
            indr = tree.cind[self.row1[i]]
            indc = tree.cind[self.col1[i]]
            rows = np.arange(indr[0], indr[1] + 1, dtype = np.int64)
            cols = np.arange(indc[0], indc[1] + 1, dtype = np.int64)
            row_grid, col_grid = np.meshgrid(rows, cols, indexing = 'ij')
            self.val[i] = fun2(row_grid.ravel(), col_grid.ravel()).reshape(row_grid.shape)

        # Compute low-rank blocks using ACA
        for i in range(len(self.row2)):
            indr = tree.cind[self.row2[i]]
            indc = tree.cind[self.col2[i]]
            rows = np.arange(indr[0], indr[1] + 1, dtype = np.int64)
            cols = np.arange(indc[0], indc[1] + 1, dtype = np.int64)

            lhs, rhs = self._aca_block(fun2, rows, cols, self.htol, self.kmax)
            self.lhs[i] = lhs
            self.rhs[i] = rhs

        return self

    def _aca_block(self,
            fun: Callable,
            rows: np.ndarray,
            cols: np.ndarray,
            htol: float,
            kmax: int) -> Tuple[np.ndarray, np.ndarray]:

        # Partially-pivoted Adaptive Cross Approximation for a single block
        # Returns (U, V) such that A ~ U @ V.T

        # Bug 3 fix: ensure rows/cols index arrays live on host (numpy).
        # Some GPU-aware callers may hand us cupy index arrays; the rest of
        # this CPU ACA path uses np.argmax / np.full / boolean masks which
        # cupy refuses to index implicitly with numpy ints.  Coerce once.
        if hasattr(rows, 'get') and not isinstance(rows, np.ndarray):
            rows = rows.get()
        else:
            rows = np.asarray(rows)
        if hasattr(cols, 'get') and not isinstance(cols, np.ndarray):
            cols = cols.get()
        else:
            cols = np.asarray(cols)

        m = len(rows)
        n = len(cols)
        max_rank = min(m, n, kmax)

        # Probe dtype from function output
        probe = fun(rows[:1], cols[:1])
        if hasattr(probe, 'get') and not isinstance(probe, np.ndarray):
            probe = probe.get()
        out_dtype = np.complex128 if np.iscomplexobj(probe) else np.float64

        # Pre-allocate U and V matrices (grow columns as needed)
        U_mat = np.empty((m, max_rank), dtype=out_dtype)
        V_mat = np.empty((n, max_rank), dtype=out_dtype)
        rank = 0

        # Boolean masks for used rows/columns
        used_row_mask = np.zeros(m, dtype=bool)
        used_col_mask = np.zeros(n, dtype=bool)

        # Start with row 0
        pivot_row_local = 0
        frobenius_sq = 0.0

        for k in range(max_rank):
            # Compute row of residual at pivot_row_local
            row_global = int(rows[pivot_row_local])
            row_c = np.full(n, row_global, dtype=np.int64)
            row_vals = fun(row_c, cols)
            # Bug 3 fix: cupy → numpy if a GPU-aware ``fun`` returned cupy.
            if hasattr(row_vals, 'get') and not isinstance(row_vals, np.ndarray):
                row_vals = row_vals.get()

            # Subtract contributions from previous approximants
            if rank > 0:
                if _HAS_NUMBA and out_dtype == np.float64:
                    row_vals = _aca_subtract_row(U_mat, V_mat, rank, pivot_row_local, row_vals)
                elif _HAS_NUMBA and out_dtype == np.complex128:
                    row_vals = _aca_subtract_row_c(U_mat, V_mat, rank, pivot_row_local, row_vals)
                else:
                    row_vals -= V_mat[:, :rank] @ U_mat[pivot_row_local, :rank]

            # Find pivot column (max absolute value in unused columns)
            abs_row = np.abs(row_vals)
            abs_row[used_col_mask] = 0.0
            pivot_col_local = int(np.argmax(abs_row))
            pivot_val = row_vals[pivot_col_local]

            if np.abs(pivot_val) < 1e-15:
                break

            # Compute column of residual at pivot_col_local
            # Bug 3 fix: explicit int() so cols[pivot_col_local] never hits
            # cupy's implicit-numpy-index refusal even if cols was cupy.
            col_global = int(cols[pivot_col_local])
            col_c = np.full(m, col_global, dtype=np.int64)
            col_vals = fun(rows, col_c)
            if hasattr(col_vals, 'get') and not isinstance(col_vals, np.ndarray):
                col_vals = col_vals.get()

            # Subtract contributions from previous approximants
            if rank > 0:
                if _HAS_NUMBA and out_dtype == np.float64:
                    col_vals = _aca_subtract_col(U_mat, V_mat, rank, pivot_col_local, col_vals)
                elif _HAS_NUMBA and out_dtype == np.complex128:
                    col_vals = _aca_subtract_col_c(U_mat, V_mat, rank, pivot_col_local, col_vals)
                else:
                    col_vals -= U_mat[:, :rank] @ V_mat[pivot_col_local, :rank]

            # New rank-1 term: u = col_vals / pivot_val, v = row_vals
            u_new = col_vals / pivot_val
            v_new = row_vals.copy()

            U_mat[:, rank] = u_new
            V_mat[:, rank] = v_new
            rank += 1

            used_row_mask[pivot_row_local] = True
            used_col_mask[pivot_col_local] = True

            # Convergence check.  For complex matrices Frobenius norm uses
            # |u|^2 = <u,u> with Hermitian inner product, so use vdot.
            if out_dtype == np.complex128:
                u_norm_sq = np.vdot(u_new, u_new).real
                v_norm_sq = np.vdot(v_new, v_new).real
            else:
                u_norm_sq = float(np.sum(u_new * u_new))
                v_norm_sq = float(np.sum(v_new * v_new))
            new_term_sq = u_norm_sq * v_norm_sq

            # Cross-terms
            if rank > 1:
                if _HAS_NUMBA and out_dtype == np.float64:
                    cross_terms = _aca_cross_terms(U_mat, V_mat, rank, u_new, v_new)
                elif _HAS_NUMBA and out_dtype == np.complex128:
                    cross_terms = _aca_cross_terms_c(U_mat, V_mat, rank, u_new, v_new)
                elif out_dtype == np.complex128:
                    dot_u = np.conj(U_mat[:, :rank - 1].T) @ u_new
                    dot_v = np.conj(V_mat[:, :rank - 1].T) @ v_new
                    cross_terms = 2.0 * float(np.real(np.dot(dot_u, dot_v)))
                else:
                    cross_terms = 2.0 * np.dot(
                        U_mat[:, :rank - 1].T @ u_new,
                        V_mat[:, :rank - 1].T @ v_new)
            else:
                cross_terms = 0.0
            frobenius_sq += new_term_sq + cross_terms

            if frobenius_sq > 0 and np.sqrt(new_term_sq) < htol * np.sqrt(abs(frobenius_sq)):
                break

            # Choose next pivot row: row with max |u_new| among unused rows
            abs_u = np.abs(u_new)
            abs_u[used_row_mask] = 0.0
            pivot_row_local = int(np.argmax(abs_u))

        if rank == 0:
            return np.zeros((m, 1), dtype=out_dtype), np.zeros((n, 1), dtype=out_dtype)

        return U_mat[:, :rank].copy(), V_mat[:, :rank].copy()

    def full(self, xp: Any = None) -> Any:

        # MATLAB: @hmatrix/full.m
        # Convert H-matrix to full dense matrix.
        #
        # Bug 5 fix (v1.5.2): when any of ``self.val`` / ``self.lhs`` /
        # ``self.rhs`` lives on a CUDA device (cupy ndarray), the host
        # numpy buffer used to live in v1.5.1 raised ``TypeError: Implicit
        # conversion to a NumPy array is not allowed`` on the slice
        # assignment.  We now auto-detect the backend by sniffing the
        # blocks for cupy ndarrays; if any are found, ``mat`` is allocated
        # on the GPU and any leftover numpy block is promoted via
        # ``cupy.asarray``.  Conversely, when ``xp`` is forced to numpy by
        # the caller, cupy blocks are pulled to host with ``.get()``.
        # This keeps every existing CPU caller bit-identical while letting
        # the BEMRetIter dense-LU preconditioner build run end-to-end on
        # GPU for Tier-3 (12672-face) Au@Ag.
        tree = self.tree
        n = tree.n

        # Auto-detect GPU presence in any block.
        on_gpu = False
        for blk_list in (self.val, self.lhs, self.rhs):
            for blk in blk_list:
                if blk is not None and hasattr(blk, 'get') and not isinstance(blk, np.ndarray):
                    on_gpu = True
                    break
            if on_gpu:
                break

        xp_was_auto = (xp is None)
        if xp is None:
            if on_gpu:
                import cupy as _xp_module
                xp = _xp_module
            else:
                xp = np

        is_cupy_backend = (xp is not np)

        # Check if any block is complex (works for both numpy and cupy).
        is_complex = False
        for v in self.val:
            if v is not None and np.issubdtype(v.dtype, np.complexfloating):
                is_complex = True
                break
        if not is_complex:
            for l in self.lhs:
                if l is not None and np.issubdtype(l.dtype, np.complexfloating):
                    is_complex = True
                    break
        if not is_complex:
            for r in self.rhs:
                if r is not None and np.issubdtype(r.dtype, np.complexfloating):
                    is_complex = True
                    break

        out_dtype = np.complex128 if is_complex else np.float64
        mat = xp.zeros((n, n), dtype = out_dtype)

        def _cast(blk: Any) -> Any:
            # Bring a block to the destination backend.  Cupy blocks expose
            # ``.get()`` returning numpy; numpy blocks accept
            # ``cupy.asarray``.
            blk_is_cupy = hasattr(blk, 'get') and not isinstance(blk, np.ndarray)
            if is_cupy_backend and not blk_is_cupy:
                return xp.asarray(blk)
            if (not is_cupy_backend) and blk_is_cupy:
                return blk.get()
            return blk

        # Fill dense blocks
        for i in range(len(self.row1)):
            if self.val[i] is None:
                continue
            r_start = tree.cind[self.row1[i], 0]
            r_end = tree.cind[self.row1[i], 1] + 1
            c_start = tree.cind[self.col1[i], 0]
            c_end = tree.cind[self.col1[i], 1] + 1
            mat[r_start:r_end, c_start:c_end] = _cast(self.val[i])

        # Fill low-rank blocks
        for i in range(len(self.row2)):
            if self.lhs[i] is None or self.rhs[i] is None:
                continue
            r_start = tree.cind[self.row2[i], 0]
            r_end = tree.cind[self.row2[i], 1] + 1
            c_start = tree.cind[self.col2[i], 0]
            c_end = tree.cind[self.col2[i], 1] + 1
            lhs_blk = _cast(self.lhs[i])
            rhs_blk = _cast(self.rhs[i])
            # lhs @ rhs.T (kept on the destination backend)
            mat[r_start:r_end, c_start:c_end] = lhs_blk @ rhs_blk.T

        # Transform from cluster ordering to particle ordering.  Fancy
        # indexing materialises a fresh N x N buffer; on GPU this would
        # double the working set and OOM the 49 GB A6000 at Tier-3
        # (12672 face -> 50 GB matrix).  Pull to host first when the
        # operand lives on GPU; the immediate downstream consumer
        # (BEMRetIter._init_precond) calls ``to_host(...)`` on the
        # result anyway.  Callers wanting to keep the result on GPU can
        # pass ``xp=cupy`` and accept the extra device allocation.
        perm = tree.ind[:, 1]
        if is_cupy_backend and xp_was_auto:
            mat_h = xp.asnumpy(mat)
            del mat
            return mat_h[np.ix_(perm, perm)]
        if is_cupy_backend:
            perm_xp = xp.asarray(perm)
            return mat[xp.ix_(perm_xp, perm_xp)]
        return mat[np.ix_(perm, perm)]

    def mtimes_vec(self, v: np.ndarray) -> np.ndarray:

        # MATLAB: mtimes2 - H-matrix times dense vector/matrix
        tree = self.tree
        n = tree.n

        # Convert to cluster ordering
        if v.ndim == 1:
            v_cluster = tree.part2cluster(v)
            result = np.zeros(n, dtype = v.dtype)
        else:
            v_cluster = tree.part2cluster(v)
            result = np.zeros((n, v.shape[1]), dtype = v.dtype)

        # Dense blocks
        for i in range(len(self.row1)):
            if self.val[i] is None:
                continue
            r_start = tree.cind[self.row1[i], 0]
            r_end = tree.cind[self.row1[i], 1] + 1
            c_start = tree.cind[self.col1[i], 0]
            c_end = tree.cind[self.col1[i], 1] + 1

            if v.ndim == 1:
                result[r_start:r_end] += self.val[i] @ v_cluster[c_start:c_end]
            else:
                result[r_start:r_end] += self.val[i] @ v_cluster[c_start:c_end]

        # Low-rank blocks
        for i in range(len(self.row2)):
            if self.lhs[i] is None or self.rhs[i] is None:
                continue
            r_start = tree.cind[self.row2[i], 0]
            r_end = tree.cind[self.row2[i], 1] + 1
            c_start = tree.cind[self.col2[i], 0]
            c_end = tree.cind[self.col2[i], 1] + 1

            # lhs @ (rhs.T @ v)
            if v.ndim == 1:
                tmp = self.rhs[i].T @ v_cluster[c_start:c_end]
                result[r_start:r_end] += self.lhs[i] @ tmp
            else:
                tmp = self.rhs[i].T @ v_cluster[c_start:c_end]
                result[r_start:r_end] += self.lhs[i] @ tmp

        # Convert back to particle ordering
        return tree.cluster2part(result)

    def __matmul__(self, other: Any) -> Any:

        if isinstance(other, np.ndarray):
            return self.mtimes_vec(other)
        elif isinstance(other, HMatrix):
            return self._mtimes_hmat(other)
        else:
            raise TypeError('[error] Unsupported type for H-matrix multiplication')

    def __rmul__(self, scalar: float) -> 'HMatrix':

        # MATLAB: mtimes with scalar * hmatrix
        result = self._copy()
        result.val = [scalar * v if v is not None else None for v in result.val]
        result.lhs = [scalar * l if l is not None else None for l in result.lhs]
        return result

    def __mul__(self, other: Any) -> Any:

        if isinstance(other, (int, float, complex)):
            return self.__rmul__(other)
        elif isinstance(other, np.ndarray):
            return self.mtimes_vec(other)
        elif isinstance(other, HMatrix):
            return self._mtimes_hmat(other)
        else:
            raise TypeError('[error] Unsupported type for H-matrix multiplication')

    def __neg__(self) -> 'HMatrix':

        # MATLAB: uminus
        result = self._copy()
        result.val = [-v if v is not None else None for v in result.val]
        result.lhs = [-l if l is not None else None for l in result.lhs]
        return result

    def __add__(self, other: 'HMatrix') -> 'HMatrix':

        # MATLAB: plus
        if not isinstance(other, HMatrix):
            raise TypeError('[error] Unsupported type for H-matrix addition')
        return self._plus_hmat(other)

    def __sub__(self, other: 'HMatrix') -> 'HMatrix':

        # MATLAB: minus
        return self.__add__(-other)

    def _plus_hmat(self, other: 'HMatrix') -> 'HMatrix':

        # MATLAB: plus2 - Add two H-matrices with same structure
        #
        # Bug 6 fix (v1.5.2): when one operand has cupy blocks and the
        # other has numpy blocks (commonly: region (0,0) goes through
        # CompGreenRet's GPU-native path while region (1,0) doesn't), the
        # naive ``a + b`` raises ``TypeError: Unsupported type
        # <numpy.ndarray>``.  We pull both operands to the same backend
        # before adding.  Preference is GPU when either side is cupy so
        # the result fits the downstream HMatrix.full() GPU path.
        result = self._copy()

        def _same_backend(a: Any, b: Any) -> Tuple[Any, Any]:
            a_is_cupy = hasattr(a, 'get') and not isinstance(a, np.ndarray)
            b_is_cupy = hasattr(b, 'get') and not isinstance(b, np.ndarray)
            if a_is_cupy == b_is_cupy:
                return a, b
            try:
                import cupy as _cp_local
            except Exception:
                # Cupy missing yet one of the inputs has .get(); fall back
                # to host arithmetic.
                a_h = a.get() if a_is_cupy else a
                b_h = b.get() if b_is_cupy else b
                return a_h, b_h
            if a_is_cupy:
                return a, _cp_local.asarray(b)
            return _cp_local.asarray(a), b

        # Add dense blocks
        for i in range(len(result.row1)):
            if result.val[i] is not None and other.val[i] is not None:
                a, b = _same_backend(result.val[i], other.val[i])
                result.val[i] = a + b
            elif other.val[i] is not None:
                result.val[i] = other.val[i].copy()

        # Add low-rank blocks: combine and recompress
        for i in range(len(result.row2)):
            lhs1 = result.lhs[i]
            rhs1 = result.rhs[i]
            lhs2 = other.lhs[i]
            rhs2 = other.rhs[i]

            if lhs1 is None and lhs2 is None:
                continue
            elif lhs1 is None:
                result.lhs[i] = lhs2.copy()
                result.rhs[i] = rhs2.copy()
            elif lhs2 is None:
                pass  # keep result as is
            else:
                # Match backends before stacking columns.
                lhs1, lhs2 = _same_backend(lhs1, lhs2)
                rhs1, rhs2 = _same_backend(rhs1, rhs2)
                lhs1_is_cupy = hasattr(lhs1, 'get') and not isinstance(lhs1, np.ndarray)
                xp_lr = np
                if lhs1_is_cupy:
                    import cupy as _cp_local
                    xp_lr = _cp_local

                # Combine: [lhs1, lhs2] and [rhs1, rhs2]
                m = lhs1.shape[0]
                n = rhs1.shape[0]
                k1 = lhs1.shape[1]
                k2 = lhs2.shape[1]
                new_lhs = xp_lr.empty((m, k1 + k2), dtype = lhs1.dtype)
                new_lhs[:, :k1] = lhs1
                new_lhs[:, k1:] = lhs2
                new_rhs = xp_lr.empty((n, k1 + k2), dtype = rhs1.dtype)
                new_rhs[:, :k1] = rhs1
                new_rhs[:, k1:] = rhs2
                result.lhs[i] = new_lhs
                result.rhs[i] = new_rhs

        # Recompress
        result.truncate()
        return result

    def truncate(self, htol: Optional[float] = None) -> 'HMatrix':

        # MATLAB: @hmatrix/truncate.m
        # Truncate low-rank blocks via SVD
        if htol is None:
            htol = self.htol

        for i in range(len(self.lhs)):
            if self.lhs[i] is None or self.rhs[i] is None:
                continue
            self.lhs[i], self.rhs[i] = self._truncate_block(
                self.lhs[i], self.rhs[i], htol)

        self.htol = htol
        return self

    def _truncate_block(self,
            lhs: np.ndarray,
            rhs: np.ndarray,
            htol: float) -> Tuple[np.ndarray, np.ndarray]:

        # MATLAB: truncate/fun
        # Bug 6 fix (v1.5.2): when blocks live on GPU (cupy) the np.linalg
        # routines below would force a host sync per block and re-allocate
        # the QR / SVD scratch on host.  Detect cupy operands and dispatch
        # to ``cupy.linalg.{qr,svd}`` so the recompression stays on-device.
        lhs_is_cupy = hasattr(lhs, 'get') and not isinstance(lhs, np.ndarray)
        if lhs_is_cupy:
            import cupy as _cp_local
            xp = _cp_local
        else:
            xp = np

        if xp.linalg.norm(lhs.ravel()) < np.finfo(float).eps:
            return lhs, rhs
        if xp.linalg.norm(rhs.ravel()) < np.finfo(float).eps:
            return lhs, rhs

        q1, r1 = xp.linalg.qr(lhs, mode = 'reduced')
        q2, r2 = xp.linalg.qr(rhs, mode = 'reduced')

        # SVD of r1 @ r2.T
        u_svd, s_svd, vt_svd = xp.linalg.svd(r1 @ r2.T, full_matrices = False)

        # Find largest singular values: keep k such that cumsum(s) < (1-htol)*sum(s).
        # Truncation thresholding stays on host (singular values are O(rank)).
        s_host = s_svd.get() if lhs_is_cupy else s_svd
        total = float(np.sum(s_host))
        if total < np.finfo(float).eps:
            return lhs[:, :1] * 0, rhs[:, :1] * 0

        cum = np.cumsum(s_host)
        threshold = (1.0 - htol) * total
        k_idx = np.where(cum < threshold)[0]

        if len(k_idx) == 0:
            # Keep at least rank 1
            k = 1
        else:
            k = len(k_idx)

        # Truncated decomposition
        new_lhs = q1 @ (u_svd[:, :k] * s_svd[:k][xp.newaxis, :])
        new_rhs = q2 @ vt_svd[:k, :].T  # q2 @ conj(v[:, :k])

        return new_lhs, new_rhs

    def compression(self) -> float:

        # MATLAB: @hmatrix/compression.m
        # Ratio of H-matrix elements to full matrix elements
        n_elements = 0
        for v in self.val:
            if v is not None:
                n_elements += v.size
        for i in range(len(self.lhs)):
            if self.lhs[i] is not None:
                n_elements += self.lhs[i].size
            if self.rhs[i] is not None:
                n_elements += self.rhs[i].size

        total = self.tree.n * self.tree.n
        if total == 0:
            return 0.0
        return n_elements / total

    def diag(self) -> np.ndarray:

        # MATLAB: @hmatrix/diag.m
        tree = self.tree
        n = tree.n
        diag_dtype = np.float64
        for v in self.val:
            if v is not None and np.iscomplexobj(v):
                diag_dtype = np.complex128
                break
        d = np.zeros(n, dtype = diag_dtype)

        # Find diagonal dense blocks
        for i in range(len(self.row1)):
            if self.row1[i] == self.col1[i] and self.val[i] is not None:
                r_start = tree.cind[self.row1[i], 0]
                r_end = tree.cind[self.row1[i], 1] + 1
                d[r_start:r_end] = np.diag(self.val[i])

        # Convert to particle indices
        return d[tree.ind[:, 1]]

    def eye_hmat(self) -> 'HMatrix':

        # MATLAB: @hmatrix/eye.m
        result = self._copy()

        # Clear all blocks
        result.val = [None] * len(result.row1)
        result.lhs = [None] * len(result.row2)
        result.rhs = [None] * len(result.row2)

        # Pad with zeros
        result.pad()

        # Set diagonal dense blocks to identity
        for i in range(len(result.row1)):
            if result.row1[i] == result.col1[i]:
                result.val[i] = np.eye(result.val[i].shape[0], result.val[i].shape[1])

        return result

    def pad(self) -> 'HMatrix':

        # MATLAB: @hmatrix/pad.m
        tree = self.tree
        siz = tree.cind[:, 1] - tree.cind[:, 0] + 1

        # Detect dtype from existing blocks
        pad_dtype = np.float64
        for v in self.val:
            if v is not None and np.iscomplexobj(v):
                pad_dtype = np.complex128
                break
        if pad_dtype == np.float64:
            for l in self.lhs:
                if l is not None and np.iscomplexobj(l):
                    pad_dtype = np.complex128
                    break

        for i in range(len(self.val)):
            if self.val[i] is None:
                m = siz[self.row1[i]]
                n_col = siz[self.col1[i]]
                self.val[i] = np.zeros((m, n_col), dtype = pad_dtype)

        for i in range(len(self.lhs)):
            if self.lhs[i] is None:
                m = siz[self.row2[i]]
                self.lhs[i] = np.zeros((m, 1), dtype = pad_dtype)
            if self.rhs[i] is None:
                n_col = siz[self.col2[i]]
                self.rhs[i] = np.zeros((n_col, 1), dtype = pad_dtype)

        return self

    def fillval(self, fun: Callable) -> 'HMatrix':

        # MATLAB: @hmatrix/fillval.m
        # Fill dense blocks with function values
        tree = self.tree
        ind_c2p = tree.ind[:, 0]

        def fun2(row_c: np.ndarray, col_c: np.ndarray) -> np.ndarray:
            return fun(ind_c2p[row_c], ind_c2p[col_c])

        for i in range(len(self.row1)):
            indr = tree.cind[self.row1[i]]
            indc = tree.cind[self.col1[i]]
            rows = np.arange(indr[0], indr[1] + 1, dtype = np.int64)
            cols = np.arange(indc[0], indc[1] + 1, dtype = np.int64)
            row_grid, col_grid = np.meshgrid(rows, cols, indexing = 'ij')
            self.val[i] = fun2(row_grid.ravel(), col_grid.ravel()).reshape(row_grid.shape)

        return self

    def lu(self, method: str = 'auto') -> 'HMatrix':
        """LU decomposition for H-matrix.

        Parameters
        ----------
        method : str
            'dense' - convert to dense and factorize (O(N^3), always correct)
            'hierarchical' - recursive block LU preserving H-matrix structure
            'auto' - use hierarchical if tree has children, else dense
        """
        if method == 'auto':
            root_sons = self.tree.son[0]
            if root_sons[0] >= 0 and root_sons[1] >= 0 and self.tree.n > 64:
                method = 'hierarchical'
            else:
                method = 'dense'

        if method == 'hierarchical':
            return self._hierarchical_lu()

        # Dense fallback using lu_factor (more efficient than full P,L,U)
        from scipy.linalg import lu_factor
        mat = self.full()
        self._lu_packed, self._lu_piv = lu_factor(mat, check_finite=False, overwrite_a=True)
        self._lu_done = True
        self._lu_method = 'dense'
        return self

    def _hierarchical_lu(self) -> 'HMatrix':
        """Block Schur complement factorization.

        Splits A into [[A11, A12], [A21, A22]] based on tree root children.
        Stores: lu(A11), A12, A21, lu(S22) where S22 = A22 - A21 @ A11^-1 @ A12.
        """
        tree = self.tree
        root_sons = tree.son[0]
        s1, s2 = root_sons[0], root_sons[1]

        if s1 < 0 or s2 < 0:
            return self.lu(method='dense')

        mat = self.full()
        n1 = tree.cind[s1, 1] - tree.cind[s1, 0] + 1

        A11 = mat[:n1, :n1]
        A12 = mat[:n1, n1:]
        A21 = mat[n1:, :n1]
        A22 = mat[n1:, n1:]

        from scipy.linalg import lu_factor, lu_solve

        lu11, piv11 = lu_factor(A11, check_finite=False)
        C12 = lu_solve((lu11, piv11), A12, check_finite=False)  # A11^-1 @ A12
        S22 = A22 - A21 @ C12               # Schur complement
        lu22, piv22 = lu_factor(S22, check_finite=False, overwrite_a=True)

        self._block_lu = {
            'lu11': lu11, 'piv11': piv11,
            'A12': A12.copy(), 'A21': A21.copy(),
            'lu22': lu22, 'piv22': piv22,
            'n1': n1,
        }
        self._lu_done = True
        self._lu_method = 'hierarchical'
        return self

    def solve(self, b: np.ndarray) -> np.ndarray:
        """Solve A*x = b using LU factored H-matrix."""
        if not hasattr(self, '_lu_done') or not self._lu_done:
            mat = self.full()
            return np.linalg.solve(mat, b)

        from scipy.linalg import lu_solve

        if self._lu_method == 'dense':
            return lu_solve((self._lu_packed, self._lu_piv), b, check_finite=False)

        # Block Schur complement solve:
        # x2 = S22^-1 @ (b2 - A21 @ A11^-1 @ b1)
        # x1 = A11^-1 @ (b1 - A12 @ x2)
        blk = self._block_lu
        n1 = blk['n1']
        b1 = b[:n1]
        b2 = b[n1:]

        temp = lu_solve((blk['lu11'], blk['piv11']), b1, check_finite=False)
        x2 = lu_solve((blk['lu22'], blk['piv22']), b2 - blk['A21'] @ temp, check_finite=False, overwrite_b=True)
        x1 = lu_solve((blk['lu11'], blk['piv11']), b1 - blk['A12'] @ x2, check_finite=False, overwrite_b=True)

        if b.ndim == 1:
            return np.concatenate([x1, x2])
        return np.vstack([x1, x2])

    def _mtimes_hmat(self, other: 'HMatrix') -> 'HMatrix':
        """H-matrix * H-matrix multiplication."""
        mat1 = self.full()
        mat2 = other.full()
        result_mat = mat1 @ mat2

        result = HMatrix(tree=self.tree, htol=self.htol, kmax=self.kmax)
        result.row1 = self.row1.copy()
        result.col1 = self.col1.copy()
        result.row2 = self.row2.copy()
        result.col2 = self.col2.copy()
        result.val = [None] * len(result.row1)
        result.lhs = [None] * len(result.row2)
        result.rhs = [None] * len(result.row2)

        def mat_fun(row, col):
            return result_mat[row, col]
        result.aca(mat_fun)
        return result

    def _copy(self) -> 'HMatrix':

        result = HMatrix.__new__(HMatrix)
        result.tree = self.tree
        result.htol = self.htol
        result.kmax = self.kmax
        result.row1 = self.row1.copy()
        result.col1 = self.col1.copy()
        result.row2 = self.row2.copy()
        result.col2 = self.col2.copy()
        result.val = [v.copy() if v is not None else None for v in self.val]
        result.lhs = [l.copy() if l is not None else None for l in self.lhs]
        result.rhs = [r.copy() if r is not None else None for r in self.rhs]
        return result

    def inv(self) -> 'HMatrix':

        # MATLAB: @hmatrix/inv.m
        # Inverse of hierarchical matrix.
        # MATLAB version uses MEX (hmatinv); Python uses dense fallback via LU.
        mat = self.full()
        inv_mat = np.linalg.inv(mat)

        result = self._copy()

        def mat_fun(row: np.ndarray, col: np.ndarray) -> np.ndarray:
            return inv_mat[row, col]

        result.aca(mat_fun)
        return result

    def lsolve(self,
            lu_hmat: 'HMatrix',
            key: str = 'N') -> 'HMatrix':

        # MATLAB: @hmatrix/lsolve.m
        # Solve (L*U) * X = B  =>  X = (L*U)^{-1} * B
        # self is B, lu_hmat is (L*U)
        # key: 'L' for lower, 'U' for upper, 'N' for both (default)
        from scipy.linalg import solve_triangular

        B = self.full()
        A = lu_hmat.full()

        if key == 'L':
            X = solve_triangular(np.tril(A), B, lower = True)
        elif key == 'U':
            X = solve_triangular(np.triu(A), B, lower = False)
        else:
            # key == 'N': solve full system A * X = B
            X = np.linalg.solve(A, B)

        result = self._copy()

        def mat_fun(row: np.ndarray, col: np.ndarray) -> np.ndarray:
            return X[row, col]

        result.aca(mat_fun)
        return result

    def rsolve(self,
            lu_hmat: 'HMatrix',
            key: str = 'N') -> 'HMatrix':

        # MATLAB: @hmatrix/rsolve.m
        # Solve X * (L*U) = B  =>  X = B * (L*U)^{-1}
        # self is B, lu_hmat is (L*U)
        # key: 'L' for lower, 'U' for upper, 'N' for both (default)
        from scipy.linalg import solve_triangular

        B = self.full()
        A = lu_hmat.full()

        if key == 'L':
            # X * L = B  =>  L^T * X^T = B^T
            X = solve_triangular(np.tril(A), B.T, lower = True).T
        elif key == 'U':
            # X * U = B  =>  U^T * X^T = B^T
            X = solve_triangular(np.triu(A), B.T, lower = False).T
        else:
            # key == 'N': X * A = B  =>  A^T * X^T = B^T
            X = np.linalg.solve(A.T, B.T).T

        result = self._copy()

        def mat_fun(row: np.ndarray, col: np.ndarray) -> np.ndarray:
            return X[row, col]

        result.aca(mat_fun)
        return result

    def stat(self) -> Dict:

        # MATLAB: @hmatrix/stat.m
        # Compression statistics for the H-matrix
        tree = self.tree
        n = tree.n

        n_dense = len(self.row1)
        n_lowrank = len(self.row2)

        # Count elements in dense blocks
        n_dense_elem = 0
        for v in self.val:
            if v is not None:
                n_dense_elem += v.size

        # Count elements in low-rank blocks and collect ranks
        n_lr_elem = 0
        ranks = []
        for i in range(n_lowrank):
            if self.lhs[i] is not None:
                n_lr_elem += self.lhs[i].size
                ranks.append(self.lhs[i].shape[1])
            if self.rhs[i] is not None:
                n_lr_elem += self.rhs[i].size

        total_full = n * n
        total_hmat = n_dense_elem + n_lr_elem
        ratio = total_hmat / total_full if total_full > 0 else 0.0

        # Memory in bytes (float64 = 8 bytes)
        mem_bytes = total_hmat * 8
        mem_mb = mem_bytes / (1024.0 * 1024.0)

        stats = {
            'n': n,
            'n_dense_blocks': n_dense,
            'n_lowrank_blocks': n_lowrank,
            'n_dense_elements': n_dense_elem,
            'n_lowrank_elements': n_lr_elem,
            'n_total_elements': total_hmat,
            'n_full_elements': total_full,
            'compression_ratio': ratio,
            'memory_mb': mem_mb,
            'ranks': ranks,
            'mean_rank': float(np.mean(ranks)) if len(ranks) > 0 else 0.0,
            'max_rank': int(np.max(ranks)) if len(ranks) > 0 else 0,
        }

        print('[info] H-matrix statistics:')
        print('  Matrix size:       {} x {}'.format(n, n))
        print('  Dense blocks:      {}'.format(n_dense))
        print('  Low-rank blocks:   {}'.format(n_lowrank))
        print('  Dense elements:    {}'.format(n_dense_elem))
        print('  Low-rank elements: {}'.format(n_lr_elem))
        print('  Total elements:    {} / {} (full)'.format(total_hmat, total_full))
        print('  Compression ratio: {:.4f}'.format(ratio))
        print('  Memory:            {:.2f} MB'.format(mem_mb))
        if len(ranks) > 0:
            print('  Rank (mean/max):   {:.1f} / {}'.format(stats['mean_rank'], stats['max_rank']))

        return stats

    def treemex(self) -> Dict:

        # MATLAB: @hmatrix/treemex.m
        # In MATLAB, this prepares tree data for MEX C++ functions (0-based indexing).
        # In Python, no MEX is needed; this returns the tree structure as a dict
        # for compatibility with code that expects the treemex interface.
        tree = self.tree

        result = {
            'sons': tree.son.copy(),
            'ind': tree.cind.copy(),
            'ind1': np.column_stack([self.row1, self.col1]) if len(self.row1) > 0 else np.empty((0, 2), dtype = np.int64),
            'ind2': np.column_stack([self.row2, self.col2]) if len(self.row2) > 0 else np.empty((0, 2), dtype = np.int64),
            'ipart': tree.ipart.copy(),
        }

        return result

    def plotfun(self,
            mat: Optional[np.ndarray] = None,
            fun: Optional[Callable] = None) -> Any:

        # MATLAB: @hmatrix/plotfun.m
        # Plot function applied to hierarchical matrix blocks
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        tree = self.tree
        n = tree.n
        cind = tree.cind

        # Default function: just return the block norm
        if fun is None:
            fun = lambda x, y: np.linalg.norm(x) if x is not None else 0.0

        # Allocate output array
        fmat = np.zeros((n, n))

        # Loop over dense blocks
        for i in range(len(self.row1)):
            r_start = cind[self.row1[i], 0]
            r_end = cind[self.row1[i], 1] + 1
            c_start = cind[self.col1[i], 0]
            c_end = cind[self.col1[i], 1] + 1

            sub = None
            if mat is not None:
                ind_r = tree.ind[r_start:r_end, 0]
                ind_c = tree.ind[c_start:c_end, 0]
                sub = mat[np.ix_(ind_r, ind_c)]

            val = fun(self.val[i], sub)
            fmat[r_start:r_end, c_start:c_end] = val

        # Loop over low-rank blocks
        for i in range(len(self.row2)):
            r_start = cind[self.row2[i], 0]
            r_end = cind[self.row2[i], 1] + 1
            c_start = cind[self.col2[i], 0]
            c_end = cind[self.col2[i], 1] + 1

            sub = None
            if mat is not None:
                ind_r = tree.ind[r_start:r_end, 0]
                ind_c = tree.ind[c_start:c_end, 0]
                sub = mat[np.ix_(ind_r, ind_c)]

            # Reconstruct low-rank block
            if self.lhs[i] is not None and self.rhs[i] is not None:
                block = self.lhs[i] @ self.rhs[i].T
            else:
                block = np.zeros((r_end - r_start, c_end - c_start))

            val = fun(block, sub)
            fmat[r_start:r_end, c_start:c_end] = val

        # Plot
        fig, ax = plt.subplots(1, 1)
        im = ax.imshow(fmat, origin = 'upper', aspect = 'equal')
        plt.colorbar(im, ax = ax)

        # Draw low-rank block boundaries
        for i in range(len(self.row2)):
            r_start = cind[self.row2[i], 0] - 0.5
            r_end = cind[self.row2[i], 1] + 0.5
            c_start = cind[self.col2[i], 0] - 0.5
            c_end = cind[self.col2[i], 1] + 0.5

            rect = patches.Rectangle(
                (c_start, r_start),
                c_end - c_start,
                r_end - r_start,
                linewidth = 0.5,
                edgecolor = 'k',
                facecolor = 'none')
            ax.add_patch(rect)

        ax.set_title('H-matrix block structure')
        return fig, ax

    def plotrank(self) -> Any:

        # MATLAB: @hmatrix/plotrank.m
        # Plot rank distribution of low-rank blocks
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        tree = self.tree
        n = tree.n
        cind = tree.cind

        # Allocate rank matrix
        mat = np.zeros((n, n), dtype = np.int32)

        # Fill low-rank block regions with their rank
        for i in range(len(self.row2)):
            r_start = cind[self.row2[i], 0]
            r_end = cind[self.row2[i], 1] + 1
            c_start = cind[self.col2[i], 0]
            c_end = cind[self.col2[i], 1] + 1

            if self.lhs[i] is not None:
                rank = self.lhs[i].shape[1]
            else:
                rank = 0
            mat[r_start:r_end, c_start:c_end] = rank

        # Plot
        fig, ax = plt.subplots(1, 1)
        im = ax.imshow(mat, origin = 'upper', aspect = 'equal')
        plt.colorbar(im, ax = ax, label = 'rank')

        # Draw low-rank block boundaries
        for i in range(len(self.row2)):
            r_start = cind[self.row2[i], 0] - 0.5
            r_end = cind[self.row2[i], 1] + 0.5
            c_start = cind[self.col2[i], 0] - 0.5
            c_end = cind[self.col2[i], 1] + 0.5

            rect = patches.Rectangle(
                (c_start, r_start),
                c_end - c_start,
                r_end - r_start,
                linewidth = 0.5,
                edgecolor = 'k',
                facecolor = 'none')
            ax.add_patch(rect)

        ax.set_title('H-matrix rank distribution')
        return fig, ax

    @staticmethod
    def from_func(tree: ClusterTree,
            fun: Callable,
            htol: float = 1e-6,
            kmax: int = 100,
            fadmiss: Optional[Callable] = None) -> 'HMatrix':

        hmat = HMatrix(tree = tree, htol = htol, kmax = kmax, fadmiss = fadmiss)
        hmat.aca(fun)
        return hmat


def make_kaware_fadmiss(k: float, eta0: float = 2.5) -> Callable:
    """Return an admissibility predicate aware of the wavenumber ``k``.

    For static problems (k=0) this reproduces the standard Boerm criterion
    ``eta0 * min(rad1, rad2) < dist``.  For retarded problems the kernel
    e^{i k R}/R oscillates and ACA rank grows like (k * diam)^{d}.  To keep
    the rank bounded we tighten the criterion when ``k * min(rad)`` exceeds
    one wavelength, requiring ``dist`` to also exceed a few wavelengths
    ``2*pi/k``.  This avoids the rank blow-up reported when wavelength is
    comparable to the cluster diameter.
    """
    if k is None or not np.isfinite(k) or abs(k) < 1e-12:
        return lambda rad1, rad2, dist: eta0 * min(rad1, rad2) < dist

    k_abs = abs(complex(k))
    wavelength = 2.0 * np.pi / k_abs
    def fadmiss(rad1: float, rad2: float, dist: float) -> bool:
        rmin = min(rad1, rad2)
        # Standard geometric admissibility
        if eta0 * rmin >= dist:
            return False
        # Wavelength constraint: when cluster spans many wavelengths the
        # block can still be highly oscillatory; require dist > one
        # wavelength so that exp(i k R) is well-resolved by low rank.
        if k_abs * rmin > 1.0 and dist < wavelength:
            return False
        return True
    return fadmiss
