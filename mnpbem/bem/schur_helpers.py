"""
Schur complement reduction for cover-layer BEM (MNPBEM v1.2.0).

When a particle carries an artificial nonlocal cover layer (EpsNonlocal +
coverlayer.shift), the BEM mesh face count doubles -- shell faces +
core faces. For the dense direct solver this inflates the BEM matrix to
(2N, 2N) and the LU factor cost by 8x.

The cover layer enters the BEM equations only as a thin boundary
condition; the shell sub-matrix G_ss tends to be small and well
conditioned. Schur complement elimination of the shell variables
collapses the system back to the (M, M) core block

    [G_ss G_sc] [sig_s]   [b_s]
    [G_cs G_cc] [sig_c] = [b_c]

becomes

    G_eff = G_cc - G_cs @ inv(G_ss) @ G_sc
    b_eff = b_c  - G_cs @ inv(G_ss) @ b_s
    sig_c = inv(G_eff) @ b_eff
    sig_s = inv(G_ss) @ (b_s - G_sc @ sig_c)

The reduced solve produces results that are mathematically identical to
the full block solve (up to floating-point round-off), but the dominant
LU factor now operates on an (M, M) matrix instead of (M+N, M+N).

Distributed variant (v1.8 Agent VRAM-Schur)
------------------------------------------
``schur_eliminate_distributed`` builds the reduced ``G_eff`` matrix
directly as a :class:`DistributedMatrix` (column block-cyclic across N
GPUs), then factors it in place via cuSolverMg.  The full ``(N, N)``
matrix is never materialised on a single device.  The reduce/recover
callables work on host vectors so the downstream BEM solve path is
unchanged.  This is the multi-GPU mirror of ``schur_eliminate`` and
combines with the column-distributed build path in
``BEMStat._init_distributed_schur`` / ``BEMStatIter._init_distributed_schur``.

Memory profile (N total = 2 * n_shell = 2 * n_core, the typical
cover-layer case):
- legacy schur_eliminate    : peak ~ N^2 + (N/2)^2 host complex128
- distributed variant       : per-GPU tile ~ (N/2)^2 / n_gpus + small
                              transient host slices for the M_ss / M_sc /
                              M_cs blocks (each ~ (N/2)^2 = 1/4 of full).
"""

from typing import Any, Callable, Optional, Tuple

import numpy as np
from scipy.linalg import lu_factor, lu_solve


def schur_eliminate(M_full: np.ndarray,
        shell_indices: np.ndarray,
        core_indices: np.ndarray) -> Tuple[np.ndarray, Callable, Callable]:

    s = np.asarray(shell_indices, dtype = int)
    c = np.asarray(core_indices, dtype = int)

    if s.size == 0:
        # No shell -- nothing to eliminate. Return identity-like wrappers
        # so callers can use the same code path.
        def _identity_rhs(b: np.ndarray) -> np.ndarray:
            return b[c]

        def _identity_recover(sig_core: np.ndarray, b_full: np.ndarray) -> np.ndarray:
            return sig_core

        return M_full[np.ix_(c, c)], _identity_rhs, _identity_recover

    M_ss = M_full[np.ix_(s, s)]
    M_sc = M_full[np.ix_(s, c)]
    M_cs = M_full[np.ix_(c, s)]
    M_cc = M_full[np.ix_(c, c)]

    lu_ss, piv_ss = lu_factor(M_ss, check_finite = False)
    M_inv_sc = lu_solve((lu_ss, piv_ss), M_sc, check_finite = False)
    M_eff = M_cc - M_cs @ M_inv_sc

    def reduce_rhs(b_full: np.ndarray) -> np.ndarray:
        b_s = b_full[s]
        b_c = b_full[c]
        if b_s.ndim == 1:
            corr = M_cs @ lu_solve((lu_ss, piv_ss), b_s, check_finite = False)
        else:
            corr = M_cs @ lu_solve((lu_ss, piv_ss),
                    b_s.reshape(b_s.shape[0], -1), check_finite = False).reshape(b_s.shape)
        return b_c - corr

    def recover_full(sig_core: np.ndarray, b_full: np.ndarray) -> np.ndarray:
        b_s = b_full[s]
        rhs_s = b_s - M_sc @ sig_core
        if rhs_s.ndim == 1:
            sig_s = lu_solve((lu_ss, piv_ss), rhs_s, check_finite = False)
        else:
            sig_s = lu_solve((lu_ss, piv_ss),
                    rhs_s.reshape(rhs_s.shape[0], -1), check_finite = False).reshape(rhs_s.shape)

        out_shape = list(b_full.shape)
        out = np.empty(out_shape, dtype = np.result_type(sig_core, sig_s))
        out[s] = sig_s
        out[c] = sig_core
        return out

    return M_eff, reduce_rhs, recover_full


def detect_shell_core_partition(particle: Any) -> Optional[Tuple[np.ndarray, np.ndarray]]:

    from ..materials import EpsNonlocal

    eps_list = getattr(particle, 'eps', None)
    inout = getattr(particle, 'inout_faces', None)

    if eps_list is None or inout is None:
        return None

    # 1-based MATLAB indices in inout. Identify which eps slots are EpsNonlocal.
    nonlocal_eps_idx_1based = set()
    for i, eps in enumerate(eps_list):
        if isinstance(eps, EpsNonlocal):
            nonlocal_eps_idx_1based.add(i + 1)

    if not nonlocal_eps_idx_1based:
        return None

    inout_arr = np.asarray(inout)
    nfaces = inout_arr.shape[0]

    # Convention (matches the EpsNonlocal cover-layer geometry built by
    # ``coverlayer.shift`` + ``make_nonlocal_pair``):
    #
    #   shell particle row in inout : [nonlocal_eps_idx, embed_eps_idx]
    #                                 -> EpsNonlocal sits on the *inside*
    #                                    of the shell face, embed on the
    #                                    outside.
    #   core particle row in inout  : [metal_eps_idx, nonlocal_eps_idx]
    #                                 -> EpsNonlocal sits on the *outside*
    #                                    of the core face, metal on the
    #                                    inside.
    #
    # Schur reduction targets the artificial cover-layer (shell) faces --
    # i.e. those whose *inside* material is EpsNonlocal.  Reducing the
    # core faces would also work mathematically but would defeat the
    # memory savings (core block is the larger one).
    in_col = inout_arr[:, 0].astype(int)
    shell_mask = np.array([int(idx) in nonlocal_eps_idx_1based for idx in in_col])
    shell_indices = np.where(shell_mask)[0]
    core_indices = np.where(~shell_mask)[0]

    if shell_indices.size == 0 or core_indices.size == 0:
        # Either no shell faces (no EpsNonlocal on the inside of any face)
        # or no remaining core faces -- in both cases the reduction is a
        # no-op or degenerate, so return None and let the caller fall back
        # to the full BEM matrix.
        return None

    return shell_indices, core_indices


def schur_memory_estimate(nfaces_total: int, nfaces_shell: int) -> dict:

    nfaces_core = nfaces_total - nfaces_shell

    # Each complex matrix entry is 16 bytes.
    bytes_per_entry = 16

    full_bytes = nfaces_total * nfaces_total * bytes_per_entry
    reduced_bytes = nfaces_core * nfaces_core * bytes_per_entry
    schur_overhead = (nfaces_shell * nfaces_shell + 2 * nfaces_shell * nfaces_core) * bytes_per_entry

    return {
        'nfaces_total': nfaces_total,
        'nfaces_shell': nfaces_shell,
        'nfaces_core': nfaces_core,
        'full_matrix_bytes': full_bytes,
        'reduced_matrix_bytes': reduced_bytes,
        'schur_temp_bytes': schur_overhead,
        'reduction_ratio': reduced_bytes / full_bytes if full_bytes else 0.0,
    }


# ---------------------------------------------------------------------------
# Distributed Schur (v1.8 VRAM-Schur)
# ---------------------------------------------------------------------------

def schur_eliminate_distributed(
        M_eval: Callable[[int, int], np.ndarray],
        N_full: int,
        shell_indices: np.ndarray,
        core_indices: np.ndarray,
        n_gpus: int,
        device_ids: Optional[list] = None,
        block_size: int = 256) -> Tuple[Any, Callable, Callable, Any]:

    # Distributed Schur reduction over a column-distributed BEM matrix.
    #
    # ``M_eval(c0, c1)`` returns the column slice ``M[:, c0:c1]`` of the
    # *full* BEM matrix as a host ndarray of shape (N_full, c1-c0).
    # The block-cyclic distributed tiles are built directly through
    # ``DistributedMatrix.from_func``; the full (N_full, N_full) host
    # buffer is never allocated.
    #
    # Algorithm
    # ---------
    # We materialise on host four (small) sub-matrices that the Schur
    # reduction needs:
    #
    #     M_ss   = M[shell, shell]   (n_shell x n_shell)
    #     M_sc   = M[shell, core ]   (n_shell x n_core)
    #     M_cs   = M[core , shell]   (n_core  x n_shell)
    #     D_inv_C = lu_solve(M_ss, M_sc)   (n_shell x n_core)
    #
    # The reduced matrix ``M_eff = M[core, core] - M_cs @ D_inv_C`` is
    # then assembled directly distributed across the N GPUs via
    # ``DistributedMatrix.from_func``: each per-GPU column tile gets
    # populated by computing the corresponding columns of M[core, core]
    # (via ``M_eval`` restricted to core rows) and subtracting the
    # corresponding columns of ``M_cs @ D_inv_C``.
    #
    # Because shell+core face counts are typically equal (cover-layer
    # geometry), each of the four blocks has shape ~(N/2, N/2) — a
    # quarter of the full matrix.  M_ss / M_sc / M_cs live on host but
    # together occupy only 3 * (N/2)^2 = 3/4 of the full matrix size; on
    # an A6000-class node the host RAM easily absorbs that while the
    # distributed M_eff tiles split the dominant (N/2, N/2) reduced LU
    # across the N GPUs.
    #
    # Returns
    # -------
    # M_eff_dm : DistributedMatrix
    #     Reduced matrix, column block-cyclic across N GPUs.  Caller
    #     factors it in place via ``M_eff_dm.lu_factor()``.
    # reduce_rhs(b_full) -> b_eff
    # recover_full(sig_core, b_full) -> sig_full
    #     Both callables operate on host ndarrays; the shell-block linear
    #     system is solved on host using the cached LU of M_ss.
    # keepalive : dict
    #     Holds references to host arrays (D_inv_C, M_cs, lu_ss, piv_ss)
    #     that the closures need.  Callers should attach this to the
    #     long-lived solver object to keep the GC from freeing the
    #     callable's captured state mid-solve.

    from ..utils.distributed_matrix import DistributedMatrix

    s = np.asarray(shell_indices, dtype = int)
    c = np.asarray(core_indices, dtype = int)
    assert s.ndim == 1 and c.ndim == 1, \
            '[error] shell/core indices must be 1-D'
    assert s.size + c.size == N_full, \
            '[error] shell+core indices must partition [0, N_full)'

    n_shell = int(s.size)
    n_core = int(c.size)

    # --- Build M_ss / M_sc / M_cs on host ---------------------------------
    # We slice column-by-column from M_eval to keep the peak host buffer at
    # (N_full, 1) instead of (N_full, N_full).  The shell+core slices are
    # then assembled into the four sub-blocks.  Total host transient peak
    # is ~ N_full * complex128 for the single-col scratch plus the three
    # sub-blocks (which we keep for the recover callables anyway).
    #
    # For shell-related slices we read M[:, s_j] one at a time and split
    # into shell-row / core-row portions.  For M_cc we never materialise
    # explicitly — we read M[:, c_j] tiles inside the eval_func.
    M_ss = np.empty((n_shell, n_shell), dtype = np.complex128)
    M_sc = np.empty((n_shell, n_core), dtype = np.complex128)
    M_cs = np.empty((n_core, n_shell), dtype = np.complex128)

    # If shell indices are contiguous and start at 0 (the standard
    # cover-layer layout from ``coverlayer.shift``), do a single slab
    # read of M[:, 0:n_shell] to avoid n_shell repeated single-column
    # reads.  Detection: indices form a contiguous run.
    def _is_contiguous(idx: np.ndarray) -> bool:
        if idx.size == 0:
            return True
        return bool(np.all(np.diff(idx) == 1))

    shell_contig = _is_contiguous(s)
    core_contig = _is_contiguous(c)

    if shell_contig and s.size > 0:
        s_start = int(s[0])
        s_stop = int(s[-1]) + 1
        # Read all shell columns in one shot.
        col_block = M_eval(s_start, s_stop)  # (N_full, n_shell)
        if col_block.dtype != np.complex128:
            col_block = col_block.astype(np.complex128, copy = False)
        # M_ss = M[shell, shell_cols]
        M_ss[:] = col_block[s, :]
        # M_cs = M[core, shell_cols]
        M_cs[:] = col_block[c, :]
        del col_block
    else:
        # Fall back to per-column reads.
        for k in range(n_shell):
            col = M_eval(int(s[k]), int(s[k]) + 1)  # (N_full, 1)
            if col.dtype != np.complex128:
                col = col.astype(np.complex128, copy = False)
            col1 = col[:, 0]
            M_ss[:, k] = col1[s]
            M_cs[:, k] = col1[c]

    if core_contig and c.size > 0:
        c_start = int(c[0])
        c_stop = int(c[-1]) + 1
        col_block = M_eval(c_start, c_stop)  # (N_full, n_core)
        if col_block.dtype != np.complex128:
            col_block = col_block.astype(np.complex128, copy = False)
        # M_sc = M[shell, core_cols]
        M_sc[:] = col_block[s, :]
        del col_block
    else:
        for k in range(n_core):
            col = M_eval(int(c[k]), int(c[k]) + 1)
            if col.dtype != np.complex128:
                col = col.astype(np.complex128, copy = False)
            M_sc[:, k] = col[s, 0]

    # --- LU factor M_ss + apply to M_sc -----------------------------------
    lu_ss, piv_ss = lu_factor(M_ss, check_finite = False)
    # D_inv_C = inv(M_ss) @ M_sc, shape (n_shell, n_core).  This is the
    # big-ish host buffer (n_shell x n_core ~= N^2 / 4 complex128) that we
    # need to broadcast to each GPU during M_eff assembly.  Done once;
    # reused for all column tiles plus the rhs reduction at solve time.
    D_inv_C = lu_solve((lu_ss, piv_ss), M_sc, check_finite = False)

    # --- Build M_eff distributed ------------------------------------------
    # For each output column tile [c0, c1) we compute:
    #     M_eff[:, c0:c1] = M_cc[:, c0:c1] - M_cs @ D_inv_C[:, c0:c1]
    # The first term is a row-slice of M[core, c[c0:c1]]; the second is
    # a single GEMM of (n_core, n_shell) @ (n_shell, tile_width).  Both
    # are done on the owning GPU to keep PCIe traffic to the small slices
    # of D_inv_C and M_cs.

    try:
        import cupy as _cp_local  # type: ignore
        _cp_ok = True
    except Exception:
        _cp_local = None
        _cp_ok = False

    # Pre-upload M_cs / D_inv_C to each GPU (once per GPU) so subsequent
    # tile builds don't re-cross PCIe.  Stored in dicts keyed by gpu_idx.
    _cs_cache: dict = {}
    _dic_cache: dict = {}

    if device_ids is None:
        device_ids = list(range(n_gpus))

    def _ensure_gpu_cache(gpu_idx: int) -> Tuple[Any, Any]:
        if gpu_idx in _cs_cache:
            return _cs_cache[gpu_idx], _dic_cache[gpu_idx]
        dev = device_ids[gpu_idx]
        _cp_local.cuda.runtime.setDevice(dev)
        with _cp_local.cuda.Device(dev):
            cs_dev = _cp_local.asarray(M_cs, dtype = np.complex128)
            dic_dev = _cp_local.asarray(D_inv_C, dtype = np.complex128)
        _cs_cache[gpu_idx] = cs_dev
        _dic_cache[gpu_idx] = dic_dev
        return cs_dev, dic_dev

    def _eval_M_eff_tile(gpu_idx: int, c0: int, c1: int) -> Any:
        # c0/c1 are output-column indices into the *core* space (0 .. n_core).
        ncol = c1 - c0
        if ncol == 0:
            return np.zeros((n_core, 0), dtype = np.complex128)

        # Resolve to the original-matrix column indices.
        core_cols = c[c0:c1]

        # --- M[core, core_cols] slice on host ---------------------------
        if core_contig:
            full_c0 = int(c[c0])
            full_c1 = int(c[c1 - 1]) + 1
            col_block = M_eval(full_c0, full_c1)  # (N_full, ncol)
        else:
            # Per-column read; falls back to a host loop that materialises
            # the slice column by column.  Avoid the inner loop for the
            # standard contiguous case (above).
            col_block = np.empty((N_full, ncol), dtype = np.complex128)
            for k, gj in enumerate(core_cols):
                col_block[:, k] = M_eval(int(gj), int(gj) + 1)[:, 0]
        if col_block.dtype != np.complex128:
            col_block = col_block.astype(np.complex128, copy = False)
        M_cc_tile_host = col_block[c, :]
        del col_block

        # --- Compute the correction on the owning GPU -------------------
        if _cp_ok:
            cs_dev, dic_dev = _ensure_gpu_cache(gpu_idx)
            with _cp_local.cuda.Device(device_ids[gpu_idx]):
                dic_slice = dic_dev[:, c0:c1]
                # correction = M_cs @ D_inv_C[:, c0:c1]
                correction = cs_dev @ dic_slice
                mcc_dev = _cp_local.asarray(M_cc_tile_host)
                out = mcc_dev - correction
                # Drop the upload now that the difference is on-device.
                del mcc_dev, correction
            return out
        # CPU fallback (cupy unavailable shouldn't happen because
        # DistributedMatrix requires cupy, but be defensive).
        correction = M_cs @ D_inv_C[:, c0:c1]
        return M_cc_tile_host - correction

    M_eff_dm = DistributedMatrix.from_func(
            shape = (n_core, n_core),
            dtype = np.complex128,
            n_gpus = n_gpus,
            device_ids = device_ids,
            block_size = block_size,
            eval_func = _eval_M_eff_tile)

    # Free the cached uploads now that the tiles are built.  The
    # closures we return only need the host copies (M_cs, D_inv_C) which
    # are kept alive via the ``keepalive`` dict.
    if _cp_ok:
        for gpu_idx, cs_dev in _cs_cache.items():
            try:
                dev = device_ids[gpu_idx]
                _cp_local.cuda.runtime.setDevice(dev)
                with _cp_local.cuda.Device(dev):
                    del cs_dev
                    if gpu_idx in _dic_cache:
                        _dic_cache[gpu_idx] = None
                    _cp_local.get_default_memory_pool().free_all_blocks()
            except Exception:
                pass
        _cs_cache.clear()
        _dic_cache.clear()

    # --- Reduce/recover callables ----------------------------------------
    # These operate on host vectors (the full sigma / rhs) so the BEM
    # solver wrapper does not have to change.
    def reduce_rhs(b_full: np.ndarray) -> np.ndarray:
        b_full = np.asarray(b_full)
        b_s = b_full[s]
        b_c = b_full[c]
        if b_s.ndim == 1:
            corr = M_cs @ lu_solve((lu_ss, piv_ss),
                    b_s, check_finite = False)
        else:
            corr = M_cs @ lu_solve((lu_ss, piv_ss),
                    b_s.reshape(b_s.shape[0], -1),
                    check_finite = False).reshape(b_s.shape)
        return b_c - corr

    def recover_full(sig_core: np.ndarray, b_full: np.ndarray) -> np.ndarray:
        sig_core = np.asarray(sig_core)
        b_full = np.asarray(b_full)
        b_s = b_full[s]
        rhs_s = b_s - M_sc @ sig_core
        if rhs_s.ndim == 1:
            sig_s = lu_solve((lu_ss, piv_ss), rhs_s, check_finite = False)
        else:
            sig_s = lu_solve((lu_ss, piv_ss),
                    rhs_s.reshape(rhs_s.shape[0], -1),
                    check_finite = False).reshape(rhs_s.shape)

        out_shape = list(b_full.shape)
        out = np.empty(out_shape,
                dtype = np.result_type(sig_core, sig_s))
        out[s] = sig_s
        out[c] = sig_core
        return out

    # Stash the small host buffers + LU on a keepalive object so callers
    # can hold a single reference instead of tracking each closure capture
    # manually.  ``M_eff_dm`` is *not* in the keepalive — caller owns
    # that and is expected to call ``lu_factor()`` then ``.free()`` /
    # close() through the resulting MultiGPULU handle.
    keepalive = {
        'M_ss_lu': (lu_ss, piv_ss),
        'M_sc': M_sc,
        'M_cs': M_cs,
        'D_inv_C': D_inv_C,
        'shell_indices': s,
        'core_indices': c,
    }
    return M_eff_dm, reduce_rhs, recover_full, keepalive
