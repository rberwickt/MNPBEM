"""Block-cyclic column-distributed dense matrix across N GPUs.

Provides the storage primitive for the *real* multi-GPU BEM build path
(B-3 in the GPU performance roadmap). A single dense matrix that is too
large to fit on one GPU (e.g. the 47 GB BEM Sigma for the 15072-face
Au@Ag dimer at complex128) is partitioned along its column axis using
the same 1-D block-cyclic layout that ``cusolverMg`` expects. Each GPU
holds a contiguous tile of ``(M, local_cols_alloc)`` complex doubles in
Fortran order; the partition map is identical to the layout used by
``mnpbem.utils.multi_gpu_lu.MultiGPULU`` so the two classes are wire
compatible — a ``DistributedMatrix`` can be handed straight to
``cusolverMg`` for LU without re-shuffling the data.

Distribution layout
-------------------
- ``block_size`` columns at a time, cycled round-robin across the N
  GPUs. Default 256 matches the panel size used in the NVIDIA cuSolverMg
  samples and is what ``MultiGPULU`` uses internally.
- Per-GPU allocation is padded to ``ceil(n_blocks / n_gpus) * block_size``
  so every device hosts the same number of full blocks (the trailing
  blocks may be partially zero-padded). This is the layout cuSolverMg
  itself requires.

Storage
-------
- ``self.local_arrays`` — list of ``cupy.ndarray`` (Fortran order,
  shape ``(M, local_cols_alloc)``). This is the format Python-level
  code uses for matmul / add / sub on the local tile. cupy owns the
  underlying device memory; the array's ``.data.ptr`` gives the raw
  pointer that cuSolverMg consumes.
- ``self.array_d()`` — ``ctypes`` array of ``c_void_p`` holding the
  raw device pointers (one per GPU), built lazily from the
  ``local_arrays``. This is what cuSolverMg's getrf / getrs eat
  directly; no scatter-back to host is required.

The two views share the underlying allocation, so cuSolverMg LU
operates in place on the same memory the Python side sees.

Public API (subset used by the BEM build pipeline)
--------------------------------------------------
- ``DistributedMatrix(shape, dtype, n_gpus, ...)`` — allocate, zero-init.
- ``from_host(A)`` — scatter a host numpy array.
- ``from_func(shape, dtype, n_gpus, eval_func)`` — call ``eval_func``
  per-GPU to fill local tiles without any host roundtrip (used by the
  BEM assembly where each Green-function block is built on the device
  that will end up owning it).
- ``to_host()`` — gather all tiles into a single ``np.ndarray`` (only
  used by tests / debug paths because of the memory footprint).
- ``__matmul__``, ``__add__``, ``__sub__`` — distributed elementwise /
  matmul. Add / sub assume identical layouts; matmul broadcasts the
  full right-hand matrix to each device and computes the local column
  slice of the result.
- ``lu_factor(backend='cusolvermg')`` — return a ``MultiGPULU`` handle
  whose distributed buffers are this matrix's tiles. The matrix data is
  consumed in place; do not reuse the matrix afterwards unless you
  scatter again.
- ``free()`` — release all per-GPU buffers.

The class does not register itself in ``__init__.py`` because callers
that need it import it explicitly (the BEM assembly path is gated
behind ``MNPBEM_VRAM_SHARE_DISTRIBUTED=1``).

References
----------
- ``mnpbem.utils.multi_gpu_lu.MultiGPULU`` — cuSolverMg block-cyclic LU
  which this class is designed to feed.
- NVIDIA cuSolverMg sample ``mgGetrf.cu`` for the canonical 1-D layout.
"""

from __future__ import annotations

import ctypes
from ctypes import c_int, c_int64, c_void_p
from typing import Any, Callable, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# cupy import — required (this module is only used in the GPU path).
# ---------------------------------------------------------------------------

try:
    import cupy as _cp  # type: ignore
    _CUPY_OK: bool = True
    _CUPY_ERR: Optional[str] = None
except Exception as _exc:  # pragma: no cover - defensive
    _cp = None  # type: ignore
    _CUPY_OK = False
    _CUPY_ERR = repr(_exc)


def _require_cupy() -> None:
    if not _CUPY_OK:
        raise RuntimeError(
            '[error] DistributedMatrix requires cupy ({})'.format(_CUPY_ERR))


# ---------------------------------------------------------------------------
# Block-cyclic distribution helpers (identical to multi_gpu_lu)
# ---------------------------------------------------------------------------

def _block_cyclic_layout(N: int,
        n_gpus: int,
        block_size: int) -> Tuple[int, int, List[int]]:
    """Return ``(n_blocks, local_cols_alloc, owners)``.

    ``owners[ib]`` is the GPU index that owns block ``ib``. Each GPU is
    allocated ``local_cols_alloc = ceil(n_blocks / n_gpus) * block_size``
    columns so every device has the same buffer shape (trailing blocks
    are zero-padded when ``N`` is not a multiple of ``block_size *
    n_gpus``). This matches what ``cusolverMg`` requires.
    """

    n_blocks = (N + block_size - 1) // block_size
    max_blocks_per_gpu = (n_blocks + n_gpus - 1) // n_gpus
    local_cols_alloc = max_blocks_per_gpu * block_size
    owners = [ib % n_gpus for ib in range(n_blocks)]
    return n_blocks, local_cols_alloc, owners


def _global_to_local_chunks(N: int,
        n_gpus: int,
        block_size: int,
        gpu_idx: int) -> List[Tuple[int, int, int, int]]:
    """Return a list of ``(g_start, g_stop, l_start, l_stop)`` tuples.

    Each tuple says: global columns ``[g_start, g_stop)`` of the full
    matrix map to local columns ``[l_start, l_stop)`` of the tile on
    GPU ``gpu_idx``. The local offset advances by ``block_size`` per
    iteration regardless of how many global columns the block has
    (trailing partial blocks pad to the right with zeros).
    """

    n_blocks = (N + block_size - 1) // block_size
    chunks: List[Tuple[int, int, int, int]] = []
    local_offset = 0
    for ib in range(n_blocks):
        if ib % n_gpus != gpu_idx:
            continue
        g_start = ib * block_size
        g_stop = min(N, g_start + block_size)
        ncols = g_stop - g_start
        chunks.append((g_start, g_stop, local_offset, local_offset + ncols))
        local_offset += block_size  # advance by full block (pad trailing)
    return chunks


# ---------------------------------------------------------------------------
# DistributedMatrix
# ---------------------------------------------------------------------------

class DistributedMatrix(object):
    """Block-cyclic column-distributed dense matrix across N GPUs.

    Storage layout matches what ``cusolverMg`` consumes (1-D column
    block-cyclic), so factoring with ``MultiGPULU`` does not require a
    rescatter.

    Parameters
    ----------
    shape : tuple of (M, N)
        Full logical matrix dimensions.
    dtype : numpy dtype
        Element dtype; ``complex128`` is the BEM default but the class
        supports any numeric dtype cupy can allocate.
    n_gpus : int
        Number of CUDA devices to split across.
    device_ids : list of int, optional
        Explicit CUDA device ids. Defaults to ``[0, 1, ..., n_gpus-1]``.
    block_size : int
        Column block size. Default 256 matches the cuSolverMg samples;
        clamped to multiples of 32 (warp size).
    zero_init : bool
        If True, the per-GPU allocations are zero-initialized. Defaults
        to True; pass False when the buffer is going to be overwritten
        by ``from_func`` to skip the launch.
    """

    def __init__(self,
            shape: Tuple[int, int],
            dtype: Any,
            n_gpus: int,
            device_ids: Optional[List[int]] = None,
            block_size: int = 256,
            zero_init: bool = True) -> None:

        _require_cupy()
        assert len(shape) == 2, '[error] DistributedMatrix is 2-D only'
        M, N = int(shape[0]), int(shape[1])
        assert M > 0 and N > 0, '[error] DistributedMatrix requires positive shape'

        self._shape: Tuple[int, int] = (M, N)
        self._dtype: np.dtype = np.dtype(dtype)
        self.n_gpus: int = int(n_gpus)
        if device_ids is None:
            self.device_ids: List[int] = list(range(self.n_gpus))
        else:
            self.device_ids = [int(d) for d in device_ids]
        assert len(self.device_ids) == self.n_gpus, \
            '[error] <device_ids> length must equal <n_gpus>'

        # Clamp block_size to >=32 and <= max practical for cuSolverMg.
        # (cuSolverMg expects multiples of warp; 256 is the recommended
        # value from the official samples).
        blk = max(32, int(block_size))
        max_blk = max(32, ((N // self.n_gpus + 31) // 32) * 32)
        blk = min(blk, max_blk)
        if blk < 32:
            blk = 32
        self.block_size: int = blk

        n_blocks, local_cols_alloc, owners = _block_cyclic_layout(
            N, self.n_gpus, self.block_size)
        self.n_blocks: int = n_blocks
        self.local_cols_alloc: int = local_cols_alloc
        self._owners: List[int] = owners

        # Per-GPU "active" column count (excluding pad). Useful for
        # downstream code that wants to know how much real data lives
        # on each device.
        self._local_cols_active: List[int] = []
        for g in range(self.n_gpus):
            cnt = 0
            for ib in range(n_blocks):
                if owners[ib] != g:
                    continue
                start = ib * self.block_size
                stop = min(N, start + self.block_size)
                cnt += (stop - start)
            self._local_cols_active.append(cnt)

        # Per-GPU buffers: cupy arrays in Fortran order, shape (M,
        # local_cols_alloc). We allocate one cupy ndarray per device
        # under that device's context so the memory pool used for the
        # allocation is the right one. We add an explicit runtime
        # setDevice in addition to the Device context manager because
        # cupy's context handling can drift after foreign-library
        # (e.g. cuSolverMg ctypes) calls leave the current device set
        # to something different from cupy's idea of "current".
        self.local_arrays: List[Any] = []
        for g, dev in enumerate(self.device_ids):
            _cp.cuda.runtime.setDevice(dev)
            with _cp.cuda.Device(dev):
                if zero_init:
                    arr = _cp.zeros(
                        (M, self.local_cols_alloc),
                        dtype=self._dtype,
                        order='F',
                    )
                else:
                    arr = _cp.empty(
                        (M, self.local_cols_alloc),
                        dtype=self._dtype,
                        order='F',
                    )
                self.local_arrays.append(arr)

        # ctypes array of c_void_p — what cuSolverMg consumes. Built
        # lazily on first request because some callers never need it.
        self._array_d_cache: Optional[Any] = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def shape(self) -> Tuple[int, int]:
        return self._shape

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    @property
    def itemsize(self) -> int:
        return int(self._dtype.itemsize)

    @property
    def local_cols_active(self) -> List[int]:
        return list(self._local_cols_active)

    # ------------------------------------------------------------------
    # Local-tile access
    # ------------------------------------------------------------------

    def local(self, gpu_idx: int) -> Any:
        """Return the per-GPU cupy view (writable, Fortran-order)."""
        return self.local_arrays[gpu_idx]

    def global_col_range(self,
            gpu_idx: int) -> List[Tuple[int, int, int, int]]:
        """Return list of ``(g_start, g_stop, l_start, l_stop)`` chunks
        mapping local cols on GPU ``gpu_idx`` to global cols."""
        return _global_to_local_chunks(
            self._shape[1], self.n_gpus, self.block_size, gpu_idx)

    def owner_of_block(self, ib: int) -> int:
        """Return the GPU index that owns global block ``ib``."""
        return self._owners[ib]

    # ------------------------------------------------------------------
    # cuSolverMg pointer array (lazy)
    # ------------------------------------------------------------------

    def array_d(self) -> Any:
        """Return ``(c_void_p * n_gpus)`` of raw device pointers.

        cuSolverMg expects this exact format for the distributed input.
        We materialize it lazily because not every consumer needs raw
        ctypes pointers.
        """
        if self._array_d_cache is not None:
            return self._array_d_cache
        arr = (c_void_p * self.n_gpus)()
        for g in range(self.n_gpus):
            arr[g] = c_void_p(int(self.local_arrays[g].data.ptr))
        self._array_d_cache = arr
        return arr

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_host(cls,
            A_host: np.ndarray,
            n_gpus: int,
            device_ids: Optional[List[int]] = None,
            block_size: int = 256) -> 'DistributedMatrix':
        """Scatter a host numpy array to N GPUs (block-cyclic)."""

        _require_cupy()
        assert A_host.ndim == 2, '[error] from_host expects 2-D array'
        M, N = A_host.shape
        dm = cls(
            (M, N), A_host.dtype, n_gpus,
            device_ids=device_ids, block_size=block_size,
            zero_init=False)

        # Build per-GPU tiles on the host first (zero-pad trailing) then
        # H2D in one shot per device. Faster than copying block by block
        # because each cudaMemcpy has fixed launch overhead.
        A_f = np.asfortranarray(A_host)
        for g, dev in enumerate(dm.device_ids):
            chunks = dm.global_col_range(g)
            tile_host = np.zeros(
                (M, dm.local_cols_alloc),
                dtype=dm._dtype,
                order='F',
            )
            for (g_start, g_stop, l_start, l_stop) in chunks:
                tile_host[:, l_start:l_stop] = A_f[:, g_start:g_stop]
            _cp.cuda.runtime.setDevice(dev)
            with _cp.cuda.Device(dev):
                dm.local_arrays[g].set(tile_host)
                _cp.cuda.runtime.deviceSynchronize()
        return dm

    @classmethod
    def from_func(cls,
            shape: Tuple[int, int],
            dtype: Any,
            n_gpus: int,
            eval_func: Callable[[int, int, int], Any],
            device_ids: Optional[List[int]] = None,
            block_size: int = 256) -> 'DistributedMatrix':
        """Build by calling ``eval_func`` per-block on the owning GPU.

        ``eval_func(gpu_idx, g_col_start, g_col_stop)`` is called inside
        the ``cupy.cuda.Device(gpu_idx)`` context manager for each block
        owned by GPU ``gpu_idx``. It must return an array (cupy or
        numpy) of shape ``(M, g_col_stop - g_col_start)`` and dtype
        ``dtype``. Cupy returns stay on the device (zero copy); numpy
        returns are uploaded on the spot.

        This is the most efficient build path because no host roundtrip
        is required — each GPU computes only its own columns.
        """

        _require_cupy()
        dm = cls(
            shape, dtype, n_gpus,
            device_ids=device_ids, block_size=block_size,
            zero_init=True)

        # Pre-sync every device so the eval_func sees a clean state.
        # Without this, residual async work from foreign libraries
        # (cuSolverMg / cuBLAS direct ctypes calls) can leave the
        # CUDA runtime current-device tracker out of sync with cupy's
        # internal state, triggering wrong-device-tagged allocations
        # inside the user's eval_func.
        for dev in dm.device_ids:
            _cp.cuda.runtime.setDevice(dev)
            _cp.cuda.runtime.deviceSynchronize()

        for g, dev in enumerate(dm.device_ids):
            chunks = dm.global_col_range(g)
            # Explicit setDevice in addition to the with-block — cupy's
            # Device context handling can get confused after cuSolverMg
            # calls have been made on a subset of devices in the same
            # process. The runtime setDevice is unconditional.
            _cp.cuda.runtime.setDevice(dev)
            with _cp.cuda.Device(dev):
                local_tile = dm.local_arrays[g]
                for (g_start, g_stop, l_start, l_stop) in chunks:
                    block = eval_func(g, g_start, g_stop)
                    if not isinstance(block, _cp.ndarray):
                        block = _cp.asarray(block, dtype=dm._dtype)
                    elif block.dtype != dm._dtype:
                        block = block.astype(dm._dtype)
                    # Defensive: if eval_func returned an array on a
                    # different device (e.g. cupy's "current device"
                    # tracker drifted after foreign ctypes calls), pull
                    # it across via host. The host roundtrip is
                    # measurable but avoids tripping cupy's peer-access
                    # fallback path in the per-element scatter kernel
                    # below, which has been observed to fail with an
                    # ``illegal memory access`` after foreign ctypes
                    # multi-device libraries (cuSolverMg) have run.
                    if block.device.id != dev:
                        block_host = _cp.asnumpy(block)
                        block = _cp.asarray(block_host, dtype=dm._dtype)
                    expected_cols = g_stop - g_start
                    assert block.shape == (dm._shape[0], expected_cols), \
                        ('[error] eval_func returned shape {}, expected ({}, {})'
                         .format(block.shape, dm._shape[0], expected_cols))
                    local_tile[:, l_start:l_stop] = block
                # Synchronize before leaving the device context to make
                # sure all assignments have committed before <block>
                # goes out of scope and gets garbage-collected.
                _cp.cuda.runtime.deviceSynchronize()
        return dm

    # ------------------------------------------------------------------
    # Host gather
    # ------------------------------------------------------------------

    def to_host(self) -> np.ndarray:
        """Gather all per-GPU tiles into a host numpy array."""

        M, N = self._shape
        out = np.empty((M, N), dtype=self._dtype, order='F')
        for g, dev in enumerate(self.device_ids):
            chunks = self.global_col_range(g)
            _cp.cuda.runtime.setDevice(dev)
            with _cp.cuda.Device(dev):
                # Synchronize first so any pending kernel that wrote to
                # this tile has finished. Without this we can hit
                # races when local_arrays[g] was the output of an
                # async kernel earlier (e.g. matmul) and we then read
                # it on the host.
                _cp.cuda.runtime.deviceSynchronize()
                tile_host = _cp.asnumpy(self.local_arrays[g])
            for (g_start, g_stop, l_start, l_stop) in chunks:
                out[:, g_start:g_stop] = tile_host[:, l_start:l_stop]
        return out

    # ------------------------------------------------------------------
    # Elementwise add / sub (same distribution required)
    # ------------------------------------------------------------------

    def _check_same_layout(self, B: 'DistributedMatrix') -> None:
        assert isinstance(B, DistributedMatrix), \
            '[error] DistributedMatrix elementwise op requires DistributedMatrix RHS'
        assert B._shape == self._shape, \
            '[error] shape mismatch: {} vs {}'.format(self._shape, B._shape)
        assert B.n_gpus == self.n_gpus, \
            '[error] n_gpus mismatch'
        assert B.block_size == self.block_size, \
            '[error] block_size mismatch'
        assert B.device_ids == self.device_ids, \
            '[error] device_ids mismatch'

    def __add__(self, B: 'DistributedMatrix') -> 'DistributedMatrix':
        self._check_same_layout(B)
        out = DistributedMatrix(
            self._shape, self._dtype, self.n_gpus,
            device_ids=self.device_ids, block_size=self.block_size,
            zero_init=False)
        for g, dev in enumerate(self.device_ids):
            _cp.cuda.runtime.setDevice(dev)
            with _cp.cuda.Device(dev):
                _cp.add(self.local_arrays[g], B.local_arrays[g],
                        out=out.local_arrays[g])
                _cp.cuda.runtime.deviceSynchronize()
        return out

    def __sub__(self, B: 'DistributedMatrix') -> 'DistributedMatrix':
        self._check_same_layout(B)
        out = DistributedMatrix(
            self._shape, self._dtype, self.n_gpus,
            device_ids=self.device_ids, block_size=self.block_size,
            zero_init=False)
        for g, dev in enumerate(self.device_ids):
            _cp.cuda.runtime.setDevice(dev)
            with _cp.cuda.Device(dev):
                _cp.subtract(self.local_arrays[g], B.local_arrays[g],
                             out=out.local_arrays[g])
                _cp.cuda.runtime.deviceSynchronize()
        return out

    def iadd(self, B: 'DistributedMatrix') -> 'DistributedMatrix':
        """In-place add (modifies ``self``)."""
        self._check_same_layout(B)
        for g, dev in enumerate(self.device_ids):
            _cp.cuda.runtime.setDevice(dev)
            with _cp.cuda.Device(dev):
                self.local_arrays[g] += B.local_arrays[g]
                _cp.cuda.runtime.deviceSynchronize()
        return self

    def isub(self, B: 'DistributedMatrix') -> 'DistributedMatrix':
        """In-place sub (modifies ``self``)."""
        self._check_same_layout(B)
        for g, dev in enumerate(self.device_ids):
            _cp.cuda.runtime.setDevice(dev)
            with _cp.cuda.Device(dev):
                self.local_arrays[g] -= B.local_arrays[g]
                _cp.cuda.runtime.deviceSynchronize()
        return self

    # ------------------------------------------------------------------
    # Distributed matmul
    # ------------------------------------------------------------------

    def __matmul__(self, B: Any) -> 'DistributedMatrix':
        """Compute ``C = self @ B`` keeping the same distribution.

        ``self`` shape: ``(M, N)``. Each GPU holds a column slice of
        ``self`` (call those columns ``A_g``); the global ``A`` is
        ``concat_g (A_g, axis=1)`` after un-padding.

        Mapping ``A @ B``:
            ``A`` column ``j`` participates only in ``(A @ B)[*, k]``
            via the row ``j`` of ``B``. Equivalently, each GPU's local
            slice of ``self`` is a *column slice*, so to compute
            ``C = A @ B`` distributed across the same column partition
            we need:

              C_g (local cols of self's partition) =
                  sum_k A_full[:, k] * B[k, output_cols_g]

            This is just ``A_full @ B[:, output_cols_g]``: each device
            computes its own *output column slice* of ``C`` by reading
            the corresponding *output column slice* of ``B`` and the
            full ``A`` matrix.

        Implementation
        --------------
        ``self`` (the LHS) is column-distributed, so the "full A" view
        each GPU needs is the all-gather of its peers' tiles. We do
        this by gathering A once to the host (using ``to_host``) and
        re-uploading it to each device, then computing the local
        output slice with cupy's native matmul.

        This is O(M*N) host traffic per call — fine for the BEM build
        path where the matmul happens once per BEM Sigma assembly.

        ``B`` may be:
        - ``DistributedMatrix`` with the same distribution
        - ``np.ndarray`` on the host
        - ``cupy.ndarray`` on any device
        """

        _require_cupy()
        M, K = self._shape
        # Bring the LHS to host once. This is the only host roundtrip;
        # the per-device matmul stays on-device thereafter.
        A_host = self.to_host()

        # Resolve B to a host ndarray for the K-dim consistency check.
        if isinstance(B, DistributedMatrix):
            B_host = B.to_host()
            B_shape = B._shape
            B_dtype = B._dtype
            out_n_gpus = B.n_gpus
            out_device_ids = B.device_ids
            out_block = B.block_size
        elif isinstance(B, _cp.ndarray):
            B_host = _cp.asnumpy(B)
            B_shape = B_host.shape
            B_dtype = B_host.dtype
            out_n_gpus = self.n_gpus
            out_device_ids = self.device_ids
            out_block = self.block_size
        else:
            B_host = np.asarray(B)
            B_shape = B_host.shape
            B_dtype = B_host.dtype
            out_n_gpus = self.n_gpus
            out_device_ids = self.device_ids
            out_block = self.block_size

        assert len(B_shape) == 2, '[error] matmul expects 2-D RHS'
        assert B_shape[0] == K, \
            ('[error] dim mismatch: self {} @ B {}'
             .format(self._shape, B_shape))

        out_dtype = np.result_type(self._dtype, B_dtype)
        N_out = B_shape[1]
        out = DistributedMatrix(
            (M, N_out), out_dtype, out_n_gpus,
            device_ids=out_device_ids, block_size=out_block,
            zero_init=False)

        for g, dev in enumerate(out.device_ids):
            chunks = out.global_col_range(g)
            _cp.cuda.runtime.setDevice(dev)
            with _cp.cuda.Device(dev):
                A_g = _cp.asarray(A_host, dtype=out_dtype)
                # Zero the entire tile first (safe even when there's no
                # output column on this device, e.g. partial last row).
                out.local_arrays[g].fill(0)
                for (g_start, g_stop, l_start, l_stop) in chunks:
                    B_slice = _cp.asarray(
                        B_host[:, g_start:g_stop], dtype=out_dtype)
                    out.local_arrays[g][:, l_start:l_stop] = A_g @ B_slice
                # Synchronize before leaving the device context so the
                # gemm finishes before A_g/B_slice get garbage-collected.
                _cp.cuda.runtime.deviceSynchronize()
                # Free transient copies of A / B on this device.
                del A_g
                _cp.get_default_memory_pool().free_all_blocks()
        return out

    # ------------------------------------------------------------------
    # LU integration with MultiGPULU
    # ------------------------------------------------------------------

    def lu_factor(self,
            backend: str = 'cusolvermg') -> Any:
        """Factorize this matrix in place using cuSolverMg.

        Returns a ``MultiGPULU`` handle whose internal distributed
        buffers point at this matrix's tiles. The matrix is consumed
        in place — after this call ``self`` holds the L/U factors.

        Notes
        -----
        ``MultiGPULU.factor()`` as written today owns its buffers
        (allocates them via cudaMalloc, scatters from host, then
        gathers back on solve). To reuse our distributed buffers
        without re-scattering, we take advantage of the fact that
        ``self.array_d()`` produces the same ctypes pointer array
        ``MultiGPULU`` would have built internally with the same block
        size, then plug our pointers into a freshly-created
        ``MultiGPULU`` handle and let it run the cuSolverMg call path.

        This avoids the duplicate ``M * N`` allocation that would
        otherwise happen if we round-tripped through the host.
        """

        # Use module reference so the lib globals stay live after
        # cusolvermg_available() finishes its lazy dlopen.
        from . import multi_gpu_lu as _mglu  # local import — avoids cycle
        from .multi_gpu_lu import (
            MultiGPULU,
            _bind_cusolvermg,
            _cuda_set_device,
            _cuda_malloc,
            _cuda_device_sync,
            _check_status,
            _cuda_dtype_for,
            _itemsize_for,
            cusolvermg_available,
            CUDA_C_64F,
            CUDA_R_64F,
            CUDA_C_32F,
            CUDA_R_32F,
            GRID_MAPPING_COL_MAJOR,
        )

        if backend != 'cusolvermg':
            raise NotImplementedError(
                '[error] DistributedMatrix.lu_factor: only cusolvermg supported, got {}'
                .format(backend))
        if not cusolvermg_available():
            raise RuntimeError(
                '[error] cuSolverMg unavailable for DistributedMatrix.lu_factor')
        _bind_cusolvermg()
        _libcusolverMg = _mglu._libcusolverMg
        _libcudart = _mglu._libcudart

        M, N = self._shape
        assert M == N, '[error] DistributedMatrix LU requires square matrix'

        # Sample of dtype for cuda_dtype_for — we need a 0-D ndarray
        # with the right dtype to feed the helper.
        dtype_sample = np.empty(0, dtype=self._dtype)
        cuda_dtype = _cuda_dtype_for(dtype_sample)
        itemsz = _itemsize_for(self._dtype)

        lu = MultiGPULU(
            self.n_gpus, backend='cusolvermg',
            device_ids=list(self.device_ids))
        lu.N = N
        lu.dtype = self._dtype
        lu.cuda_dtype = cuda_dtype
        lu.col_blk_size = self.block_size
        lu.local_cols_alloc = self.local_cols_alloc
        lu.tile_cols_per_gpu = list(self._local_cols_active)

        # Create cuSolverMg handle, grid, descriptor.
        h = c_void_p(0)
        _check_status(
            _libcusolverMg.cusolverMgCreate(ctypes.byref(h)),
            'cusolverMgCreate')
        lu.handle = h

        dev_arr_c = (c_int * self.n_gpus)(*self.device_ids)
        _check_status(
            _libcusolverMg.cusolverMgDeviceSelect(
                h, c_int(self.n_gpus), dev_arr_c),
            'cusolverMgDeviceSelect')

        grid = c_void_p(0)
        dev_arr32 = (ctypes.c_int32 * self.n_gpus)(*self.device_ids)
        _check_status(
            _libcusolverMg.cusolverMgCreateDeviceGrid(
                ctypes.byref(grid), ctypes.c_int32(1),
                ctypes.c_int32(self.n_gpus),
                dev_arr32, c_int(GRID_MAPPING_COL_MAJOR)),
            'cusolverMgCreateDeviceGrid')
        lu.grid = grid

        descr = c_void_p(0)
        _check_status(
            _libcusolverMg.cusolverMgCreateMatrixDesc(
                ctypes.byref(descr), c_int64(N), c_int64(N),
                c_int64(N), c_int64(self.block_size),
                c_int(cuda_dtype), grid),
            'cusolverMgCreateMatrixDesc')
        lu.descr = descr

        # Hand our tile pointers to MultiGPULU. Allocate IPIV separately
        # — that's per-GPU state owned by the LU handle.
        lu.array_d_A = self.array_d()
        ptrs_IPIV = (c_void_p * self.n_gpus)()
        for g, dev in enumerate(self.device_ids):
            _cuda_set_device(dev)
            ptrs_IPIV[g] = c_void_p(_cuda_malloc(self.local_cols_alloc * 4))
        lu.array_d_IPIV = ptrs_IPIV

        # Query and allocate workspace.
        lwork = c_int64(0)
        _check_status(
            _libcusolverMg.cusolverMgGetrf_bufferSize(
                h, c_int(N), c_int(N), lu.array_d_A,
                c_int(1), c_int(1), descr,
                lu.array_d_IPIV, c_int(cuda_dtype),
                ctypes.byref(lwork)),
            'cusolverMgGetrf_bufferSize')
        lu.work_lwork = int(lwork.value)

        ptrs_w = (c_void_p * self.n_gpus)()
        for g, dev in enumerate(self.device_ids):
            _cuda_set_device(dev)
            ptrs_w[g] = c_void_p(_cuda_malloc(lu.work_lwork * itemsz))
        lu.array_d_work = ptrs_w

        info = c_int(0)
        _check_status(
            _libcusolverMg.cusolverMgGetrf(
                h, c_int(N), c_int(N), lu.array_d_A,
                c_int(1), c_int(1), descr,
                lu.array_d_IPIV, c_int(cuda_dtype),
                lu.array_d_work, c_int64(lu.work_lwork),
                ctypes.byref(info)),
            'cusolverMgGetrf')
        if info.value != 0:
            raise RuntimeError(
                '[error] cusolverMgGetrf reports info={} (singular or invalid)'
                .format(info.value))
        _cuda_device_sync()

        # The LU factors live in the DistributedMatrix's tiles. Override
        # the LU handle's close() so it doesn't double-free our buffers
        # (DistributedMatrix.free() owns them via cupy.ndarray refcount).
        # We still need MultiGPULU.close() to release IPIV / workspace
        # and the cuSolverMg objects, so we replace close() with a
        # version that frees IPIV explicitly and skips array_d_A.
        def _close_keep_A() -> None:
            try:
                # Sync first so any in-flight kernels don't trip the
                # subsequent destroy calls.
                for dev_id in lu.device_ids:
                    try:
                        _cuda_set_device(dev_id)
                        _cuda_device_sync()
                    except Exception:
                        pass
                # Free workspace.
                if lu.array_d_work is not None:
                    for g_idx, dev_id in enumerate(lu.device_ids):
                        try:
                            _cuda_set_device(dev_id)
                            _libcudart.cudaFree(c_void_p(int(lu.array_d_work[g_idx] or 0)))
                        except Exception:
                            pass
                    lu.array_d_work = None
                # Free IPIV (we allocated it). Skip array_d_A.
                if lu.array_d_IPIV is not None:
                    for g_idx, dev_id in enumerate(lu.device_ids):
                        try:
                            _cuda_set_device(dev_id)
                            _libcudart.cudaFree(c_void_p(int(lu.array_d_IPIV[g_idx] or 0)))
                        except Exception:
                            pass
                    lu.array_d_IPIV = None
                # Mark array_d_A as None so the parent destructor (if
                # ever invoked) doesn't loop over it.
                lu.array_d_A = None
                # Tear down cuSolverMg objects.
                if lu.descr is not None:
                    _libcusolverMg.cusolverMgDestroyMatrixDesc(lu.descr)
                    lu.descr = None
                if lu.grid is not None:
                    _libcusolverMg.cusolverMgDestroyGrid(lu.grid)
                    lu.grid = None
                if lu.handle is not None:
                    _libcusolverMg.cusolverMgDestroy(lu.handle)
                    lu.handle = None
            except Exception:
                pass

        lu.close = _close_keep_A
        return lu

    # ------------------------------------------------------------------
    # Free
    # ------------------------------------------------------------------

    def free(self) -> None:
        """Release all per-GPU buffers.

        Drop refs *under* the right device context for each tile, then
        sync the device so any pending dealloc has completed before we
        return. Without the per-device sync, cupy can dealloc on the
        wrong device later (during GC), leading to ``illegal memory
        access`` on subsequent multi-device work.
        """
        if not self.local_arrays:
            return
        # Replace the list with Nones one slot at a time. Each
        # assignment drops the previous cupy.ndarray refcount; the
        # destructor runs synchronously inside the with-block while
        # we're on the right device.
        for g, dev in enumerate(self.device_ids):
            try:
                _cp.cuda.runtime.setDevice(dev)
                with _cp.cuda.Device(dev):
                    arr = self.local_arrays[g]
                    self.local_arrays[g] = None
                    del arr
                    _cp.cuda.runtime.deviceSynchronize()
                    _cp.get_default_memory_pool().free_all_blocks()
            except Exception:
                pass
        self.local_arrays = []
        self._array_d_cache = None

    def __del__(self) -> None:  # pragma: no cover - destructor
        try:
            self.free()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            'DistributedMatrix(shape={}, dtype={}, n_gpus={}, '
            'block_size={}, device_ids={}, local_cols_alloc={})'
            .format(self._shape, self._dtype, self.n_gpus,
                    self.block_size, self.device_ids, self.local_cols_alloc))
