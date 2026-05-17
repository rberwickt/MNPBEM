"""GPU dispatch helpers for LU factor / solve.

Provides a unified API that automatically routes large dense linear systems
to a CuPy + cuSOLVER backend when:

- ``cupy`` is importable
- ``MNPBEM_GPU=1`` (default OFF — explicit opt-in)
- the matrix dimension is at least ``MNPBEM_GPU_THRESHOLD`` (default 1500)

Below the threshold the helpers fall back to ``scipy.linalg.lu_factor`` /
``lu_solve`` to avoid host <-> device transfer overhead on small problems.

The factor object is opaque: a tuple whose first element is ``'cpu'`` or
``'gpu'`` indicating where the LU lives.  ``lu_solve_dispatch`` returns a
NumPy array regardless of backend so callers do not need to be aware of the
device.
"""

from __future__ import annotations

import os
import warnings
from typing import Any, Optional, Tuple

import numpy as np
from scipy.linalg import lu_factor as _scipy_lu_factor
from scipy.linalg import lu_solve as _scipy_lu_solve
from scipy.linalg import solve as _scipy_solve

USE_GPU: bool = os.environ.get("MNPBEM_GPU", "0") == "1"
GPU_THRESHOLD: int = int(os.environ.get("MNPBEM_GPU_THRESHOLD", "1500"))

try:
    import cupy as _cp  # type: ignore
    from cupyx.scipy.linalg import lu_factor as _cp_lu_factor  # type: ignore
    from cupyx.scipy.linalg import lu_solve as _cp_lu_solve  # type: ignore
    _CUPY_OK: bool = True
    _CUPY_IMPORT_ERROR: Optional[str] = None
except Exception as _exc:
    _cp = None  # type: ignore
    _CUPY_OK = False
    _CUPY_IMPORT_ERROR = repr(_exc)


def gpu_available() -> bool:
    return _CUPY_OK and USE_GPU


def get_install_hint() -> str:
    """User-facing install guidance for missing optional extras."""
    return (
        'For GPU acceleration: pip install "mnpbem[gpu]"\n'
        'For multi-node MPI:   pip install "mnpbem[mpi]"\n'
        'For FMM acceleration: pip install "mnpbem[fmm]"\n'
        'For all features:     pip install "mnpbem[all]"\n'
        'See docs/INSTALL.md for prerequisites and troubleshooting.'
    )


def has_gpu_capability(verbose: bool = False) -> bool:
    """Return True iff cupy is importable and at least one CUDA device exists.

    The function never raises: it always returns a boolean and emits a
    ``RuntimeWarning`` (only when ``verbose=True``) describing the reason
    GPU acceleration is unavailable.  This lets callers gate code paths
    without try / except boilerplate.
    """
    if not _CUPY_OK:
        if verbose:
            msg = (
                'cupy is not importable ({}). GPU acceleration disabled.\n'
                '{}').format(_CUPY_IMPORT_ERROR, get_install_hint())
            warnings.warn(msg, RuntimeWarning, stacklevel=2)
        return False
    try:
        n_gpus = int(_cp.cuda.runtime.getDeviceCount())
    except Exception as exc:
        if verbose:
            msg = (
                'cupy is installed but the CUDA runtime check failed '
                '({}). CPU-only mode.').format(repr(exc))
            warnings.warn(msg, RuntimeWarning, stacklevel=2)
        return False
    if n_gpus == 0:
        if verbose:
            warnings.warn(
                'cupy is installed but no CUDA devices were found. '
                'CPU-only mode.',
                RuntimeWarning, stacklevel=2)
        return False
    return True


def require_gpu_or_raise() -> None:
    """Raise a friendly RuntimeError when MNPBEM_GPU=1 but cupy is missing.

    Used by entry points that explicitly opt into GPU. The error message
    embeds the install hint so the user does not need to look up extras
    separately.
    """
    if not USE_GPU:
        return
    if _CUPY_OK:
        return
    msg = (
        'MNPBEM_GPU=1 was set, but cupy is not available ({}).\n{}'
    ).format(_CUPY_IMPORT_ERROR, get_install_hint())
    raise RuntimeError(msg)


def _vram_share_env_defaults() -> Tuple[Optional[int], Optional[str], Optional[list]]:
    """Read ``MNPBEM_VRAM_SHARE_*`` env vars to fill in dispatch kwargs.

    Returns ``(n_gpus, backend, device_ids)`` triple where each entry is
    ``None`` when the corresponding env var is unset / disabled. The
    distributed multi-GPU path is gated by ``MNPBEM_VRAM_SHARE_GPUS>=2``;
    the optional ``MNPBEM_VRAM_SHARE`` master switch (when set to '0')
    forces the helper off so callers can disable wiring without unsetting
    every variable. Backend default is ``'cusolvermg'``; ``device_ids``
    is parsed from a comma-separated list (e.g. '0,1,2,3').
    """
    if os.environ.get('MNPBEM_VRAM_SHARE', '1').strip() in ('0', 'false', 'False'):
        return None, None, None
    raw = os.environ.get('MNPBEM_VRAM_SHARE_GPUS', '').strip()
    if raw == '':
        return None, None, None
    try:
        n_gpus_env = int(raw)
    except ValueError:
        return None, None, None
    if n_gpus_env < 2:
        return None, None, None
    backend_env = os.environ.get('MNPBEM_VRAM_SHARE_BACKEND', 'cusolvermg').strip() or 'cusolvermg'
    devs_raw = os.environ.get('MNPBEM_VRAM_SHARE_DEVICE_IDS', '').strip()
    if devs_raw:
        try:
            device_ids_env: Optional[list] = [int(x) for x in devs_raw.split(',') if x.strip() != '']
        except ValueError:
            device_ids_env = None
    else:
        device_ids_env = None
    return n_gpus_env, backend_env, device_ids_env


def lu_factor_dispatch(A: np.ndarray, **kwargs: Any) -> Tuple:
    """Factorize A on GPU when beneficial, else CPU.

    Extra ``kwargs`` are forwarded to ``scipy.linalg.lu_factor`` for the CPU
    path; GPU path uses CuPy defaults (``check_finite`` / ``overwrite_a``
    are not exposed by ``cupyx.scipy.linalg.lu_factor`` in the same way).

    Multi-GPU VRAM-share path
    -------------------------
    Pass ``n_gpus=N`` (with ``N>=2``) to distribute the matrix across
    multiple GPUs via cuSolverMg. The return tag becomes ``'mgpu'`` and
    the second slot holds a ``MultiGPULU`` handle. Falls back to single
    GPU / CPU with a warning when cuSolverMg / drivers are unavailable.
    Optional kwargs: ``backend`` ('cusolvermg'|'magma'|'nccl'),
    ``device_ids`` (list of CUDA device ids).

    Env-var auto-wiring (v1.6.2)
    ----------------------------
    When ``n_gpus`` is omitted, ``MNPBEM_VRAM_SHARE_GPUS`` (>=2) is read
    automatically along with ``MNPBEM_VRAM_SHARE_BACKEND`` and
    ``MNPBEM_VRAM_SHARE_DEVICE_IDS``. Explicit kwargs always win over
    the env defaults. Set ``MNPBEM_VRAM_SHARE=0`` to disable the
    auto-wiring without unsetting the other variables.
    """
    env_n, env_backend, env_devs = _vram_share_env_defaults()
    n_gpus = int(kwargs.pop('n_gpus', env_n if env_n is not None else 1))
    backend = kwargs.pop('backend', env_backend if env_backend is not None else 'cusolvermg')
    device_ids = kwargs.pop('device_ids', env_devs)
    if n_gpus >= 2:
        try:
            from .multi_gpu_lu import factor_multi_gpu, cusolvermg_available, warn_fallback
            if backend == 'cusolvermg' and not cusolvermg_available():
                warn_fallback('libcusolverMg.so / libcudart.so not loadable')
            else:
                lu_handle = factor_multi_gpu(
                    A, n_gpus=n_gpus, backend=backend, device_ids=device_ids)
                return ("mgpu", lu_handle, None)
        except (RuntimeError, NotImplementedError, ValueError) as exc:
            from .multi_gpu_lu import warn_fallback
            warn_fallback(repr(exc))
    if _CUPY_OK and USE_GPU and A.shape[0] >= GPU_THRESHOLD:
        A_gpu = _cp.asarray(A)
        lu_gpu, piv_gpu = _cp_lu_factor(A_gpu, overwrite_a=True)
        return ("gpu", lu_gpu, piv_gpu)
    kwargs.setdefault("check_finite", False)
    lu, piv = _scipy_lu_factor(A, **kwargs)
    return ("cpu", lu, piv)


def lu_solve_dispatch(piv_pkg: Tuple, b: np.ndarray, **kwargs: Any) -> np.ndarray:
    """Solve A x = b given a factorization produced by ``lu_factor_dispatch``.

    Returns a NumPy array on the host irrespective of where the LU lives.
    Supports the multi-GPU ``'mgpu'`` tag (cuSolverMg distributed solve).

    For the ``'mgpu'`` tag, multi-RHS solves (b.ndim == 2, ncol > chunk)
    are automatically column-chunked to dodge the cuSolverMg precision
    regression / EXECUTION_FAILED status seen when nrhs approaches N.
    The chunk width is controlled by ``MNPBEM_VRAM_SHARE_SOLVE_CHUNK``
    (default 1024). Pass ``chunk_size=...`` to override per-call.
    """
    tag = piv_pkg[0]
    if tag == "mgpu":
        lu_handle = piv_pkg[1]
        trans = kwargs.pop('trans', 'N')
        if isinstance(trans, int):
            trans = {0: 'N', 1: 'T', 2: 'C'}.get(trans, 'N')
        chunk_size = kwargs.pop('chunk_size', None)
        b_host = b
        if _CUPY_OK and isinstance(b, _cp.ndarray):
            b_host = _cp.asnumpy(b)
        # ``solve_chunked`` short-circuits to a single ``solve`` call
        # when ``b.ndim == 1`` or ``ncol <= chunk_size``, so the chunked
        # path is precision-safe AND zero-overhead for small RHS.
        return lu_handle.solve_chunked(
            np.ascontiguousarray(b_host),
            chunk_size=chunk_size,
            trans=trans)
    if tag == "gpu":
        b_gpu = _cp.asarray(b)
        x_gpu = _cp_lu_solve((piv_pkg[1], piv_pkg[2]), b_gpu)
        return _cp.asnumpy(x_gpu)
    kwargs.setdefault("check_finite", False)
    return _scipy_lu_solve((piv_pkg[1], piv_pkg[2]), b, **kwargs)


def lu_solve_native(piv_pkg: Tuple, b: Any, **kwargs: Any):
    """Cupy-passthrough variant of ``lu_solve_dispatch``.

    When the LU package is on GPU and ``b`` is a cupy ndarray, returns a
    cupy ndarray (no host round-trip).  Otherwise behaves like
    ``lu_solve_dispatch``.

    For the multi-GPU ``'mgpu'`` tag, always returns a NumPy array (no
    single-device cupy view exists for a distributed solve).
    """
    tag = piv_pkg[0]
    if tag == "mgpu":
        return lu_solve_dispatch(piv_pkg, b, **kwargs)
    if tag == "gpu":
        if _CUPY_OK and isinstance(b, _cp.ndarray):
            return _cp_lu_solve((piv_pkg[1], piv_pkg[2]), b)
        b_gpu = _cp.asarray(b)
        x_gpu = _cp_lu_solve((piv_pkg[1], piv_pkg[2]), b_gpu)
        if _CUPY_OK and isinstance(b, _cp.ndarray):
            return x_gpu
        return _cp.asnumpy(x_gpu)
    # CPU LU: if b is cupy, bring it to host
    if _CUPY_OK and isinstance(b, _cp.ndarray):
        b = _cp.asnumpy(b)
    kwargs.setdefault("check_finite", False)
    return _scipy_lu_solve((piv_pkg[1], piv_pkg[2]), b, **kwargs)


def lu_backend(piv_pkg: Tuple) -> str:
    return piv_pkg[0]


def eye_like_lu(piv_pkg: Tuple,
        n: int,
        dtype: Any = None) -> Any:
    # Return an identity matrix on the same device as the LU package.
    # Used by callers that build A^{-1} = lu_solve(LU, eye(n)) and then
    # multiply with another matrix that lives on the same device as the
    # LU.  For the multi-GPU 'mgpu' tag the distributed solve only
    # accepts host arrays, so fall back to numpy.
    tag = piv_pkg[0]
    if tag == 'gpu' and _CUPY_OK:
        if dtype is None:
            return _cp.eye(n)
        return _cp.eye(n, dtype = dtype)
    if dtype is None:
        return np.eye(n)
    return np.eye(n, dtype = dtype)


def to_host(x: Any) -> np.ndarray:
    # Materialize ``x`` on the host as a NumPy array.  Accepts numpy
    # arrays, cupy arrays, and array-like scalars.  Used at boundaries
    # where downstream code cannot accept a cupy ndarray.
    if _CUPY_OK and isinstance(x, _cp.ndarray):
        return _cp.asnumpy(x)
    return np.asarray(x)


def is_cupy_array(x: Any) -> bool:
    # True iff ``x`` is a cupy ndarray.  Cheap helper that avoids the
    # ``hasattr(x, 'get')`` idiom (which also matches dict).
    return _CUPY_OK and isinstance(x, _cp.ndarray)


def solve_dispatch(A: np.ndarray, b: np.ndarray, **kwargs: Any) -> np.ndarray:
    """One-shot Ax=b: dense solve on GPU when beneficial, else CPU.

    Used by code paths that build a small dense system on the fly without
    reusing the factorization.  Falls back to ``scipy.linalg.solve`` below
    threshold or when CuPy is unavailable.

    When ``MNPBEM_VRAM_SHARE_GPUS>=2`` is set (v1.6.2), the system is
    factorized via the cuSolverMg multi-GPU path and solved in one step.
    Explicit ``n_gpus`` / ``backend`` kwargs override the env defaults.
    """
    env_n, _, _ = _vram_share_env_defaults()
    has_mgpu_kw = ('n_gpus' in kwargs and int(kwargs['n_gpus']) >= 2)
    if env_n is not None or has_mgpu_kw:
        lu_pkg = lu_factor_dispatch(A, **kwargs)
        return lu_solve_dispatch(lu_pkg, b)
    if _CUPY_OK and USE_GPU and A.shape[0] >= GPU_THRESHOLD:
        A_gpu = _cp.asarray(A)
        b_gpu = _cp.asarray(b)
        x_gpu = _cp.linalg.solve(A_gpu, b_gpu)
        return _cp.asnumpy(x_gpu)
    kwargs.setdefault("check_finite", False)
    return _scipy_solve(A, b, **kwargs)


def eigh_dispatch(A: np.ndarray,
        k: Optional[int] = None,
        **kwargs: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Hermitian eigendecomposition dispatch.

    Returns ``(w, v)`` as host NumPy arrays regardless of backend.

    Backend selection (v1.7.4 — eigh GPU acceleration)
    ----------------------------------------------------
    1. ``k`` is None or ``k >= n - 1`` → full spectrum
       - Multi-GPU cuSolverMg eigh when ``MNPBEM_VRAM_SHARE_GPUS>=2`` AND
         the matrix is real-symmetric / complex-hermitian (cusolverMgSyevd
         requires that).
       - Single-GPU cupy.linalg.eigh when MNPBEM_GPU=1 and n>=GPU_THRESHOLD.
       - Else scipy.linalg.eigh.
    2. ``k < n - 1`` (partial spectrum, smallest real eigenvalues)
       - Full GPU eigh on cupy + select smallest k on host.
       - Else scipy.sparse.linalg.eigsh ('SR') for memory savings on
         very large matrices.

    Notes
    -----
    cuSolverMg Syevd handles real-symmetric (float32/64) and complex-
    Hermitian (complex64/128) input only; non-Hermitian general
    eigenvalue problems must go through ``eig_dispatch``.

    Parameters
    ----------
    A : ndarray, (n, n) Hermitian or symmetric
    k : int, optional
        If given and k < n - 1, return only the k smallest-real-part
        eigenvalues / eigenvectors.  Otherwise full spectrum is returned.

    Returns
    -------
    w : (k or n,) ndarray, ascending real-part order.
    v : (n, k or n) ndarray, columns are eigenvectors.
    """
    n = A.shape[0]
    # 1. partial-spectrum path
    if k is not None and 0 < k < n - 1:
        # Strategy: prefer full GPU eigh + slice for moderate n;
        # scipy.sparse eigsh for very large CPU-only matrices.
        if _CUPY_OK and USE_GPU and n >= GPU_THRESHOLD:
            A_gpu = _cp.asarray(A)
            w_gpu, v_gpu = _cp.linalg.eigh(A_gpu)
            w = _cp.asnumpy(w_gpu)
            v = _cp.asnumpy(v_gpu)
            idx = np.argsort(w.real)[:k]
            return w[idx], v[:, idx]
        # CPU partial path
        try:
            from scipy.sparse.linalg import eigsh as _scipy_eigsh
            w, v = _scipy_eigsh(A, k=k, which='SA', maxiter=1000)
            idx = np.argsort(w.real)
            return w[idx], v[:, idx]
        except Exception:
            # Fallback to full LAPACK
            pass
        from scipy.linalg import eigh as _scipy_eigh
        kwargs.setdefault("check_finite", False)
        w, v = _scipy_eigh(A, **kwargs)
        idx = np.argsort(w.real)[:k]
        return w[idx], v[:, idx]

    # 2. full-spectrum path
    # 2a. multi-GPU cuSolverMg path — opt-in via MNPBEM_GPU_EIGH_MGPU=1.
    #
    # v1.7.5 — Three fixes make the binding production-ready:
    #   (1) Pass ``W`` as a host pointer (NVIDIA sample pattern); device
    #       pointer caused SIGSEGV during the Householder back-transform.
    #   (2) Call ``cudaDeviceEnablePeerAccess`` between every device
    #       pair before grid creation (also from the NVIDIA sample).
    #   (3) Cache the cusolverMg handle + grid at the class level; CUDA
    #       12.4 returns status 6 (EXECUTION_FAILED) on the second
    #       ``cusolverMgCreate`` in the same process.
    #
    # The path is still opt-in (default OFF) because we have not yet
    # tested it across all production matrix sizes / dtypes.  Set
    # ``MNPBEM_GPU_EIGH_MGPU=1`` to enable.
    env_n, _, env_devs = _vram_share_env_defaults()
    want_mgeig = (
        os.environ.get('MNPBEM_GPU_EIGH_MGPU', '0').strip() == '1'
        and env_n is not None and n >= GPU_THRESHOLD)
    if want_mgeig:
        try:
            mgeig = MultiGPUEigh(n_gpus=env_n, device_ids=env_devs)
            w, v = mgeig.eigh(A)
            mgeig.close()
            return w, v
        except (RuntimeError, NotImplementedError, ValueError) as exc:
            try:
                from .multi_gpu_lu import warn_fallback as _wf
                _wf('cusolverMgSyevd unavailable: ' + repr(exc))
            except Exception:
                pass

    # 2b. single-GPU cupy path
    if _CUPY_OK and USE_GPU and n >= GPU_THRESHOLD:
        A_gpu = _cp.asarray(A)
        w_gpu, v_gpu = _cp.linalg.eigh(A_gpu)
        return _cp.asnumpy(w_gpu), _cp.asnumpy(v_gpu)

    # 2c. CPU fallback
    from scipy.linalg import eigh as _scipy_eigh
    kwargs.setdefault("check_finite", False)
    return _scipy_eigh(A, **kwargs)


def eig_dispatch(A: np.ndarray,
        k: Optional[int] = None,
        left: bool = False,
        right: bool = True,
        which: str = 'SR',
        **kwargs: Any) -> Tuple[np.ndarray, ...]:
    """Non-Hermitian general eigendecomposition dispatch.

    Returns one of:
        (w, vr)            when left=False, right=True   (default)
        (w, vl, vr)        when left=True,  right=True
        (w, vl)            when left=True,  right=False
        (w,)               when left=False, right=False

    Backend selection (v1.7.4)
    --------------------------
    Non-Hermitian eig has NO cuSolverMg multi-GPU equivalent
    (cusolverMgGeev does not exist in the public API).
    Backends:
      - cupy.linalg.eig  (single GPU, full spectrum) when MNPBEM_GPU=1
        and n>=GPU_THRESHOLD.
      - scipy.linalg.eig (LAPACK, full spectrum, supports left+right).
      - scipy.sparse.linalg.eigs (Arnoldi, partial spectrum).

    Partial-spectrum on GPU is emulated by full ``cupy.linalg.eig`` +
    host sort + slice; this is much faster than scipy.sparse.eigs for
    small/medium n on a single GPU, and keeps left/right eigenvectors
    paired with the SAME eigenvalue ordering (matters for biorthogonal
    expansions like plasmonmode).

    Parameters
    ----------
    A : (n, n) ndarray
    k : int, optional
        If given and ``k < n - 1``, return only the k eigenvalues
        selected by ``which``.  Otherwise full spectrum.
    left, right : bool
        Whether to return left / right eigenvectors.  ``cupy.linalg.eig``
        returns right only; if ``left=True`` the multi-output path uses
        scipy LAPACK on the GPU result by recomputing the left set on
        host, OR falls through to CPU scipy.linalg.eig.
    which : str
        Selection criterion for partial spectrum.  'SR'=smallest real,
        'LR'=largest real, 'SM'=smallest magnitude, etc.  Matches
        scipy.sparse.eigs.

    Returns
    -------
    See description above.  All arrays are host NumPy arrays.
    """
    n = A.shape[0]
    want_partial = k is not None and 0 < k < n - 1

    def _select(w: np.ndarray, k_: int) -> np.ndarray:
        if which == 'SR':
            return np.argsort(w.real)[:k_]
        if which == 'LR':
            return np.argsort(w.real)[::-1][:k_]
        if which == 'SM':
            return np.argsort(np.abs(w))[:k_]
        if which == 'LM':
            return np.argsort(np.abs(w))[::-1][:k_]
        if which == 'SI':
            return np.argsort(w.imag)[:k_]
        if which == 'LI':
            return np.argsort(w.imag)[::-1][:k_]
        # default smallest real
        return np.argsort(w.real)[:k_]

    # GPU full-eig path (single device)
    if _CUPY_OK and USE_GPU and n >= GPU_THRESHOLD and not left:
        # cupy.linalg.eig: right eigenvectors only.
        try:
            A_gpu = _cp.asarray(A)
            w_gpu, v_gpu = _cp.linalg.eig(A_gpu)
            w = _cp.asnumpy(w_gpu)
            vr = _cp.asnumpy(v_gpu)
            if want_partial:
                idx = _select(w, int(k))
                w = w[idx]
                vr = vr[:, idx]
            if right:
                return (w, vr)
            return (w,)
        except Exception:
            # Fall through to CPU
            pass

    # CPU path: scipy.linalg.eig handles left=True + right=True with
    # consistent eigenvalue ordering (required by plasmonmode).
    if not want_partial:
        # full spectrum on LAPACK
        from scipy.linalg import eig as _scipy_eig
        kwargs.setdefault("check_finite", False)
        if left and right:
            w, vl, vr = _scipy_eig(A, left=True, right=True, **kwargs)
            return (w, vl, vr)
        if left and not right:
            w, vl = _scipy_eig(A, left=True, right=False, **kwargs)
            return (w, vl)
        if not left and right:
            w, vr = _scipy_eig(A, left=False, right=True, **kwargs)
            return (w, vr)
        w = _scipy_eig(A, left=False, right=False, **kwargs)
        return (w,)

    # Partial-spectrum CPU path
    from scipy.sparse.linalg import eigs as _scipy_eigs
    eigs_kwargs = dict(which=which, maxiter=kwargs.pop('maxiter', 1000))
    if right:
        w_r, vr = _scipy_eigs(A, k=int(k), **eigs_kwargs)
    else:
        w_r = _scipy_eigs(A, k=int(k), return_eigenvectors=False, **eigs_kwargs)
    if left:
        if right:
            w_l, vl = _scipy_eigs(A.T, k=int(k), **eigs_kwargs)
        else:
            w_l, vl = _scipy_eigs(A.T, k=int(k), **eigs_kwargs)
        # NOTE: scipy.sparse.eigs may permute eigenvalues differently for
        # A and A.T; the caller is responsible for re-pairing via overlap.
        if right:
            return (w_r, vl, vr)
        return (w_l, vl)
    if right:
        return (w_r, vr)
    return (w_r,)


# ---------------------------------------------------------------------------
# Multi-GPU Hermitian eigendecomposition via cusolverMgSyevd
# ---------------------------------------------------------------------------
#
# v1.7.4: cusolverMgSyevd IS in the public cusolverMg.h header (unlike
# cusolverMgGemm), so a ctypes binding is safe ABI-wise.  Distribution
# mirrors the LU path: block-cyclic column scatter of A; eigenvalues
# (W) are stored on device 0; result is gathered back to host.
#
# Hermitian only — non-Hermitian general eig has no cusolverMg
# equivalent.  Callers that need a non-Hermitian path must use
# ``eig_dispatch`` (which falls back to single-GPU cupy.linalg.eig
# or CPU scipy.linalg.eig).
# ---------------------------------------------------------------------------


# cusolverEigMode_t
CUSOLVER_EIG_MODE_NOVECTOR = 0
CUSOLVER_EIG_MODE_VECTOR = 1

# cublasFillMode_t
CUBLAS_FILL_MODE_LOWER = 0
CUBLAS_FILL_MODE_UPPER = 1

# cudaDataType
_CUDA_R_32F = 0
_CUDA_R_64F = 1
_CUDA_C_32F = 4
_CUDA_C_64F = 5


class MultiGPUEigh(object):
    """Block-cyclic Hermitian eigendecomposition across multiple GPUs.

    Uses cusolverMgSyevd (single-binding, in-place: A returns eigenvectors,
    W returns ascending real eigenvalues).

    Backend
    -------
    cuSolverMg only.  Falls back to ``RuntimeError`` if the system has
    no libcusolverMg.so or libcudart.so loadable.

    Process-singleton handle
    ------------------------
    ``cusolverMgCreate`` allocates internal state that is NOT fully
    released by ``cusolverMgDestroy`` in some CUDA 12.x builds (every
    second instantiation in the same process fails with status 6
    EXECUTION_FAILED).  We therefore cache the handle and the
    device grid at class level — the first ``eigh()`` call creates them
    and every subsequent call reuses them.  The per-call matrix
    descriptor and device buffers are still allocated and freed each
    time.
    """

    # Class-level singletons.  Initialised lazily.
    _shared_handle = None  # type: Optional[Any]
    _shared_grid = None    # type: Optional[Any]
    _shared_devs = None    # type: Optional[Tuple[int, ...]]
    _peer_access_done = False

    def __init__(self,
            n_gpus: int,
            device_ids: Optional[list] = None) -> None:
        # Lazy import — multi_gpu_lu's module-level _libcusolverMg /
        # _libcudart are mutated by cusolvermg_available() (the first
        # call to it triggers dlopen).  Bind to the module itself so
        # we always read the *current* attribute, not a stale None
        # captured at import time.
        from . import multi_gpu_lu as _mg
        if not _mg.cusolvermg_available():
            raise RuntimeError(
                '[error] cuSolverMg unavailable — '
                '<libcusolverMg.so> or <libcudart.so> not found')
        _mg._bind_cusolvermg()

        # Bind cusolverMgSyevd here (not in _bind_cusolvermg to keep that
        # function focused on LU; ABI of Syevd matches the public header).
        from ctypes import c_int as _c_int, c_int64 as _c_int64
        from ctypes import c_void_p as _c_void_p, POINTER as _PTR
        lib = _mg._libcusolverMg
        if lib is None:
            raise RuntimeError(
                '[error] cuSolverMg lib handle is None despite cusolvermg_available()=True')
        if not getattr(lib, '_mnpbem_syevd_bound', False):
            lib.cusolverMgSyevd_bufferSize.argtypes = [
                _c_void_p, _c_int, _c_int, _c_int,
                _PTR(_c_void_p), _c_int, _c_int, _c_void_p,
                _c_void_p, _c_int, _c_int, _PTR(_c_int64)]
            lib.cusolverMgSyevd_bufferSize.restype = _c_int
            lib.cusolverMgSyevd.argtypes = [
                _c_void_p, _c_int, _c_int, _c_int,
                _PTR(_c_void_p), _c_int, _c_int, _c_void_p,
                _c_void_p, _c_int, _c_int,
                _PTR(_c_void_p), _c_int64, _PTR(_c_int)]
            lib.cusolverMgSyevd.restype = _c_int
            lib._mnpbem_syevd_bound = True

        self.n_gpus = int(n_gpus)
        self.device_ids = (
            list(range(self.n_gpus)) if device_ids is None
            else list(device_ids))
        assert len(self.device_ids) == self.n_gpus, \
            '[error] <device_ids> length must equal <n_gpus>'

        self._lib = lib
        self._rt = _mg._libcudart
        self.handle = None
        self.grid = None
        self.descr = None

    def _real_dtype_for(self, dtype: np.dtype) -> np.dtype:
        # W (eigenvalues) is always real even for complex Hermitian inputs.
        if dtype == np.complex128 or dtype == np.float64:
            return np.dtype(np.float64)
        if dtype == np.complex64 or dtype == np.float32:
            return np.dtype(np.float32)
        raise ValueError(
            '[error] unsupported dtype <{}> for cusolverMgSyevd'.format(dtype))

    def _cuda_dtype_for(self, dtype: np.dtype) -> int:
        if dtype == np.complex128:
            return _CUDA_C_64F
        if dtype == np.complex64:
            return _CUDA_C_32F
        if dtype == np.float64:
            return _CUDA_R_64F
        if dtype == np.float32:
            return _CUDA_R_32F
        raise ValueError(
            '[error] unsupported dtype <{}> for cusolverMgSyevd'.format(dtype))

    def _w_cuda_dtype_for(self, dtype: np.dtype) -> int:
        rdt = self._real_dtype_for(dtype)
        if rdt == np.dtype(np.float64):
            return _CUDA_R_64F
        return _CUDA_R_32F

    def eigh(self, A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute (w, v) of a Hermitian matrix A on multiple GPUs.

        Returns
        -------
        w : (n,) real ndarray, ascending order.
        v : (n, n) ndarray, columns are eigenvectors (in-place
            replacement of A on device; gathered to host).
        """
        from ctypes import (
            c_int as _c_int, c_int32 as _c_int32, c_int64 as _c_int64,
            c_void_p as _c_void_p, POINTER as _PTR, byref as _byref,
        )
        from .multi_gpu_lu import (
            _cuda_set_device, _cuda_malloc, _cuda_free,
            _cuda_memcpy_h2d, _cuda_memcpy_d2h, _cuda_device_sync,
            _check_status, _itemsize_for,
            GRID_MAPPING_COL_MAJOR,
        )

        was_cupy = hasattr(A, 'get') and not isinstance(A, np.ndarray)
        if was_cupy:
            try:
                A = A.get()
            except Exception:
                A = np.asarray(A)
        # Free other cupy pool blocks before allocating multi-GPU buffers
        try:
            _cp.get_default_memory_pool().free_all_blocks()  # type: ignore
            _cp.get_default_pinned_memory_pool().free_all_blocks()  # type: ignore
        except Exception:
            pass

        assert A.ndim == 2 and A.shape[0] == A.shape[1], \
            '[error] cusolverMgSyevd requires square matrix'
        N = A.shape[0]
        dtype = A.dtype
        cuda_dt = self._cuda_dtype_for(dtype)
        w_cuda_dt = self._w_cuda_dtype_for(dtype)
        w_dt = self._real_dtype_for(dtype)
        itemsz = _itemsize_for(dtype)
        w_itemsz = _itemsize_for(w_dt)
        lib = self._lib

        # Optional ABI-debug tracing (set MNPBEM_EIGH_TRACE=1 for stderr
        # log of each cuSolverMg call sequence) — disabled by default.
        _trace = os.environ.get('MNPBEM_EIGH_TRACE', '0') == '1'
        def _t(msg: str) -> None:
            if _trace:
                import sys
                sys.stderr.write('[MultiGPUEigh] ' + msg + '\n')
                sys.stderr.flush()

        # ------------------------------------------------------------------
        # Reuse process-singleton handle + grid (see class docstring).
        # ``cusolverMgCreate`` followed by ``cusolverMgDestroy`` leaves
        # the next ``cusolverMgSyevd`` in status 6 (EXECUTION_FAILED) in
        # CUDA 12.4; caching handle/grid keeps the internal cuBLAS
        # bookkeeping consistent across calls.
        # ------------------------------------------------------------------
        cls = type(self)
        devs_tuple = tuple(self.device_ids)
        if (cls._shared_handle is None
                or cls._shared_devs != devs_tuple):
            # Destroy any prior cached handle/grid (different device list)
            if cls._shared_grid is not None:
                try:
                    lib.cusolverMgDestroyGrid(cls._shared_grid)
                except Exception:
                    pass
                cls._shared_grid = None
            if cls._shared_handle is not None:
                try:
                    lib.cusolverMgDestroy(cls._shared_handle)
                except Exception:
                    pass
                cls._shared_handle = None
                cls._peer_access_done = False

            _t('cusolverMgCreate (cached)')
            h = _c_void_p(0)
            _check_status(lib.cusolverMgCreate(_byref(h)), 'cusolverMgCreate')
            _t('cusolverMgDeviceSelect')
            dev_arr_c = (_c_int * self.n_gpus)(*self.device_ids)
            _check_status(
                lib.cusolverMgDeviceSelect(h, _c_int(self.n_gpus), dev_arr_c),
                'cusolverMgDeviceSelect')

            # Enable peer access — only on first init for this device set.
            # cusolverMgSyevd needs peer access for the distributed
            # tridiagonal reduction stage; absence of peer access surfaces
            # as a segfault inside the cuSolverMg internal copies.  We
            # mirror the NVIDIA cusolver_MgSyevd_example1.cu pattern
            # (enablePeerAccess).
            if not cls._peer_access_done:
                rt = self._rt
                for src in self.device_ids:
                    rt.cudaSetDevice(_c_int(src))
                    for dst in self.device_ids:
                        if src == dst:
                            continue
                        can = _c_int(0)
                        _can_fn = getattr(rt, 'cudaDeviceCanAccessPeer', None)
                        if _can_fn is None:
                            continue
                        _can_fn.argtypes = [_PTR(_c_int), _c_int, _c_int]
                        _can_fn.restype = _c_int
                        _can_fn(_byref(can), _c_int(src), _c_int(dst))
                        if can.value:
                            _en_fn = getattr(rt, 'cudaDeviceEnablePeerAccess', None)
                            if _en_fn is None:
                                continue
                            _en_fn.argtypes = [_c_int, _c_int]
                            _en_fn.restype = _c_int
                            _en_fn(_c_int(dst), _c_int(0))
                cls._peer_access_done = True

            _t('cusolverMgCreateDeviceGrid')
            grid = _c_void_p(0)
            dev_arr32 = (_c_int32 * self.n_gpus)(*self.device_ids)
            _check_status(
                lib.cusolverMgCreateDeviceGrid(
                    _byref(grid), _c_int32(1), _c_int32(self.n_gpus),
                    dev_arr32, _c_int(GRID_MAPPING_COL_MAJOR)),
                'cusolverMgCreateDeviceGrid')

            cls._shared_handle = h
            cls._shared_grid = grid
            cls._shared_devs = devs_tuple
            _t('handle/grid cached')
        else:
            _t('reuse cached handle/grid')

        h = cls._shared_handle
        grid = cls._shared_grid
        # The instance keeps **references** so callers / close() don't try
        # to destroy the shared singletons — close() now only frees the
        # per-call matrix descriptor.
        self.handle = h
        self.grid = grid

        # Block-cyclic distribution params (mirror MultiGPULU)
        blk = int(os.environ.get('MNPBEM_VRAM_SHARE_BLK', '256'))
        max_blk = max(32, ((N // self.n_gpus + 31) // 32) * 32)
        blk = min(blk, max_blk)
        if blk < 32:
            blk = 32

        nblocks = (N + blk - 1) // blk
        max_blocks_per_gpu = (nblocks + self.n_gpus - 1) // self.n_gpus
        local_cols_alloc = max_blocks_per_gpu * blk

        _t('alloc A tiles N={} local_cols_alloc={} blk={}'.format(N, local_cols_alloc, blk))
        # Per-GPU A tiles
        ptrs_A = (_c_void_p * self.n_gpus)()
        for g, dev in enumerate(self.device_ids):
            _cuda_set_device(dev)
            nbytes_A = N * local_cols_alloc * itemsz
            ptrs_A[g] = _c_void_p(_cuda_malloc(nbytes_A))

        # Scatter A (block-cyclic, F-order)
        A_f = np.asfortranarray(A)
        for g, dev in enumerate(self.device_ids):
            _cuda_set_device(dev)
            tile_full = np.zeros((N, local_cols_alloc), dtype=A_f.dtype, order='F')
            local_offset = 0
            for ib in range(nblocks):
                if ib % self.n_gpus != g:
                    continue
                start = ib * blk
                stop = min(N, start + blk)
                ncols = stop - start
                tile_full[:, local_offset:local_offset + ncols] = A_f[:, start:stop]
                local_offset += blk
            _cuda_memcpy_h2d(int(ptrs_A[g] or 0), tile_full)

        # Descriptor for A
        _t('CreateMatrixDesc')
        descr = _c_void_p(0)
        _check_status(
            lib.cusolverMgCreateMatrixDesc(
                _byref(descr), _c_int64(N), _c_int64(N),
                _c_int64(N), _c_int64(blk),
                _c_int(cuda_dt), grid),
            'cusolverMgCreateMatrixDesc(A)')
        self.descr = descr

        # W (eigenvalues) is a HOST array.
        #
        # CRITICAL: cusolverMgSyevd expects ``W`` as a pointer to host
        # memory, NOT device memory.  This is documented in the NVIDIA
        # ``cusolver_MgSyevd_example1.cu`` sample where ``D`` is a
        # ``std::vector<double>`` on host and ``D.data()`` is passed via
        # ``reinterpret_cast<void*>``.  Passing a cudaMalloc-backed
        # device pointer causes cuSolverMg to write into an unmapped
        # host address during the back-transform stage, which surfaces
        # as a segmentation fault.  v1.7.5 fix.
        _t('alloc W on host')
        W_host_buf = np.zeros(N, dtype=w_dt)
        ptr_W = W_host_buf.ctypes.data_as(_c_void_p)

        # Sync every participating device before bufferSize so the
        # H2D scatter completes (mirrors NVIDIA sample cudaDeviceSynchronize
        # after memcpyH2D).
        for dev in self.device_ids:
            _cuda_set_device(dev)
            _cuda_device_sync()

        # Query workspace size
        _t('Syevd_bufferSize')
        lwork = _c_int64(0)
        _check_status(
            lib.cusolverMgSyevd_bufferSize(
                h,
                _c_int(CUSOLVER_EIG_MODE_VECTOR),
                _c_int(CUBLAS_FILL_MODE_LOWER),
                _c_int(N), ptrs_A,
                _c_int(1), _c_int(1), descr,
                ptr_W, _c_int(w_cuda_dt), _c_int(cuda_dt),
                _byref(lwork)),
            'cusolverMgSyevd_bufferSize')
        work_lwork = int(lwork.value)
        _t('work_lwork={}'.format(work_lwork))

        # Workspace per device — ``lwork`` is **number of elements** per
        # device (matching the NVIDIA sample's ``sizeof(data_type) *
        # lwork`` allocation).  Use the **compute** dtype itemsize, not
        # the W (real) dtype, since the workspace stores complex
        # off-diagonal and Householder tau elements for the complex
        # path.
        ptrs_w = (_c_void_p * self.n_gpus)()
        for g, dev in enumerate(self.device_ids):
            _cuda_set_device(dev)
            ptrs_w[g] = _c_void_p(_cuda_malloc(max(1, work_lwork) * itemsz))

        # Sync after workspace allocation
        for dev in self.device_ids:
            _cuda_set_device(dev)
            _cuda_device_sync()

        # Run Syevd
        _t('Syevd call')
        info = _c_int(0)
        _check_status(
            lib.cusolverMgSyevd(
                h,
                _c_int(CUSOLVER_EIG_MODE_VECTOR),
                _c_int(CUBLAS_FILL_MODE_LOWER),
                _c_int(N), ptrs_A,
                _c_int(1), _c_int(1), descr,
                ptr_W, _c_int(w_cuda_dt), _c_int(cuda_dt),
                ptrs_w, _c_int64(work_lwork), _byref(info)),
            'cusolverMgSyevd')
        if info.value != 0:
            # Free intermediates before raising
            for g, dev in enumerate(self.device_ids):
                try:
                    _cuda_set_device(dev)
                    _cuda_free(int(ptrs_w[g] or 0))
                except Exception:
                    pass
            for g, dev in enumerate(self.device_ids):
                try:
                    _cuda_set_device(dev)
                    _cuda_free(int(ptrs_A[g] or 0))
                except Exception:
                    pass
            raise RuntimeError(
                '[error] cusolverMgSyevd reports info={}'.format(info.value))

        # Sync after Syevd before reading W back / D2H copy
        for dev in self.device_ids:
            _cuda_set_device(dev)
            _cuda_device_sync()

        # W is already on host (we passed a host pointer to Syevd).
        W_host = W_host_buf

        # Gather A (= eigenvectors V) back to host, block-cyclic.
        V_f = np.empty((N, N), dtype=dtype, order='F')
        for g, dev in enumerate(self.device_ids):
            _cuda_set_device(dev)
            tile_full = np.empty((N, local_cols_alloc), dtype=dtype, order='F')
            nbytes = N * local_cols_alloc * itemsz
            _cuda_memcpy_d2h(tile_full, int(ptrs_A[g] or 0), nbytes)
            local_offset = 0
            for ib in range(nblocks):
                if ib % self.n_gpus != g:
                    continue
                start = ib * blk
                stop = min(N, start + blk)
                ncols = stop - start
                V_f[:, start:stop] = tile_full[:, local_offset:local_offset + ncols]
                local_offset += blk

        # Free intermediates — W lives in NumPy host memory now, so no
        # cudaFree for it (will be reclaimed when ``W_host_buf`` is
        # garbage-collected).
        for g, dev in enumerate(self.device_ids):
            _cuda_set_device(dev)
            _cuda_free(int(ptrs_w[g] or 0))
            _cuda_free(int(ptrs_A[g] or 0))
        return W_host, np.ascontiguousarray(V_f)

    def close(self) -> None:
        lib = self._lib
        if self.descr is not None:
            try:
                lib.cusolverMgDestroyMatrixDesc(self.descr)
            except Exception:
                pass
            self.descr = None
        # NOTE: ``self.grid`` and ``self.handle`` reference the class-level
        # singleton entries — DO NOT destroy them on instance close, or
        # the next ``eigh()`` call will run on an invalidated handle.
        # The singletons are released only when the class explicitly
        # invokes :meth:`shutdown_shared` at interpreter exit (best
        # effort).
        self.grid = None
        self.handle = None

    @classmethod
    def shutdown_shared(cls) -> None:
        """Destroy the cached cusolverMg handle / grid at interpreter exit.

        Safe to call multiple times.  After this call the next
        :meth:`eigh` will re-create fresh handle/grid.
        """
        try:
            from . import multi_gpu_lu as _mg
            lib = _mg._libcusolverMg
            if lib is None:
                return
            if cls._shared_grid is not None:
                try:
                    lib.cusolverMgDestroyGrid(cls._shared_grid)
                except Exception:
                    pass
                cls._shared_grid = None
            if cls._shared_handle is not None:
                try:
                    lib.cusolverMgDestroy(cls._shared_handle)
                except Exception:
                    pass
                cls._shared_handle = None
            cls._shared_devs = None
            cls._peer_access_done = False
        except Exception:
            pass

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def matmul_dispatch(A: np.ndarray, B: np.ndarray, **kwargs: Any) -> np.ndarray:
    """Dense matrix product on GPU when beneficial, else CPU.

    Used by field-application code paths where a single big dense GEMM
    dominates and the inputs are still on the host.

    Multi-GPU VRAM-share path
    -------------------------
    Pass ``n_gpus=N`` (with ``N>=2``) to distribute ``C = A @ B`` across
    multiple GPUs via a column-split strategy (cupy + per-device
    streams). ``cusolverMgGemm`` exists in the library binary but its
    public header does not declare it, so we deliberately use a
    cupy-based fallback instead of binding an undocumented ABI. See
    ``multi_gpu_lu.matmul_multi_gpu`` for the implementation notes.

    Env-var auto-wiring
    -------------------
    When ``n_gpus`` is omitted, ``MNPBEM_VRAM_SHARE_GPUS`` (>=2) is read
    automatically along with ``MNPBEM_VRAM_SHARE_DEVICE_IDS``. Explicit
    kwargs win over env defaults. Backend is fixed to the column-split
    cupy path regardless of ``MNPBEM_VRAM_SHARE_BACKEND`` because
    cusolverMgGemm is not part of the public API.
    """
    env_n, env_backend, env_devs = _vram_share_env_defaults()
    n_gpus = int(kwargs.pop('n_gpus', env_n if env_n is not None else 1))
    backend = kwargs.pop('backend', env_backend if env_backend is not None else 'cusolvermg')
    device_ids = kwargs.pop('device_ids', env_devs)
    if n_gpus >= 2:
        try:
            from .multi_gpu_lu import matmul_multi_gpu, warn_fallback
            # Backend note: cusolverMgGemm is undocumented (header
            # omits it); we honour the env var only as a label and
            # always use the cupy column-split implementation.
            if backend not in ('cusolvermg', 'cupy_split'):
                warn_fallback(
                    'matmul backend <{}> not supported; using cupy_split'.format(backend))
            return matmul_multi_gpu(A, B, n_gpus=n_gpus, device_ids=device_ids)
        except (RuntimeError, NotImplementedError, ValueError) as exc:
            from .multi_gpu_lu import warn_fallback
            warn_fallback(repr(exc))
    if _CUPY_OK and USE_GPU and A.shape[0] >= GPU_THRESHOLD:
        A_gpu = _cp.asarray(A)
        B_gpu = _cp.asarray(B)
        C_gpu = A_gpu @ B_gpu
        return _cp.asnumpy(C_gpu)
    return A @ B


# ---------------------------------------------------------------------------
# Lane C — layer-Green / Sommerfeld integral GPU helpers
#
# These do NOT participate in the BLAS-level dispatch; they are designed for
# the elementwise-heavy kernels (outer products, propagation factors, weighted
# sum reductions) that dominate ``_intbessel_batch`` / ``_inthankel_batch`` in
# layer_structure.py.  Activation is tied to ``MNPBEM_GPU=1`` AND a separate
# ``MNPBEM_GPU_LAYER`` flag so the BEM-solver dispatch above can be tuned
# independently of the layer-Green path.
# ---------------------------------------------------------------------------

LAYER_GPU: bool = (
    USE_GPU and os.environ.get("MNPBEM_GPU_LAYER", "1").strip() not in ("", "0", "false", "False")
)
LAYER_GPU_THRESHOLD: int = int(os.environ.get("MNPBEM_GPU_LAYER_THRESHOLD", "5000"))


def layer_gpu_available() -> bool:
    return _CUPY_OK and LAYER_GPU


def layer_gpu_active(n_flat: int) -> bool:
    """Decide whether to route a layer-Green elementwise kernel to GPU.

    ``n_flat`` is the size of the flattened (n1*n2) array.  The host->device
    copy and kernel launch overhead make GPU profitable only above a few
    thousand entries; below that NumPy wins.
    """
    return _CUPY_OK and LAYER_GPU and n_flat >= LAYER_GPU_THRESHOLD


def get_layer_xp(n_flat: int):
    """Return (xp, asnumpy, on_gpu) for layer-Green elementwise kernels.

    ``xp`` is either ``cupy`` or ``numpy``; ``asnumpy(arr)`` materializes
    arrays on the host; ``on_gpu`` is a boolean for the active backend.
    """
    if layer_gpu_active(n_flat):
        return _cp, _cp.asnumpy, True
    return np, (lambda a: np.asarray(a)), False
