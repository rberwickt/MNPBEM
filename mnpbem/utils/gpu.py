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
    """
    n_gpus = int(kwargs.pop('n_gpus', 1))
    backend = kwargs.pop('backend', 'cusolvermg')
    device_ids = kwargs.pop('device_ids', None)
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
    """
    tag = piv_pkg[0]
    if tag == "mgpu":
        lu_handle = piv_pkg[1]
        trans = kwargs.pop('trans', 'N')
        if isinstance(trans, int):
            trans = {0: 'N', 1: 'T', 2: 'C'}.get(trans, 'N')
        b_host = b
        if _CUPY_OK and isinstance(b, _cp.ndarray):
            b_host = _cp.asnumpy(b)
        return lu_handle.solve(np.ascontiguousarray(b_host), trans=trans)
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
    """
    if _CUPY_OK and USE_GPU and A.shape[0] >= GPU_THRESHOLD:
        A_gpu = _cp.asarray(A)
        b_gpu = _cp.asarray(b)
        x_gpu = _cp.linalg.solve(A_gpu, b_gpu)
        return _cp.asnumpy(x_gpu)
    kwargs.setdefault("check_finite", False)
    return _scipy_solve(A, b, **kwargs)


def eigh_dispatch(A: np.ndarray, **kwargs: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Hermitian eigendecomposition on GPU when beneficial, else CPU.

    Returns ``(w, v)`` as host NumPy arrays regardless of backend.  Routes
    to ``cupy.linalg.eigh`` when GPU is enabled and the matrix is at least
    ``GPU_THRESHOLD`` rows.
    """
    if _CUPY_OK and USE_GPU and A.shape[0] >= GPU_THRESHOLD:
        A_gpu = _cp.asarray(A)
        w_gpu, v_gpu = _cp.linalg.eigh(A_gpu)
        return _cp.asnumpy(w_gpu), _cp.asnumpy(v_gpu)
    from scipy.linalg import eigh as _scipy_eigh
    kwargs.setdefault("check_finite", False)
    return _scipy_eigh(A, **kwargs)


def matmul_dispatch(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Dense matrix product on GPU when beneficial, else CPU.

    Used by field-application code paths where a single big dense GEMM
    dominates and the inputs are still on the host.
    """
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
