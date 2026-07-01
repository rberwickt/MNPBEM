"""Multi-GPU wavelength-batched BEM dispatch (Lane D, M4 GPU Phase 2).

Strategy
--------
Wavelengths in a spectrum sweep (e.g. 100 lambdas in 500-1000 nm) are
embarrassingly parallel for the BEMRet solve: each lambda independently
builds its own G/H matrices and solves Ax=b, with no cross-lambda
coupling.  We therefore split the wavelength array across the available
GPUs and run a worker process per GPU.

Each worker:
  1. binds itself to one CUDA device via CUDA_VISIBLE_DEVICES,
  2. constructs its own ``BEMRet`` solver (independent host RAM),
  3. iterates over its assigned wavelengths,
  4. computes extinction / scattering for each polarization,
  5. ships the resulting NumPy arrays back through a multiprocessing
     queue.

The driver function ``solve_spectrum_multi_gpu`` returns the merged
arrays in the original wavelength order so the caller can drop in the
result without any per-lambda glue.

Usage
-----
>>> from mnpbem.utils.multi_gpu import solve_spectrum_multi_gpu
>>> ext_x, sca_x, ext_y, sca_y = solve_spectrum_multi_gpu(
...     particle_factory=lambda: build_dimer(),
...     enei=np.linspace(500, 1000, 100),
...     pol_dirs=[[1,0,0], [0,1,0]],
...     prop_dirs=[[0,0,1], [0,0,1]],
...     n_gpus=4,
... )

Notes
-----
* The particle and excitation are *re-built* inside each worker because
  cupy / Numba caches are bound to the parent CUDA context.  We pass
  factory callables instead of pickled objects to keep the API ergonomic
  even when the geometry construction itself is not picklable.
* Workers print progress via ``sys.stdout`` prefixed with ``[gpu=<idx>]``.
* If ``n_gpus`` exceeds the number of physical GPUs detected, we cap it.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import sys
import time
from typing import Any, Callable, List, Optional, Sequence, Tuple

import numpy as np


def _detect_gpu_count() -> int:
    try:
        import cupy as cp  # type: ignore
        return int(cp.cuda.runtime.getDeviceCount())
    except Exception:
        return 0


def _resolve_bem_class(bem_class_name: Optional[str]) -> Any:
    # Bug 4 fix: workers run in spawn-ed processes so we cannot pickle a
    # class reference reliably across all setups.  Pass the BEM solver as
    # a string ('BEMRet', 'BEMRetIter', 'BEMRetLayer', 'BEMRetLayerIter')
    # and resolve it inside the worker after MNPBEM_GPU has been set.
    from mnpbem import bem as _bem_pkg

    if bem_class_name is None or bem_class_name == 'BEMRet':
        return _bem_pkg.BEMRet
    if bem_class_name == 'BEMRetIter':
        return _bem_pkg.BEMRetIter
    if bem_class_name == 'BEMRetLayer':
        return _bem_pkg.BEMRetLayer
    if bem_class_name == 'BEMRetLayerIter':
        return _bem_pkg.BEMRetLayerIter
    raise ValueError('[error] Unsupported <bem_class>: {}'.format(bem_class_name))


def _worker(gpu_idx: int,
            wl_indices: List[int],
            enei_chunk: List[float],
            particle_factory: Callable[[], Any],
            pol_dirs: Sequence[Sequence[float]],
            prop_dirs: Sequence[Sequence[float]],
            queue: 'mp.Queue',
            bem_kwargs: dict,
            bem_class_name: Optional[str] = None) -> None:
    """Worker process: solve one wavelength chunk on a single GPU."""
    # Bind CUDA device.  Must happen before any cupy import is triggered
    # transitively by mnpbem.* .
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_idx)
    os.environ.setdefault('MNPBEM_GPU', '1')
    os.environ.setdefault('MNPBEM_NUMBA', '1')

    try:
        from mnpbem.simulation import PlaneWaveRet

        BEMClass = _resolve_bem_class(bem_class_name)

        p = particle_factory()
        bem = BEMClass(p, **bem_kwargs)
        exc = PlaneWaveRet(list(pol_dirs), list(prop_dirs))

        # Warmup at first lambda so cupy/numba kernel cache is warm before
        # we time the loop.
        if enei_chunk:
            sig, bem = bem.solve(exc(p, enei_chunk[0]))

        n_pol = len(pol_dirs)
        ext = np.zeros((len(enei_chunk), n_pol))
        sca = np.zeros((len(enei_chunk), n_pol))

        t0 = time.time()
        for k, e in enumerate(enei_chunk):
            sig, bem = bem.solve(exc(p, e))
            ev = np.asarray(exc.extinction(sig)).real.flatten()
            sv_raw = exc.scattering(sig)
            sv = np.asarray(sv_raw[0] if isinstance(sv_raw, tuple) else sv_raw).real.flatten()
            ext[k] = ev[:n_pol]
            sca[k] = sv[:n_pol]
            if (k + 1) % max(1, len(enei_chunk) // 4) == 0:
                pct = 100.0 * (k + 1) / max(1, len(enei_chunk))
                print(f'[gpu={gpu_idx}] {k + 1}/{len(enei_chunk)}  '
                      f'({pct:.0f}%)  elapsed={(time.time() - t0) / 60:.1f}min',
                      flush=True)

        queue.put({
            'gpu_idx': gpu_idx,
            'wl_indices': wl_indices,
            'ext': ext,
            'sca': sca,
            'wall_s': time.time() - t0,
            'ok': True,
        })
    except Exception as exc:  # pragma: no cover - best-effort error path
        import traceback
        queue.put({
            'gpu_idx': gpu_idx,
            'wl_indices': wl_indices,
            'ok': False,
            'error': repr(exc),
            'traceback': traceback.format_exc(),
        })


def solve_spectrum_multi_gpu(
        particle_factory: Callable[[], Any],
        enei: Sequence[float],
        pol_dirs: Sequence[Sequence[float]],
        prop_dirs: Sequence[Sequence[float]],
        n_gpus: Optional[int] = None,
        bem_kwargs: Optional[dict] = None,
        bem_class: Optional[Any] = None) -> dict:
    """Solve a wavelength spectrum across multiple GPUs.

    Parameters
    ----------
    particle_factory : callable
        Zero-arg callable that returns a freshly-constructed
        ``ComParticle``.  Called once inside each worker.
    enei : sequence of float
        Wavelengths in nm.
    pol_dirs, prop_dirs : sequence of 3-vectors
        Polarization / propagation directions for ``PlaneWaveRet``.
    n_gpus : int, optional
        Worker count.  Defaults to the detected GPU count, capped at 4.
    bem_kwargs : dict, optional
        Extra keyword arguments forwarded to the BEM solver (e.g. hmode).
    bem_class : class or str, optional
        BEM solver class to use inside each worker.  Accepts the class
        object (e.g. ``mnpbem.bem.BEMRetIter``) or its name as a string.
        Defaults to ``BEMRet``.  Bug 4 fix: previously hard-coded so
        ``simulation.type=ret_iter`` was silently ignored.

    Returns
    -------
    dict with keys:
        'ext'        : (n_lambda, n_pol) extinction
        'sca'        : (n_lambda, n_pol) scattering
        'enei'       : 1-D wavelength array (echoed)
        'wall_s'     : wall time of the slowest worker (== overall wall)
        'per_gpu_s'  : list of per-worker wall times
    """
    if bem_kwargs is None:
        bem_kwargs = {}
    enei = np.asarray(enei, dtype=float)

    # Bug 4 fix: accept either a class object or its bare name.  We pass
    # only the name to the spawn-ed worker so the parent module does not
    # need to be importable / picklable in every environment.
    if bem_class is None:
        bem_class_name = None
    elif isinstance(bem_class, str):
        bem_class_name = bem_class
    else:
        bem_class_name = bem_class.__name__

    if n_gpus is None:
        n_gpus = _detect_gpu_count()
    n_gpus = max(1, min(n_gpus, len(enei)))

    # Split wavelengths into n_gpus chunks of (almost) equal size.
    chunks: List[List[int]] = [[] for _ in range(n_gpus)]
    for i in range(len(enei)):
        chunks[i % n_gpus].append(i)

    n_pol = len(pol_dirs)
    ctx = mp.get_context('spawn')
    queue = ctx.Queue()

    procs = []
    t0 = time.time()
    for g in range(n_gpus):
        wl_indices = chunks[g]
        enei_chunk = enei[wl_indices].tolist()
        if not wl_indices:
            continue
        p = ctx.Process(
            target=_worker,
            args=(g, wl_indices, enei_chunk, particle_factory,
                  list(pol_dirs), list(prop_dirs), queue, dict(bem_kwargs),
                  bem_class_name),
        )
        p.start()
        procs.append(p)

    # Collect results.
    ext = np.zeros((len(enei), n_pol))
    sca = np.zeros((len(enei), n_pol))
    per_gpu_s: List[float] = []
    errors = []
    for _ in procs:
        r = queue.get()
        if not r.get('ok', False):
            errors.append(r)
            continue
        idxs = r['wl_indices']
        ext[idxs] = r['ext']
        sca[idxs] = r['sca']
        per_gpu_s.append(r.get('wall_s', 0.0))

    for p in procs:
        p.join(timeout=10)
    wall = time.time() - t0

    if errors:
        raise RuntimeError(
            'Multi-GPU spectrum solve failed for {} workers: {}'.format(
                len(errors), errors[0]))

    return {
        'ext': ext,
        'sca': sca,
        'enei': enei,
        'wall_s': wall,
        'per_gpu_s': per_gpu_s,
        'n_gpus': len(per_gpu_s),
    }
