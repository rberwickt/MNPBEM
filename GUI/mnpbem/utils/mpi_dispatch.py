"""Multi-node MPI wavelength-batched BEM dispatch.

Strategy
--------
Wavelengths in a spectrum sweep are embarrassingly parallel.  Where
``solve_spectrum_multi_gpu`` distributes them across the GPUs of a single
machine, ``solve_spectrum_mpi`` adds a second axis: it splits the
wavelengths across MPI ranks (typically one per HPC node), and each rank
then uses ``solve_spectrum_multi_gpu`` to feed its local GPUs.

So with 2 nodes x 4 GPUs each you get 8-way wavelength parallelism with
no extra coordination beyond a single ``gather`` of the per-rank results.

Per-rank flow
-------------
1. Read its rank / size from ``MPI.COMM_WORLD``.
2. ``np.array_split`` the wavelength index range across all ranks.
3. Hand its slice to ``solve_spectrum_multi_gpu`` (or a serial CPU loop
   if no GPU is visible to this rank).
4. ``comm.gather`` the per-rank ``ext`` / ``sca`` arrays to rank 0.

Rank 0 reassembles the full spectrum in original wavelength order and
returns the merged result; non-zero ranks return ``None``.

Fallbacks
---------
* ``mpi4py`` not importable        -> single-process call to
  ``solve_spectrum_multi_gpu`` (or CPU serial if no GPU).
* ``MPI.COMM_WORLD.Get_size() == 1`` -> same single-process path.
* No GPU visible to a rank          -> that rank runs a serial CPU loop.

Usage
-----
With SLURM (recommended)::

    #!/bin/bash
    #SBATCH --nodes=2
    #SBATCH --ntasks-per-node=1
    #SBATCH --gres=gpu:4
    srun python my_script.py

In ``my_script.py``::

    from mnpbem.utils.mpi_dispatch import solve_spectrum_mpi
    result = solve_spectrum_mpi(
        particle_factory=lambda: build_dimer(),
        enei=np.linspace(500, 1000, 100),
        pol_dirs=[[1, 0, 0], [0, 1, 0]],
        prop_dirs=[[0, 0, 1], [0, 0, 1]],
    )
    if result is not None:           # rank 0 only
        save_spectrum(result)

Notes
-----
* ``mpi4py`` is an *optional* dependency.  If it is not installed the
  dispatcher silently falls back to single-node behaviour.
* Each rank rebuilds the particle from ``particle_factory`` -- same
  rationale as ``solve_spectrum_multi_gpu``: cupy/Numba caches are bound
  to the local CUDA context.
* GPU isolation across ranks on the same node is the launcher's job
  (SLURM ``--gres=gpu:N`` sets ``CUDA_VISIBLE_DEVICES`` automatically).
* The result dict on rank 0 mirrors ``solve_spectrum_multi_gpu``'s shape
  but adds ``per_rank_s`` and ``n_ranks`` for diagnostics.
"""

from __future__ import annotations

import os
import sys
import time
import warnings
from typing import Any, Callable, Optional, Sequence

import numpy as np

from .multi_gpu import _detect_gpu_count, solve_spectrum_multi_gpu


def _mpi4py_available() -> bool:
    try:
        import mpi4py  # noqa: F401
        return True
    except Exception:
        return False


def _solve_chunk_cpu(
        particle_factory: Callable[[], Any],
        enei_chunk: Sequence[float],
        pol_dirs: Sequence[Sequence[float]],
        prop_dirs: Sequence[Sequence[float]],
        bem_kwargs: dict) -> dict:
    """Serial CPU loop over a wavelength chunk (no GPU visible)."""
    os.environ.setdefault('MNPBEM_NUMBA', '1')
    from mnpbem.bem import BEMRet
    from mnpbem.simulation import PlaneWaveRet

    p = particle_factory()
    bem = BEMRet(p, **bem_kwargs)
    exc = PlaneWaveRet(list(pol_dirs), list(prop_dirs))

    n_pol = len(pol_dirs)
    enei_chunk = list(enei_chunk)
    ext = np.zeros((len(enei_chunk), n_pol))
    sca = np.zeros((len(enei_chunk), n_pol))

    if not enei_chunk:
        return {'ext': ext, 'sca': sca, 'wall_s': 0.0}

    sig, bem = bem.solve(exc(p, enei_chunk[0]))  # warmup

    t0 = time.time()
    for k, e in enumerate(enei_chunk):
        sig, bem = bem.solve(exc(p, e))
        ev = np.asarray(exc.extinction(sig)).real.flatten()
        sv_raw = exc.scattering(sig)
        sv = np.asarray(sv_raw[0] if isinstance(sv_raw, tuple) else sv_raw).real.flatten()
        ext[k] = ev[:n_pol]
        sca[k] = sv[:n_pol]

    return {'ext': ext, 'sca': sca, 'wall_s': time.time() - t0}


def solve_spectrum_mpi(
        particle_factory: Callable[[], Any],
        enei: Sequence[float],
        pol_dirs: Sequence[Sequence[float]],
        prop_dirs: Sequence[Sequence[float]],
        n_gpus_per_node: Optional[int] = None,
        bem_kwargs: Optional[dict] = None) -> Optional[dict]:
    """Solve a wavelength spectrum across MPI ranks (multi-node).

    Parameters
    ----------
    particle_factory : callable
        Zero-arg callable returning a freshly-built ``ComParticle``.
    enei : sequence of float
        Wavelengths in nm.
    pol_dirs, prop_dirs : sequence of 3-vectors
        Polarization / propagation directions.
    n_gpus_per_node : int, optional
        GPU count to use *per rank*.  Defaults to the GPUs visible to
        each rank (typically what SLURM/PBS allocated to the node).
        Set 0 to force a CPU fallback even when GPUs are present.
    bem_kwargs : dict, optional
        Extra kwargs forwarded to ``BEMRet`` (e.g. ``hmode='dense'``).

    Returns
    -------
    dict on rank 0 with:
        'ext'         : (n_lambda, n_pol) extinction
        'sca'         : (n_lambda, n_pol) scattering
        'enei'        : 1-D wavelength array (echoed)
        'wall_s'      : slowest rank wall (== overall wall)
        'per_rank_s'  : list of per-rank wall times
        'n_ranks'     : MPI world size
        'n_gpus_per_node' : GPUs each rank used (0 = CPU fallback)
    None on non-zero ranks.

    If ``mpi4py`` is unavailable or ``MPI.COMM_WORLD.Get_size() == 1``
    the call falls back to ``solve_spectrum_multi_gpu`` and returns the
    same dict shape (with ``per_rank_s`` containing one entry).
    """
    if bem_kwargs is None:
        bem_kwargs = {}
    enei = np.asarray(enei, dtype=float)

    if not _mpi4py_available():
        # No MPI runtime available -> single-node path.
        return _single_node_fallback(
            particle_factory, enei, pol_dirs, prop_dirs,
            n_gpus_per_node, bem_kwargs,
            reason='mpi4py not installed')

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if size == 1:
        return _single_node_fallback(
            particle_factory, enei, pol_dirs, prop_dirs,
            n_gpus_per_node, bem_kwargs,
            reason='single MPI rank')

    if n_gpus_per_node is None:
        n_gpus_per_node = _detect_gpu_count()

    # Partition wavelength indices evenly across ranks.
    chunks = np.array_split(np.arange(len(enei)), size)
    my_idx = chunks[rank]
    my_enei = enei[my_idx]

    # Each rank solves its slice with whatever local resources it has.
    if n_gpus_per_node >= 1 and len(my_enei) > 0:
        local = solve_spectrum_multi_gpu(
            particle_factory=particle_factory,
            enei=my_enei,
            pol_dirs=pol_dirs,
            prop_dirs=prop_dirs,
            n_gpus=n_gpus_per_node,
            bem_kwargs=bem_kwargs,
        )
        local_payload = {
            'wl_indices': my_idx.tolist(),
            'ext': local['ext'],
            'sca': local['sca'],
            'wall_s': local['wall_s'],
        }
    else:
        local = _solve_chunk_cpu(
            particle_factory, my_enei, pol_dirs, prop_dirs, bem_kwargs)
        local_payload = {
            'wl_indices': my_idx.tolist(),
            'ext': local['ext'],
            'sca': local['sca'],
            'wall_s': local['wall_s'],
        }

    # Gather on rank 0.
    all_payloads = comm.gather(local_payload, root=0)

    if rank != 0:
        return None

    n_pol = len(pol_dirs)
    ext = np.zeros((len(enei), n_pol))
    sca = np.zeros((len(enei), n_pol))
    per_rank_s = []
    for payload in all_payloads:
        idxs = payload['wl_indices']
        ext[idxs] = payload['ext']
        sca[idxs] = payload['sca']
        per_rank_s.append(payload['wall_s'])

    return {
        'ext': ext,
        'sca': sca,
        'enei': enei,
        'wall_s': float(max(per_rank_s)) if per_rank_s else 0.0,
        'per_rank_s': per_rank_s,
        'n_ranks': size,
        'n_gpus_per_node': n_gpus_per_node,
    }


def _single_node_fallback(particle_factory, enei, pol_dirs, prop_dirs,
                          n_gpus_per_node, bem_kwargs, reason: str) -> dict:
    """Return the same dict shape as solve_spectrum_mpi but on one host."""
    if n_gpus_per_node is None:
        n_gpus_per_node = _detect_gpu_count()

    if n_gpus_per_node >= 1:
        result = solve_spectrum_multi_gpu(
            particle_factory=particle_factory,
            enei=enei,
            pol_dirs=pol_dirs,
            prop_dirs=prop_dirs,
            n_gpus=n_gpus_per_node,
            bem_kwargs=bem_kwargs,
        )
        wall = result['wall_s']
        ext = result['ext']
        sca = result['sca']
    else:
        warnings.warn(
            'solve_spectrum_mpi falling back to serial CPU '
            f'({reason}; no GPU detected)',
            stacklevel=2,
        )
        result = _solve_chunk_cpu(
            particle_factory, list(enei), pol_dirs, prop_dirs, bem_kwargs)
        wall = result['wall_s']
        ext = result['ext']
        sca = result['sca']

    return {
        'ext': ext,
        'sca': sca,
        'enei': enei,
        'wall_s': wall,
        'per_rank_s': [wall],
        'n_ranks': 1,
        'n_gpus_per_node': n_gpus_per_node,
    }
