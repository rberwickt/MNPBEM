"""
Example 07 — GPU-accelerated and multi-GPU wavelength dispatch.

This example shows two opt-in performance paths:

  1. ``MNPBEM_GPU=1`` environment variable: routes the dense BEMRet
     matrix assembly through cupy when running. The Python API does
     not change.

  2. ``solve_spectrum_multi_gpu``: splits a wavelength sweep across
     several GPUs (one worker process per GPU). Each worker rebuilds
     the particle and excitation, solves its own chunk, and the
     driver merges the result.

Both paths require ``cupy`` and (for the multi-GPU path) at least one
CUDA-capable GPU. If neither is available the script falls back to
the CPU dense solver — the only difference is wall time.

Run:
    # CPU only (always works)
    python examples/07_gpu_multigpu.py

    # GPU on / multi-GPU on (requires cupy + GPUs)
    MNPBEM_GPU=1 python examples/07_gpu_multigpu.py
"""

import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))


def have_cupy():
    try:
        import cupy as cp                                                # noqa
        return True
    except Exception:
        return False


def gpu_count():
    if not have_cupy():
        return 0
    try:
        import cupy as cp
        return int(cp.cuda.runtime.getDeviceCount())
    except Exception:
        return 0


def build_particle():
    """Factory used by the multi-GPU dispatcher (must be picklable in
    spawned subprocesses)."""
    from mnpbem.materials import EpsConst, EpsTable
    from mnpbem.geometry import trisphere, ComParticle
    epstab = [EpsConst(1.0), EpsTable("gold.dat")]
    return ComParticle(epstab, [trisphere(144, 20.0)],
                       [[2, 1]], 1, interp="curv")


def run_cpu(enei):
    from mnpbem.bem import BEMRet
    from mnpbem.simulation import PlaneWaveRet
    p = build_particle()
    bem = BEMRet(p)
    exc = PlaneWaveRet(np.array([[1.0, 0.0, 0.0]]),
                       np.array([[0.0, 0.0, 1.0]]))
    n = len(enei)
    ext = np.zeros(n)
    t0 = time.time()
    for i, e in enumerate(enei):
        sig, bem = bem.solve(exc.potential(p, e))
        ext[i] = float(np.real(np.ravel(exc.extinction(sig))[0]))
    return ext, time.time() - t0


def run_multigpu(enei, n_gpus):
    from mnpbem.utils.multi_gpu import solve_spectrum_multi_gpu
    res = solve_spectrum_multi_gpu(
        particle_factory=build_particle,
        enei=enei,
        pol_dirs=[[1.0, 0.0, 0.0]],
        prop_dirs=[[0.0, 0.0, 1.0]],
        n_gpus=n_gpus,
    )
    return res["ext"][:, 0], res["wall_s"]


def main():
    out_png = os.path.join(HERE, "07_gpu_multigpu.png")
    enei = np.linspace(450.0, 700.0, 9)

    n_gpu = gpu_count()
    print("[07] GPUs detected: {}".format(n_gpu))

    ext_cpu, t_cpu = run_cpu(enei)
    print("[07] CPU dense  : {:.2f} s for {} wls".format(t_cpu, len(enei)))

    if n_gpu >= 1:
        ext_gpu, t_gpu = run_multigpu(enei, n_gpus=min(n_gpu, 4))
        speedup = t_cpu / t_gpu if t_gpu > 0 else float("inf")
        print("[07] {}-GPU dense: {:.2f} s ({:.2f}x vs CPU)"
              .format(min(n_gpu, 4), t_gpu, speedup))
        rel = np.max(np.abs(ext_gpu - ext_cpu) / np.abs(ext_cpu))
        print("[07] max relative diff GPU vs CPU: {:.2e}".format(rel))
    else:
        print("[07] no GPU detected, plotting CPU-only spectrum")
        ext_gpu = None

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(enei, ext_cpu, "b-", lw=2, label="CPU dense")
    if ext_gpu is not None:
        ax.plot(enei, ext_gpu, "ro--", lw=1.5, label="multi-GPU dense")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Extinction (nm^2)")
    ax.set_title("Au sphere d=30 nm, GPU dispatch")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_png, dpi=140)
    print("[07] saved -> {}".format(out_png))


if __name__ == "__main__":
    main()
