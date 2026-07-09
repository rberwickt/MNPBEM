"""
Example 06 — Iterative ACA / GMRES solver vs dense solver.

Equivalent MATLAB demo: Demo/iter/demoiter1.m (and friends).

For meshes with more than a few thousand faces the dense solver
factorizes a full block-dense BEM matrix every wavelength, which
quickly becomes memory-bound. The iterative solver `BEMRetIter`:

  - compresses the off-diagonal Green-function blocks with ACA,
  - solves the system with GMRES,
  - reuses the H-matrix factor across wavelengths.

This example runs the same gold sphere as `01_sphere_extinction.py`
through both `BEMRet` and `BEMRetIter`, prints wall times, and overlays
the two extinction spectra.

Run:
    python examples/06_iterative_aca.py
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

from mnpbem.materials import EpsConst, EpsTable
from mnpbem.geometry import trisphere, ComParticle
from mnpbem.bem import BEMRet, BEMRetIter
from mnpbem.simulation import PlaneWaveRet


def run(bem_cls, p, exc, enei, label):
    bem = bem_cls(p)
    n = len(enei)
    ext = np.zeros(n)
    t0 = time.time()
    for i, e in enumerate(enei):
        sig, bem = bem.solve(exc.potential(p, e))
        ext[i] = float(np.real(np.ravel(exc.extinction(sig))[0]))
    dt = time.time() - t0
    print("[06] {:<14s}: {:.2f} s for {} wavelengths".format(label, dt, n))
    return ext, dt


def main():
    out_png = os.path.join(HERE, "06_iterative_aca.png")

    epstab = [EpsConst(1.0), EpsTable("gold.dat")]
    p = ComParticle(epstab,
                    [trisphere(256, 30.0)],
                    [[2, 1]], 1, interp="curv")
    print("[06] particle: {} faces".format(p.nfaces))

    exc = PlaneWaveRet(np.array([[1.0, 0.0, 0.0]]),
                       np.array([[0.0, 0.0, 1.0]]))

    enei = np.linspace(450.0, 700.0, 9)

    ext_dense, t_dense = run(BEMRet,     p, exc, enei, "BEMRet (dense)")
    ext_iter,  t_iter  = run(BEMRetIter, p, exc, enei, "BEMRetIter")

    rel_diff = np.max(np.abs(ext_iter - ext_dense) / np.abs(ext_dense))
    print("[06] max relative diff iter vs dense: {:.2e}".format(rel_diff))
    print("[06] speedup iter / dense: {:.2f}x"
          .format(t_dense / t_iter if t_iter > 0 else float("inf")))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(enei, ext_dense, "b-",  lw=2,  label="BEMRet (dense)")
    ax.plot(enei, ext_iter,  "ro--", lw=1.5, label="BEMRetIter (ACA+GMRES)")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Extinction (nm^2)")
    ax.set_title("Au sphere d=30 nm, dense vs iterative")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_png, dpi=140)
    print("[06] saved -> {}".format(out_png))


if __name__ == "__main__":
    main()
