"""
Example 01 — Gold sphere extinction spectrum (retarded BEM).

Equivalent MATLAB demo: Demo/planewave/ret/demospecret1.m

This is the smallest possible end-to-end script: build a 144-face gold
sphere, sweep 41 wavelengths from 400-800 nm, plot extinction /
scattering / absorption, and save the figure as PNG.

Run:
    python examples/01_sphere_extinction.py
"""

import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Make the package importable when running from a fresh checkout
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from mnpbem.materials import EpsConst, EpsTable
from mnpbem.geometry import trisphere, ComParticle
from mnpbem.bem import BEMRet
from mnpbem.simulation import PlaneWaveRet


def main():
    out_png = os.path.join(HERE, "01_sphere_extinction.png")

    # Materials: vacuum outside, gold (Johnson & Christy) inside
    epstab = [EpsConst(1.0), EpsTable("gold.dat")]

    # 20 nm diameter gold sphere, 144 triangular faces
    sphere = trisphere(144, 20)
    p = ComParticle(epstab, [sphere], [[2, 1]], 1, interp="curv")
    print("[01] particle: {} faces".format(p.nfaces))

    # Retarded BEM solver (full Maxwell)
    bem = BEMRet(p)

    # Plane wave: x-polarized, propagating along z
    exc = PlaneWaveRet(np.array([[1.0, 0.0, 0.0]]),
                       np.array([[0.0, 0.0, 1.0]]))

    # 21 wavelengths is enough to resolve the plasmon peak; bump to 41
    # for a smoother curve (about 2x runtime).
    enei = np.linspace(400.0, 800.0, 21)
    ext = np.zeros_like(enei)
    sca = np.zeros_like(enei)

    t0 = time.time()
    for i, e in enumerate(enei):
        sig, bem = bem.solve(exc.potential(p, e))
        ext[i] = float(np.real(np.ravel(exc.extinction(sig))[0]))
        sca_val = exc.scattering(sig)
        sca[i] = float(np.real(
            np.ravel(sca_val[0] if isinstance(sca_val, tuple) else sca_val)[0]
        ))
    abs_ = ext - sca
    elapsed = time.time() - t0
    print("[01] sweep: {} wavelengths in {:.2f} s".format(len(enei), elapsed))

    # Plot
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(enei, ext, "b-",  lw=2, label="extinction")
    ax.plot(enei, sca, "g--", lw=2, label="scattering")
    ax.plot(enei, abs_, "r:", lw=2, label="absorption")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Cross section (nm^2)")
    ax.set_title("Au sphere d=20 nm, BEMRet, 144 faces")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_png, dpi=140)
    print("[01] saved -> {}".format(out_png))

    # Sanity check: extinction peaks near plasmon resonance ~520 nm
    idx = int(np.argmax(ext))
    print("[01] plasmon peak: lambda = {:.0f} nm, ext = {:.2f} nm^2"
          .format(enei[idx], ext[idx]))


if __name__ == "__main__":
    main()
