"""
Example 02 — Gold sphere dimer (gap-mode plasmon).

Loosely follows Demo/planewave/ret/demospecret11.m (two-sphere dimer).

Two 20 nm gold spheres separated by a 2 nm gap, illuminated normal to
the dimer axis. Polarization along the axis lights up the strong gap
mode (red-shifted from the single-sphere resonance); polarization
perpendicular to it gives a near-single-sphere response.

Run:
    python examples/02_dimer_gold.py
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
from mnpbem.bem import BEMRet
from mnpbem.simulation import PlaneWaveRet


def main():
    out_png = os.path.join(HERE, "02_dimer_gold.png")

    # gold listed twice -> two distinct dielectric blocks for the two
    # sphere surfaces (avoids singular BEM matrix when surfaces share
    # an interior dielectric index).
    epstab = [EpsConst(1.0), EpsTable("gold.dat"), EpsTable("gold.dat")]

    diameter = 20.0
    gap = 2.0
    half = (diameter + gap) / 2.0

    p1 = trisphere(144, diameter); p1.shift([-half, 0.0, 0.0])
    p2 = trisphere(144, diameter); p2.shift([+half, 0.0, 0.0])

    # inout: sphere 1 -> eps[1]/eps[0], sphere 2 -> eps[2]/eps[0]
    p = ComParticle(epstab,
                    [p1, p2],
                    [[2, 1], [3, 1]],
                    [1, 2],
                    interp="curv")
    print("[02] particle: {} faces".format(p.nfaces))

    bem = BEMRet(p)

    pol = np.array([[1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0]])
    dir_vec = np.array([[0.0, 0.0, 1.0],
                        [0.0, 0.0, 1.0]])
    exc = PlaneWaveRet(pol, dir_vec)

    enei = np.linspace(450.0, 800.0, 13)
    ext_par = np.zeros_like(enei)
    ext_per = np.zeros_like(enei)

    t0 = time.time()
    for i, e in enumerate(enei):
        sig, bem = bem.solve(exc.potential(p, e))
        ext = np.real(np.ravel(exc.extinction(sig)))
        ext_par[i] = float(ext[0])
        ext_per[i] = float(ext[1])
    elapsed = time.time() - t0
    print("[02] sweep: {} wavelengths in {:.2f} s".format(len(enei), elapsed))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(enei, ext_par, "b-",  lw=2, label="parallel pol (x, gap mode)")
    ax.plot(enei, ext_per, "r--", lw=2, label="perpendicular pol (y)")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Extinction (nm^2)")
    ax.set_title("Au dimer d=20 nm, gap=2 nm, BEMRet")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_png, dpi=140)
    print("[02] saved -> {}".format(out_png))


if __name__ == "__main__":
    main()
