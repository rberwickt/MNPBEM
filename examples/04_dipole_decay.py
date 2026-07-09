"""
Example 04 — Dipole decay rate above a gold sphere.

Equivalent MATLAB demo: Demo/dipole/ret/demodipret1.m

A radiating dipole hovers above a 150 nm gold sphere. As the dipole-
sphere distance decreases the total decay rate is enhanced (Purcell
effect) while the radiative rate eventually drops as quenching takes
over.

The script plots both x-oriented and z-oriented dipoles, and the
total / radiative decay rates, on a log scale.

Run:
    python examples/04_dipole_decay.py
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
from mnpbem.geometry import trisphere, ComParticle, ComPoint
from mnpbem.bem import BEMRet
from mnpbem.simulation import DipoleRet


def main():
    out_png = os.path.join(HERE, "04_dipole_decay.png")

    epstab = [EpsConst(1.0), EpsTable("gold.dat")]
    diameter = 150.0
    sphere = trisphere(144, diameter)
    p = ComParticle(epstab, [sphere], [[2, 1]], 1, interp="curv")
    print("[04] particle: {} faces".format(p.nfaces))

    enei = 550.0
    z = np.linspace(0.6, 1.5, 21) * diameter        # dipole height (nm)
    pos = np.column_stack([np.zeros_like(z),
                           np.zeros_like(z),
                           z])
    pt = ComPoint(p, pos)

    # Two dipole orientations: x and z
    dip_dir = np.array([[1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0]])
    exc = DipoleRet(pt, dip_dir)

    bem = BEMRet(p)
    t0 = time.time()
    sig, bem = bem.solve(exc.potential(p, enei))
    tot, rad, _rad0 = exc.decayrate(sig)
    elapsed = time.time() - t0
    print("[04] solved at lambda={} nm in {:.2f} s".format(enei, elapsed))

    tot = np.asarray(tot).real
    rad = np.asarray(rad).real
    # tot / rad shape: (n_positions, n_orientations)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.semilogy(z, tot[:, 0], "b-",  lw=2, label="tot (x)")
    ax.semilogy(z, tot[:, 1], "r-",  lw=2, label="tot (z)")
    ax.semilogy(z, rad[:, 0], "b--", lw=2, label="rad (x)")
    ax.semilogy(z, rad[:, 1], "r--", lw=2, label="rad (z)")
    ax.set_xlabel("Dipole position z (nm)")
    ax.set_ylabel("Decay rate / Gamma_0")
    ax.set_title("Dipole above Au sphere d=150 nm, lambda={} nm".format(enei))
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    plt.tight_layout()
    fig.savefig(out_png, dpi=140)
    print("[04] saved -> {}".format(out_png))


if __name__ == "__main__":
    main()
