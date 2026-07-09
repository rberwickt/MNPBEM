"""
Example 05 — EELS spectrum of a silver nanosphere.

Equivalent MATLAB demo: Demo/eels/ret/demoeelsret1.m

A 200 keV electron beam passes near a 150 nm silver nanosphere with
20 nm impact parameter. The script computes the surface-mode loss
probability vs. loss energy in 1.5-4.5 eV.

Run:
    python examples/05_eels.py
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
from mnpbem.simulation import EELSRet, EELSBase
from mnpbem.misc import EV2NM


def main():
    out_png = os.path.join(HERE, "05_eels.png")

    epstab = [EpsConst(1.0), EpsTable("silver.dat")]
    diameter = 150.0
    sphere = trisphere(256, diameter)
    p = ComParticle(epstab, [sphere], [[2, 1]], 1, interp="curv")
    print("[05] particle: {} faces".format(p.nfaces))

    width = 0.5                                  # beam width (nm)
    vel = EELSBase.ene2vel(200e3)                # 200 keV electron
    impact = np.array([[diameter / 2.0 + 20.0, 0.0]])  # nm offset

    ene_eV = np.linspace(1.5, 4.5, 21)
    enei = EV2NM / ene_eV                        # nm wavelength

    bem = BEMRet(p)
    exc = EELSRet(p, impact, width, vel)

    psurf = np.zeros_like(ene_eV)
    t0 = time.time()
    for i, e_nm in enumerate(enei):
        sig, bem = bem.solve(exc(p, e_nm))
        loss = exc.loss(sig)
        psurf[i] = float(np.real(np.ravel(loss)[0]))
    elapsed = time.time() - t0
    print("[05] sweep: {} energies in {:.2f} s".format(len(ene_eV), elapsed))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(ene_eV, psurf, "bo-", lw=2, label="BEM")
    ax.set_xlabel("Loss energy (eV)")
    ax.set_ylabel("Loss probability (1/eV)")
    ax.set_title("EELS, Ag sphere d=150 nm, 200 keV, b=20 nm")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_png, dpi=140)
    print("[05] saved -> {}".format(out_png))


if __name__ == "__main__":
    main()
