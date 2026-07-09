"""
Example 03 — Gold sphere on a glass substrate (BEMRetLayer).

Equivalent MATLAB demo: Demo/planewave/ret/demospecret7.m

A 20 nm gold sphere sits 1 nm above a glass interface (eps=2.25).
Plane wave illumination from above, normal incidence, x-polarized.

The substrate Green function is precomputed via `GreenTabLayer` for a
short list of wavelengths and reused inside the spectrum loop.

Run:
    python examples/03_substrate_layered.py
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
from mnpbem.geometry import trisphere, ComParticle, LayerStructure
from mnpbem.bem import BEMRetLayer
from mnpbem.simulation import PlaneWaveRetLayer
from mnpbem.greenfun import GreenTabLayer


def main():
    out_png = os.path.join(HERE, "03_substrate_layered.png")

    # vacuum (1) above, glass (2.25) below (n = 1.5)
    epstab = [EpsConst(1.0), EpsTable("gold.dat"), EpsConst(2.25)]
    layer = LayerStructure(epstab, [1, 3], [0.0])

    # 20 nm sphere, lifted so its lowest point is 1 nm above z=0
    sphere = trisphere(144, 20.0)
    sphere.shift([0.0, 0.0, -sphere.pos[:, 2].min() + 1.0])
    p = ComParticle(epstab, [sphere], [[2, 1]], [1])
    print("[03] particle: {} faces".format(p.nfaces))

    # Precompute the substrate Green function on a short tabulation
    tab = layer.tabspace(p)
    gt = GreenTabLayer(layer, tab=tab)
    gt.set(np.linspace(450.0, 750.0, 5))
    bem = BEMRetLayer(p, layer, greentab=gt)

    # Plane wave from above: x-polarized, propagating in -z
    pol = np.array([[1.0, 0.0, 0.0]])
    dir_vec = np.array([[0.0, 0.0, -1.0]])
    exc = PlaneWaveRetLayer(pol, dir_vec, layer)

    enei = np.linspace(450.0, 750.0, 11)
    ext = np.zeros_like(enei)
    sca = np.zeros_like(enei)

    t0 = time.time()
    for i, e in enumerate(enei):
        sig, _ = bem.solve(exc(p, e))
        sca_val, _ = exc.scattering(sig)
        ext[i] = float(exc.extinction(sig))
        sca[i] = float(sca_val)
    abs_ = ext - sca
    elapsed = time.time() - t0
    print("[03] sweep: {} wavelengths in {:.2f} s".format(len(enei), elapsed))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(enei, ext, "b-",  lw=2, label="extinction")
    ax.plot(enei, sca, "g--", lw=2, label="scattering")
    ax.plot(enei, abs_, "r:", lw=2, label="absorption")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Cross section (nm^2)")
    ax.set_title("Au sphere on glass, gap=1 nm, BEMRetLayer")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_png, dpi=140)
    print("[03] saved -> {}".format(out_png))


if __name__ == "__main__":
    main()
