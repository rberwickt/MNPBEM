"""
Sphere on substrate - retarded BEM comparison (Python version).
Gold nanosphere (diameter 20nm) on glass substrate (eps=2.25).
Normal incidence planewave, TM polarization.
Saves scattering, extinction, absorption cross sections to CSV.
"""
import os
import sys
import time

import numpy as np

from mnpbem.materials import EpsConst, EpsTable
from mnpbem.geometry import trisphere, ComParticle, LayerStructure
from mnpbem.bem import BEMRetLayer
from mnpbem.simulation import PlaneWaveRetLayer

# 1. Materials (indices match MATLAB 1-based convention)
epstab = [EpsConst(1.0), EpsTable('gold.dat'), EpsConst(2.25)]

# 2. Layer structure: vacuum above z=0, glass below
layer = LayerStructure(epstab, [1, 3], [0.0])

# 3. Particle: 20nm diameter gold sphere, 1nm above substrate
sphere = trisphere(144, 20.0)
z_min = sphere.pos[:, 2].min()
sphere.shift([0, 0, -z_min + 1.0])

# 4. ComParticle: gold(2) inside, vacuum(1) outside
p = ComParticle(epstab, [sphere], [[2, 1]], [1])

# 5. Normal-incidence TM planewave from above
pol = np.array([[1.0, 0.0, 0.0]])
dir_vec = np.array([[0.0, 0.0, -1.0]])
exc = PlaneWaveRetLayer(pol, dir_vec, layer)

# 6. BEM solver (retarded + layer)
bem = BEMRetLayer(p, layer)

# 7. Wavelength loop
enei_arr = np.linspace(450, 750, 31)
sca = np.zeros(len(enei_arr))
ext = np.zeros(len(enei_arr))
ab = np.zeros(len(enei_arr))

print("Python simulation starting...")
t0 = time.time()

for i, enei in enumerate(enei_arr):
    exc_pot = exc(p, enei)
    sig, _ = bem.solve(exc_pot)
    sca_val, _ = exc.scattering(sig)
    ext_val = exc.extinction(sig)
    sca[i] = sca_val
    ext[i] = ext_val
    ab[i] = ext_val - sca_val
    print("  [{}/{}] lambda = {:.1f} nm, sca = {:.4e}, ext = {:.4e}".format(
        i + 1, len(enei_arr), enei, sca_val, ext_val))

elapsed = time.time() - t0

# 8. Save results
output_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(output_dir, 'python_results.csv')

with open(output_path, 'w') as f:
    f.write('wavelength_nm,scattering,extinction,absorption\n')
    for i in range(len(enei_arr)):
        f.write('{:.6f},{:.10e},{:.10e},{:.10e}\n'.format(
            enei_arr[i], sca[i], ext[i], ab[i]))

print("\nPython simulation complete in {:.1f} sec.".format(elapsed))
print("Results saved to {}".format(output_path))
idx = np.argmax(sca)
print("Peak scattering at {:.1f} nm: {:.4e} nm^2".format(enei_arr[idx], sca[idx]))
