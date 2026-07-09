"""Check g.con and eps1/eps2 for case_g."""
import sys
sys.path.insert(0, '/home/yoojk20/scratch/v151_beta_iter_drift')
import numpy as np
from mnpbem.materials import EpsConst, EpsTable
from mnpbem.geometry import trisphere, ComParticle
from mnpbem.bem import BEMRet
from mnpbem.greenfun import CompGreenRet

epstab = [EpsConst(1.77), EpsTable('gold.dat'), EpsTable('silver.dat')]
core_d, shell_t = 5.0, 1.5
outer_d = core_d + 2*shell_t
gap = 0.6
half = (outer_d + gap) / 2.0
p1_shell = trisphere(144, outer_d); p1_core = trisphere(144, core_d)
p1_shell.shift([-half, 0, 0]); p1_core.shift([-half, 0, 0])
p2_shell = trisphere(144, outer_d); p2_core = trisphere(144, core_d)
p2_shell.shift([+half, 0, 0]); p2_core.shift([+half, 0, 0])
inds = [[3, 1], [2, 3], [3, 1], [2, 3]]
p = ComParticle(epstab, [p1_shell, p1_core, p2_shell, p2_core], inds, [1, 2], interp='curv')

print('p.nfaces =', p.nfaces)
print('p.eps =', p.eps)

g = CompGreenRet(p, p)
print('g.con[0][1] =', np.array(g.con[0][1]))
print('g.con[1][0] =', np.array(g.con[1][0]))
print('all g.con[0][1] zero?', np.all(np.array(g.con[0][1]) == 0))

# eps1 / eps2 at 540nm (mid-band where drift is highest)
e = 540.0
eps1 = p.eps1(e)
eps2 = p.eps2(e)
print('eps1 unique:', np.unique(eps1))
print('eps2 unique:', np.unique(eps2))
print('eps1 uniform?', np.allclose(eps1, eps1[0]))
print('eps2 uniform?', np.allclose(eps2, eps2[0]))

# So in BEMRet, since eps is non-scalar AND con is non-zero -> uses L1 = G eps G^-1
# In BEMRetIter, _afun uses element-wise eps * Gsig
print()
print('==> BEMRet would use L1 = G1 @ eps1 @ G1i (full case)')
print('==> BEMRetIter _afun uses eps1 * Gsig1 (point-wise)')
print('These are NOT equivalent when eps is non-uniform.')
