"""Multi-RHS (multi-dipole) BEM solve correctness vs sequential.

When a dipole excitation has ndip > 1 (e.g. 3 orthogonal dipoles), the
excitation potentials form a multi-column RHS that BEM solvers handle in
one call. This test verifies bit-by-bit equality (within 1e-12) against
running the BEM solve once per dipole component.
"""
import os
import sys

import numpy as np


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mnpbem.materials import EpsConst, EpsTable
from mnpbem.geometry import trisphere, ComParticle, ComPoint
from mnpbem.bem import BEMStat, BEMRet
from mnpbem.simulation import DipoleStat, DipoleRet


def _build():
    epstab = [EpsConst(1), EpsTable('gold.dat')]
    sphere = trisphere(144, 20)
    p = ComParticle(epstab, [sphere], [[2, 1]], 1, interp = 'curv')
    pt = ComPoint(p, np.array([[0.0, 0.0, 25.0]]))
    return p, pt


def test_dipolestat_multi_rhs_vs_sequential():
    p, pt = _build()
    enei = 600.0

    # Multi-RHS: 3 orthogonal dipoles in one solve
    dip3 = DipoleStat(pt, dip = np.eye(3))
    bem = BEMStat(p)
    exc3 = dip3(p, enei)
    sig3, _ = bem.solve(exc3)
    sig3_arr = sig3.sig.reshape(sig3.sig.shape[0], -1)

    # Sequential: one solve per dipole component
    seqs = []
    for d in [[1, 0, 0], [0, 1, 0], [0, 0, 1]]:
        di = DipoleStat(pt, dip = np.array([d], dtype = float))
        si, _ = bem.solve(di(p, enei))
        seqs.append(si.sig.ravel())
    seq_arr = np.stack(seqs, axis = -1)

    assert np.max(np.abs(sig3_arr - seq_arr)) < 1e-12


def test_dipoleret_multi_rhs_vs_sequential():
    p, pt = _build()
    enei = 600.0

    dip3 = DipoleRet(pt, dip = np.eye(3))
    bem = BEMRet(p)
    sig3, _ = bem.solve(dip3(p, enei))

    seqs_sig1, seqs_sig2, seqs_h1, seqs_h2 = [], [], [], []
    for d in [[1, 0, 0], [0, 1, 0], [0, 0, 1]]:
        di = DipoleRet(pt, dip = np.array([d], dtype = float))
        si, _ = bem.solve(di(p, enei))
        seqs_sig1.append(si.sig1.ravel())
        seqs_sig2.append(si.sig2.ravel())
        seqs_h1.append(si.h1.reshape(-1, 3))
        seqs_h2.append(si.h2.reshape(-1, 3))

    sig1_seq = np.stack(seqs_sig1, axis = -1)
    sig2_seq = np.stack(seqs_sig2, axis = -1)
    h1_seq = np.stack(seqs_h1, axis = -1)
    h2_seq = np.stack(seqs_h2, axis = -1)

    assert np.max(np.abs(sig3.sig1 - sig1_seq)) < 1e-12
    assert np.max(np.abs(sig3.sig2 - sig2_seq)) < 1e-12
    assert np.max(np.abs(sig3.h1 - h1_seq)) < 1e-12
    assert np.max(np.abs(sig3.h2 - h2_seq)) < 1e-12
