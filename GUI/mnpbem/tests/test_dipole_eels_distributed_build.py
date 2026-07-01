"""Dipole / EELS excitation runners — distributed multi-GPU build regression.

These tests verify that the full ``MNPBEM_VRAM_SHARE_DISTRIBUTED=1`` path
(B-3 distributed Green-function build + cuSolverMg multi-GPU LU + 'mgpu'
tag dispatch) works end-to-end for the Dipole and EELS excitations, not
just the PlaneWave/extinction path the rest of the regression suite
already covers.

The existing ``test_v17_a5_excite_gpu.py`` validates that the excitation
runners host-materialise cupy ``sig`` correctly. This file goes one step
further: it builds the BEM solver under the FULL distributed env and
checks that:

  1. The LU tags inside BEMRet become ``'mgpu'`` (distributed build active).
  2. ``DipoleRet.decayrate(sig)`` matches the CPU baseline.
  3. ``EELSRet.loss(sig)`` and ``EELSRet.rad(sig)`` match the CPU baseline.
  4. Outputs are pure host NumPy (no cupy leak).

The test is skipped when cupy / cuSolverMg / >=2 CUDA devices are not
available so the file can sit in the default test suite without breaking
CPU-only CI.
"""

from __future__ import annotations

import os

import numpy as np
import pytest


try:
    import cupy as _cp  # type: ignore
    from mnpbem.utils.multi_gpu_lu import cusolvermg_available
    _N_GPUS = int(_cp.cuda.runtime.getDeviceCount())
    _HAS_MGPU = (_N_GPUS >= 2) and cusolvermg_available()
except Exception:
    _N_GPUS = 0
    _HAS_MGPU = False


mgpu_required = pytest.mark.skipif(
        not _HAS_MGPU,
        reason = 'cupy + cuSolverMg + >=2 GPUs required for distributed build')


# ---------------------------------------------------------------------------
# Env isolation
# ---------------------------------------------------------------------------


_ENV_KEYS = (
        'MNPBEM_GPU',
        'MNPBEM_GPU_NATIVE',
        'MNPBEM_GPU_THRESHOLD',
        'MNPBEM_VRAM_SHARE',
        'MNPBEM_VRAM_SHARE_DISTRIBUTED',
        'MNPBEM_VRAM_SHARE_GPUS',
        'MNPBEM_VRAM_SHARE_BACKEND',
        'MNPBEM_VRAM_SHARE_DEVICE_IDS')


@pytest.fixture
def cpu_env(monkeypatch):
    """Force all MNPBEM_VRAM_SHARE_* / MNPBEM_GPU* envs OFF."""

    for k in _ENV_KEYS:
        monkeypatch.delenv(k, raising = False)
    yield


@pytest.fixture
def mgpu_env(monkeypatch):
    """Enable full distributed multi-GPU build with the available device count."""

    for k in _ENV_KEYS:
        monkeypatch.delenv(k, raising = False)

    monkeypatch.setenv('MNPBEM_GPU', '1')
    monkeypatch.setenv('MNPBEM_VRAM_SHARE', '1')
    monkeypatch.setenv('MNPBEM_VRAM_SHARE_DISTRIBUTED', '1')
    monkeypatch.setenv('MNPBEM_VRAM_SHARE_GPUS', str(min(_N_GPUS, 4)))
    monkeypatch.setenv('MNPBEM_VRAM_SHARE_BACKEND', 'cusolvermg')
    yield


# ---------------------------------------------------------------------------
# Common fixture: tiny sphere with material
# ---------------------------------------------------------------------------


def _build_sphere(faces = 624):
    from mnpbem import EpsDrude, EpsConst, trisphere, ComParticle

    eps_in = EpsDrude(9.5, 138.0, 138.0)
    eps_out = EpsConst(1.0)

    return ComParticle([eps_out, eps_in], [trisphere(faces, 10.0)], [[2, 1]])


def _isclose_rel(a, b, tol = 1e-6):
    a, b = float(np.asarray(a).ravel()[0]), float(np.asarray(b).ravel()[0])
    return abs(a - b) / max(abs(a), 1e-30) < tol


def _is_host(arr):
    """Return True iff ``arr`` is a pure-host array (NumPy or scalar)."""
    if isinstance(arr, (float, int, complex)):
        return True
    return isinstance(arr, np.ndarray)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@mgpu_required
def test_dipole_ret_distributed_build_matches_cpu(cpu_env):
    """DipoleRet.decayrate produces CPU-identical output under mgpu build."""

    # CPU baseline first (cpu_env active)
    from mnpbem.bem import BEMRet
    from mnpbem.simulation import DipoleRet
    from mnpbem import ComPoint

    p = _build_sphere(624)
    bem = BEMRet(p)
    pt = ComPoint(p, np.array([[0.0, 0.0, 15.0]]))
    exc = DipoleRet(pt, np.array([0.0, 0.0, 1.0]))
    sig_cpu, _ = bem.solve(exc(p, 600.0))
    tot_cpu, rad_cpu, rad0_cpu = exc.decayrate(sig_cpu)

    # mgpu run — manipulate env directly (cannot mix fixtures cleanly)
    import importlib
    import sys

    for m in list(sys.modules):
        if m.startswith('mnpbem'):
            del sys.modules[m]

    saved = {k: os.environ.get(k) for k in _ENV_KEYS}

    try:
        os.environ['MNPBEM_GPU'] = '1'
        os.environ['MNPBEM_VRAM_SHARE'] = '1'
        os.environ['MNPBEM_VRAM_SHARE_DISTRIBUTED'] = '1'
        os.environ['MNPBEM_VRAM_SHARE_GPUS'] = str(min(_N_GPUS, 4))
        os.environ['MNPBEM_VRAM_SHARE_BACKEND'] = 'cusolvermg'

        from mnpbem.bem import BEMRet as BEMRet2
        from mnpbem.simulation import DipoleRet as DipoleRet2
        from mnpbem import ComPoint as ComPoint2

        p2 = _build_sphere(624)
        bem2 = BEMRet2(p2)
        pt2 = ComPoint2(p2, np.array([[0.0, 0.0, 15.0]]))
        exc2 = DipoleRet2(pt2, np.array([0.0, 0.0, 1.0]))
        sig_mg, _ = bem2.solve(exc2(p2, 600.0))

        # 1. distributed build is actually used — LU tags must be 'mgpu'
        assert bem2.G1_lu[0] == 'mgpu', (
                'expected G1_lu tag <mgpu>, got <{}>'.format(bem2.G1_lu[0]))
        assert bem2.G2_lu[0] == 'mgpu'
        assert bem2.Delta_lu[0] == 'mgpu'
        assert bem2.Sigma_lu[0] == 'mgpu'

        # 2. decayrate runs without error and matches CPU
        tot_mg, rad_mg, rad0_mg = exc2.decayrate(sig_mg)

        # 3. host-only output
        assert _is_host(tot_mg)
        assert _is_host(rad_mg)
        assert _is_host(rad0_mg)

        # 4. numerical agreement
        assert _isclose_rel(tot_cpu, tot_mg, tol = 1e-6), (
                'decayrate mismatch: cpu={} mg={}'.format(tot_cpu, tot_mg))

    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

        for m in list(sys.modules):
            if m.startswith('mnpbem'):
                del sys.modules[m]


@mgpu_required
def test_eels_ret_distributed_build_matches_cpu(cpu_env):
    """EELSRet.loss / EELSRet.rad produce CPU-identical output under mgpu build."""

    # CPU baseline first (cpu_env active)
    from mnpbem.bem import BEMRet
    from mnpbem.simulation import EELSRet

    p = _build_sphere(624)
    bem = BEMRet(p)
    impact = np.array([[12.0, 0.0]])
    vel = EELSRet.ene2vel(200e3)
    exc = EELSRet(p, impact, 0.5, vel)
    sig_cpu, _ = bem.solve(exc(p, 600.0))
    psurf_cpu, pbulk_cpu = exc.loss(sig_cpu)
    prad_cpu, _ = exc.rad(sig_cpu)

    # mgpu run
    import sys

    for m in list(sys.modules):
        if m.startswith('mnpbem'):
            del sys.modules[m]

    saved = {k: os.environ.get(k) for k in _ENV_KEYS}

    try:
        os.environ['MNPBEM_GPU'] = '1'
        os.environ['MNPBEM_VRAM_SHARE'] = '1'
        os.environ['MNPBEM_VRAM_SHARE_DISTRIBUTED'] = '1'
        os.environ['MNPBEM_VRAM_SHARE_GPUS'] = str(min(_N_GPUS, 4))
        os.environ['MNPBEM_VRAM_SHARE_BACKEND'] = 'cusolvermg'

        from mnpbem.bem import BEMRet as BEMRet2
        from mnpbem.simulation import EELSRet as EELSRet2

        p2 = _build_sphere(624)
        bem2 = BEMRet2(p2)
        impact2 = np.array([[12.0, 0.0]])
        exc2 = EELSRet2(p2, impact2, 0.5, EELSRet2.ene2vel(200e3))
        sig_mg, _ = bem2.solve(exc2(p2, 600.0))

        # 1. distributed build active
        assert bem2.G1_lu[0] == 'mgpu', (
                'expected G1_lu tag <mgpu>, got <{}>'.format(bem2.G1_lu[0]))

        # 2. loss / rad run without error and match CPU
        psurf_mg, pbulk_mg = exc2.loss(sig_mg)
        prad_mg, _ = exc2.rad(sig_mg)

        # 3. host-only output
        assert _is_host(psurf_mg)
        assert _is_host(pbulk_mg)
        assert _is_host(prad_mg)

        # 4. numerical agreement (1e-6 — single wavelength, full mgpu chain)
        assert _isclose_rel(psurf_cpu, psurf_mg, tol = 1e-6), (
                'eels.loss psurf mismatch: cpu={} mg={}'.format(psurf_cpu, psurf_mg))
        # pbulk is 0 for this geometry, both must match identically
        assert abs(float(np.asarray(pbulk_mg).ravel()[0])
                - float(np.asarray(pbulk_cpu).ravel()[0])) < 1e-12

        assert _isclose_rel(np.real(prad_cpu), np.real(prad_mg), tol = 1e-6), (
                'eels.rad mismatch: cpu={} mg={}'.format(prad_cpu, prad_mg))

    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

        for m in list(sys.modules):
            if m.startswith('mnpbem'):
                del sys.modules[m]


@mgpu_required
def test_dipole_eels_mgpu_wavelength_sweep_stable(cpu_env):
    """The mgpu wavelength loop holds across multiple wavelengths.

    Catches state-carry bugs in the mgpu LU handle cache that would
    surface only across iterations (re-factorisations, cleanup races).
    """

    import sys

    for m in list(sys.modules):
        if m.startswith('mnpbem'):
            del sys.modules[m]

    saved = {k: os.environ.get(k) for k in _ENV_KEYS}

    try:
        os.environ['MNPBEM_GPU'] = '1'
        os.environ['MNPBEM_VRAM_SHARE'] = '1'
        os.environ['MNPBEM_VRAM_SHARE_DISTRIBUTED'] = '1'
        os.environ['MNPBEM_VRAM_SHARE_GPUS'] = str(min(_N_GPUS, 4))
        os.environ['MNPBEM_VRAM_SHARE_BACKEND'] = 'cusolvermg'

        from mnpbem.bem import BEMRet
        from mnpbem.simulation import DipoleRet, EELSRet
        from mnpbem import ComPoint

        p = _build_sphere(624)

        # Dipole sweep
        bem = BEMRet(p)
        pt = ComPoint(p, np.array([[0.0, 0.0, 15.0]]))
        exc_d = DipoleRet(pt, np.array([0.0, 0.0, 1.0]))

        wavelengths = [500.0, 600.0, 700.0]
        decay_vals = []

        for wl in wavelengths:
            sig, _ = bem.solve(exc_d(p, wl))
            tot, _, _ = exc_d.decayrate(sig)
            decay_vals.append(float(np.asarray(tot).ravel()[0]))

        # Decay rate must monotonically increase for this sphere — sanity
        for i in range(1, len(decay_vals)):
            assert decay_vals[i] > decay_vals[i - 1], (
                    'expected monotonic decay across wl, got {}'.format(decay_vals))

        # EELS sweep
        bem_e = BEMRet(p)
        impact = np.array([[12.0, 0.0]])
        exc_e = EELSRet(p, impact, 0.5, EELSRet.ene2vel(200e3))

        psurf_vals = []
        prad_vals = []

        for wl in wavelengths:
            sig, _ = bem_e.solve(exc_e(p, wl))
            ps, _ = exc_e.loss(sig)
            pr, _ = exc_e.rad(sig)
            psurf_vals.append(float(np.asarray(ps).ravel()[0]))
            prad_vals.append(float(np.asarray(pr).real.ravel()[0]))

        # All outputs are finite host floats
        for v in decay_vals + psurf_vals + prad_vals:
            assert np.isfinite(v), 'non-finite value in sweep'

    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

        for m in list(sys.modules):
            if m.startswith('mnpbem'):
                del sys.modules[m]
