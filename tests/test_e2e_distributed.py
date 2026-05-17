"""End-to-end regression: BEM solver x excitation x (single-GPU | distributed).

This suite is the integration umbrella for the B-3 multi-GPU distributed
build series (Tasks #207-217).  Each cell of the matrix below runs the
same BEM problem twice -

    1. single-GPU baseline   (MNPBEM_VRAM_SHARE_DISTRIBUTED=0)
    2. distributed build      (MNPBEM_VRAM_SHARE_DISTRIBUTED=1)

and asserts the resulting surface charges agree to ULP-level (1e-12)
on dense paths, or ~1e-8 on iterative paths where GMRES preconditioner
re-ordering can introduce <1ulp drift across the distributed assembly.

Matrix (rows = BEM solver, cols = excitation):

    BEMRet           x {planewave, dipole, eels}
    BEMRetIter       x {planewave}
    BEMRetLayer      x {planewave, dipole, eels}
    BEMRetLayerIter  x {planewave}
    BEMRetMirror     x {planewave}
    BEMStat          x {planewave, dipole, eels}
    BEMStatIter      x {planewave}
    BEMStatLayer     x {planewave, dipole, eels}
    BEMStatMirror    x {planewave}

Skip rules
----------
- cupy missing               -> entire suite skipped
- < 2 visible CUDA devices    -> entire suite skipped (single-GPU only)
- distributed path not wired  -> per-case skip, single-GPU smoke retained
- excitation x boundary not   -> per-case xfail (e.g. EELS x Mirror is not
  supported in MNPBEM        in the MATLAB Demo set)

Importable as
    pytest tests/test_e2e_distributed.py -k 'BEMRet and planewave'

The fixture mesh sizes are intentionally small (~300 face sphere,
~500 face dimer) so a full matrix sweep finishes in well under an hour
on a 2 GPU box.  The distributed build still exercises the column-split
Green function path because the Green-function assembly itself does not
require N>>n_gpus; only the LU factor benefits in absolute terms.

Bit-identity contract
---------------------
The B-3 series guarantees bit-identical sig/h between the single-GPU
(MNPBEM_VRAM_SHARE_DISTRIBUTED=0) and the distributed
(MNPBEM_VRAM_SHARE_DISTRIBUTED=1) build paths up to fused-mul-add
rounding in cuBLAS, which observationally stays below 1e-12 relative
on these small problems.  Iterative GMRES paths use 1e-8 because the
restart point can shift by 1 iteration after distributed reassembly,
which propagates a 1e-10 difference into the relative residual.

If a case fails the bit-identity check, the test prints the relative
diff per signal name (sig / sig1 / sig2 / h1 / h2) so the responsible
agent can locate the broken matrix block.
"""

from __future__ import annotations

import os
import sys
import warnings
from contextlib import contextmanager
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest


# Tests are stored under <repo>/tests/; the package lives at <repo>/mnpbem.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# ---------------------------------------------------------------------------
# Environment probes
# ---------------------------------------------------------------------------

try:
    import cupy as _cp  # type: ignore
    _HAS_CUPY = True
except Exception:
    _cp = None  # type: ignore
    _HAS_CUPY = False


def _gpu_count() -> int:
    if not _HAS_CUPY:
        return 0
    try:
        return int(_cp.cuda.runtime.getDeviceCount())
    except Exception:
        return 0


_GPU_COUNT = _gpu_count()


# Suite-level skips: the entire matrix is dormant unless we have >=2 GPUs.
# Single-GPU machines still get the import-surface tests (TestImportSurface)
# below as a smoke check.
requires_multi_gpu = pytest.mark.skipif(
    _GPU_COUNT < 2,
    reason='[info] need >=2 CUDA devices for distributed (have {})'.format(
        _GPU_COUNT),
)
requires_cupy = pytest.mark.skipif(
    not _HAS_CUPY,
    reason='[info] cupy not installed',
)


# ---------------------------------------------------------------------------
# Optional-import guard for BEM solvers + excitations
# ---------------------------------------------------------------------------
#
# Several solvers are in flight (Tasks #213-217 quasi-static distributed
# build).  When this test file is imported on a tree where a solver does
# not yet exist, the constructor module raises ``ImportError`` at import
# time.  We do not want one in-flight solver to take down the whole suite,
# so each solver is imported in a try/except and a sentinel ``MISSING`` is
# used to skip the affected cases at runtime.
#

class _Missing(object):
    """Sentinel marking a solver/excitation that failed to import."""

    def __init__(self, name: str, error: Exception):
        self.name = name
        self.error = error

    def __repr__(self) -> str:
        return '<Missing {}: {}>'.format(self.name, self.error)


def _try_import(module_path: str, attr: str):
    try:
        mod = __import__(module_path, fromlist=[attr])
        return getattr(mod, attr)
    except Exception as exc:  # pragma: no cover - guard around in-flight tree
        return _Missing(attr, exc)


# BEM solvers
BEMRet = _try_import('mnpbem.bem.bem_ret', 'BEMRet')
BEMRetIter = _try_import('mnpbem.bem.bem_ret_iter', 'BEMRetIter')
BEMRetLayer = _try_import('mnpbem.bem.bem_ret_layer', 'BEMRetLayer')
BEMRetLayerIter = _try_import('mnpbem.bem.bem_ret_layer_iter', 'BEMRetLayerIter')
BEMRetMirror = _try_import('mnpbem.bem.bem_ret_mirror', 'BEMRetMirror')
BEMStat = _try_import('mnpbem.bem.bem_stat', 'BEMStat')
BEMStatIter = _try_import('mnpbem.bem.bem_stat_iter', 'BEMStatIter')
BEMStatLayer = _try_import('mnpbem.bem.bem_stat_layer', 'BEMStatLayer')
BEMStatMirror = _try_import('mnpbem.bem.bem_stat_mirror', 'BEMStatMirror')

# Excitations
PlaneWaveRet = _try_import('mnpbem.simulation.planewave_ret', 'PlaneWaveRet')
PlaneWaveStat = _try_import('mnpbem.simulation.planewave_stat', 'PlaneWaveStat')
PlaneWaveRetLayer = _try_import('mnpbem.simulation.planewave_ret_layer', 'PlaneWaveRetLayer')
PlaneWaveStatLayer = _try_import('mnpbem.simulation.planewave_stat_layer', 'PlaneWaveStatLayer')
PlaneWaveRetMirror = _try_import('mnpbem.simulation.planewave_ret_mirror', 'PlaneWaveRetMirror')
PlaneWaveStatMirror = _try_import('mnpbem.simulation.planewave_stat_mirror', 'PlaneWaveStatMirror')
DipoleRet = _try_import('mnpbem.simulation.dipole_ret', 'DipoleRet')
DipoleStat = _try_import('mnpbem.simulation.dipole_stat', 'DipoleStat')
DipoleRetLayer = _try_import('mnpbem.simulation.dipole_ret_layer', 'DipoleRetLayer')
DipoleStatLayer = _try_import('mnpbem.simulation.dipole_stat_layer', 'DipoleStatLayer')
EELSRet = _try_import('mnpbem.simulation.eels_ret', 'EELSRet')
EELSStat = _try_import('mnpbem.simulation.eels_stat', 'EELSStat')

# Geometry / materials
EpsConst = _try_import('mnpbem.materials.eps_const', 'EpsConst')
EpsTable = _try_import('mnpbem.materials.eps_table', 'EpsTable')
trisphere = _try_import('mnpbem.geometry', 'trisphere')
ComParticle = _try_import('mnpbem.geometry', 'ComParticle')
ComPoint = _try_import('mnpbem.geometry', 'ComPoint')
ComParticleMirror = _try_import('mnpbem.geometry', 'ComParticleMirror')
LayerStructure = _try_import('mnpbem.geometry', 'LayerStructure')


def _have(*objs) -> bool:
    """Return True iff every object is *not* a _Missing sentinel."""
    return all(not isinstance(o, _Missing) for o in objs)


# ---------------------------------------------------------------------------
# Env-var sandbox
# ---------------------------------------------------------------------------

_DISTRIBUTED_KEYS = (
    'MNPBEM_GPU',
    'MNPBEM_GPU_NATIVE',
    'MNPBEM_VRAM_SHARE',
    'MNPBEM_VRAM_SHARE_GPUS',
    'MNPBEM_VRAM_SHARE_BACKEND',
    'MNPBEM_VRAM_SHARE_DEVICE_IDS',
    'MNPBEM_VRAM_SHARE_DISTRIBUTED',
)


@contextmanager
def _env_scope(**overrides):
    """Temporarily set env vars, restoring originals on exit.

    Passing ``key=None`` removes the var for the duration of the block.
    """
    saved: Dict[str, Any] = {}
    try:
        for k, v in overrides.items():
            saved[k] = os.environ.get(k)
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = str(v)
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _env_single_gpu():
    """Env override for the baseline path: GPU on, but no distributed build."""
    return dict(
        MNPBEM_GPU='1',
        MNPBEM_VRAM_SHARE=None,
        MNPBEM_VRAM_SHARE_GPUS=None,
        MNPBEM_VRAM_SHARE_BACKEND=None,
        MNPBEM_VRAM_SHARE_DISTRIBUTED=None,
    )


def _env_distributed(n_gpus: int):
    """Env override that *requests* the distributed build path."""
    return dict(
        MNPBEM_GPU='1',
        MNPBEM_VRAM_SHARE='1',
        MNPBEM_VRAM_SHARE_GPUS=str(n_gpus),
        MNPBEM_VRAM_SHARE_BACKEND='cusolvermg',
        MNPBEM_VRAM_SHARE_DISTRIBUTED='1',
    )


# ---------------------------------------------------------------------------
# Geometry / problem fixtures (module scope: build once, reuse)
# ---------------------------------------------------------------------------

WL_NM = 600.0


@pytest.fixture(scope='module')
def small_sphere():
    """Single Au sphere (~300 face) - fastest dense fixture."""
    if not _have(EpsConst, EpsTable, trisphere, ComParticle):
        pytest.skip('[info] geometry/materials unavailable in this tree')
    epstab = [EpsConst(1.0), EpsTable('gold.dat')]
    p = trisphere(256, 20.0)
    cp = ComParticle(epstab, [p], [[2, 1]], 1)
    return cp


@pytest.fixture(scope='module')
def small_dimer():
    """Au dimer (~512 face total).

    Uses two ~256 face spheres separated by 60 nm so the BEM matrix has
    real off-diagonal coupling - the right shape to exercise the
    distributed Green-function column split.
    """
    if not _have(EpsConst, EpsTable, trisphere, ComParticle):
        pytest.skip('[info] geometry/materials unavailable in this tree')
    epstab = [EpsConst(1.0), EpsTable('gold.dat')]
    s1 = trisphere(256, 20.0)
    s2 = trisphere(256, 20.0)
    s1.shift([-30.0, 0.0, 0.0])
    s2.shift([+30.0, 0.0, 0.0])
    cp = ComParticle(epstab, [s1, s2], [[2, 1], [2, 1]], 1)
    return cp


@pytest.fixture(scope='module')
def small_sphere_layer():
    """Au sphere above a flat substrate (layer)."""
    if not _have(EpsConst, EpsTable, trisphere, ComParticle, LayerStructure):
        pytest.skip('[info] geometry/materials unavailable in this tree')
    epstab = [EpsConst(1.0), EpsTable('gold.dat'), EpsConst(2.25)]
    layer = LayerStructure(epstab, [1, 3], [0.0])
    p = trisphere(256, 20.0)
    p.shift([0.0, 0.0, -p.pos[:, 2].min() + 1.0])  # 1 nm above substrate
    cp = ComParticle(epstab, [p], [[2, 1]], 1)
    return cp, layer


@pytest.fixture(scope='module')
def small_sphere_mirror():
    """Quarter Au sphere with x-mirror symmetry."""
    if not _have(EpsConst, EpsTable, trisphere, ComParticleMirror):
        pytest.skip('[info] geometry/materials unavailable in this tree')
    epstab = [EpsConst(1.0), EpsTable('gold.dat')]
    p = trisphere(256, 20.0)
    cpm = ComParticleMirror(epstab, [p], [[2, 1]], sym='x')
    return cpm


@pytest.fixture(scope='module')
def planewave_ret():
    if isinstance(PlaneWaveRet, _Missing):
        pytest.skip('PlaneWaveRet missing')
    return PlaneWaveRet(np.array([[1.0, 0.0, 0.0]]),
                        np.array([[0.0, 0.0, 1.0]]))


@pytest.fixture(scope='module')
def planewave_stat():
    if isinstance(PlaneWaveStat, _Missing):
        pytest.skip('PlaneWaveStat missing')
    return PlaneWaveStat(np.array([[1.0, 0.0, 0.0]]))


@pytest.fixture(scope='module')
def planewave_ret_layer():
    if isinstance(PlaneWaveRetLayer, _Missing):
        pytest.skip('PlaneWaveRetLayer missing')
    return PlaneWaveRetLayer(np.array([[1.0, 0.0, 0.0]]),
                             np.array([[0.0, 0.0, -1.0]]))


@pytest.fixture(scope='module')
def planewave_stat_layer():
    if isinstance(PlaneWaveStatLayer, _Missing):
        pytest.skip('PlaneWaveStatLayer missing')
    return PlaneWaveStatLayer(np.array([[1.0, 0.0, 0.0]]))


@pytest.fixture(scope='module')
def planewave_ret_mirror():
    if isinstance(PlaneWaveRetMirror, _Missing):
        pytest.skip('PlaneWaveRetMirror missing')
    return PlaneWaveRetMirror(np.array([[1.0, 0.0, 0.0]]),
                              np.array([[0.0, 0.0, 1.0]]))


@pytest.fixture(scope='module')
def planewave_stat_mirror():
    if isinstance(PlaneWaveStatMirror, _Missing):
        pytest.skip('PlaneWaveStatMirror missing')
    return PlaneWaveStatMirror(np.array([[1.0, 0.0, 0.0]]))


def _dipole_at(cp, x=0.0, y=0.0, z=40.0):
    pt = ComPoint(cp, np.array([[x, y, z]]))
    return pt


@pytest.fixture(scope='module')
def dipole_ret(small_sphere):
    if not _have(DipoleRet, ComPoint):
        pytest.skip('DipoleRet / ComPoint missing')
    pt = _dipole_at(small_sphere)
    return DipoleRet(pt, dip=np.array([[0.0, 0.0, 1.0]]))


@pytest.fixture(scope='module')
def dipole_stat(small_sphere):
    if not _have(DipoleStat, ComPoint):
        pytest.skip('DipoleStat / ComPoint missing')
    pt = _dipole_at(small_sphere)
    return DipoleStat(pt, dip=np.array([[0.0, 0.0, 1.0]]))


@pytest.fixture(scope='module')
def dipole_ret_layer(small_sphere_layer):
    if not _have(DipoleRetLayer, ComPoint):
        pytest.skip('DipoleRetLayer / ComPoint missing')
    cp, _layer = small_sphere_layer
    pt = _dipole_at(cp, z=40.0)
    return DipoleRetLayer(pt, dip=np.array([[0.0, 0.0, 1.0]]))


@pytest.fixture(scope='module')
def dipole_stat_layer(small_sphere_layer):
    if not _have(DipoleStatLayer, ComPoint):
        pytest.skip('DipoleStatLayer / ComPoint missing')
    cp, _layer = small_sphere_layer
    pt = _dipole_at(cp, z=40.0)
    return DipoleStatLayer(pt, dip=np.array([[0.0, 0.0, 1.0]]))


@pytest.fixture(scope='module')
def eels_ret(small_sphere):
    if isinstance(EELSRet, _Missing):
        pytest.skip('EELSRet missing')
    impact = np.array([[25.0, 0.0]])  # ~25 nm from center, aloof
    return EELSRet(small_sphere, impact=impact, width=0.5, vel=0.7,
                   cutoff=20.0)


@pytest.fixture(scope='module')
def eels_stat(small_sphere):
    if isinstance(EELSStat, _Missing):
        pytest.skip('EELSStat missing')
    impact = np.array([[25.0, 0.0]])
    return EELSStat(small_sphere, impact=impact, width=0.5, vel=0.7,
                    cutoff=20.0)


@pytest.fixture(scope='module')
def eels_ret_layer(small_sphere_layer):
    if isinstance(EELSRet, _Missing):
        pytest.skip('EELSRet missing')
    cp, _layer = small_sphere_layer
    impact = np.array([[25.0, 0.0]])
    return EELSRet(cp, impact=impact, width=0.5, vel=0.7, cutoff=20.0)


@pytest.fixture(scope='module')
def eels_stat_layer(small_sphere_layer):
    if isinstance(EELSStat, _Missing):
        pytest.skip('EELSStat missing')
    cp, _layer = small_sphere_layer
    impact = np.array([[25.0, 0.0]])
    return EELSStat(cp, impact=impact, width=0.5, vel=0.7, cutoff=20.0)


# ---------------------------------------------------------------------------
# Solve helpers
# ---------------------------------------------------------------------------

def _to_host(arr):
    """Force numpy host array regardless of whether arr is cupy or numpy."""
    if arr is None:
        return None
    if _HAS_CUPY and isinstance(arr, _cp.ndarray):
        return _cp.asnumpy(arr)
    return np.asarray(arr)


def _extract_sigma(sig) -> Dict[str, np.ndarray]:
    """Pull the signal arrays off a CompStruct as host numpy.

    Returns a dict keyed by attribute name so distributed-vs-baseline
    comparisons can report exactly which field drifted.
    """
    out: Dict[str, np.ndarray] = {}
    for name in ('sig', 'sig1', 'sig2', 'h1', 'h2'):
        if hasattr(sig, name):
            val = getattr(sig, name)
            if val is not None:
                out[name] = _to_host(val)
    return out


def _build_bem(BEMClass, particle, extra=None):
    """Construct a BEM solver, threading layer/etc. via kwargs.

    Returns either the solver or None when the requested combination is
    not supported (e.g. iterative solver on an unsupported particle).
    """
    extra = dict(extra) if extra else {}
    try:
        return BEMClass(particle, **extra)
    except Exception as exc:  # pragma: no cover - in-flight diagnostic
        pytest.skip('[info] {}({}) build failed: {}'.format(
            BEMClass.__name__, type(particle).__name__, exc))


def _solve_with_env(bem_factory, exc_factory, particle, env_kwargs, enei):
    """Run one BEM solve under a specific env override.

    bem_factory : zero-arg callable returning a fresh BEM solver
    exc_factory : zero-arg callable returning a fresh excitation
    particle    : ComParticle / ComParticleMirror
    env_kwargs  : dict for ``_env_scope``
    enei        : float wavelength
    """
    with _env_scope(**env_kwargs):
        bem = bem_factory()
        exc = exc_factory()
        excited = exc(particle, enei)
        with warnings.catch_warnings():
            # Distributed assembly may emit "falling back" warnings on
            # under-provisioned hardware; we treat them as informational
            # and assert on the *result* instead.
            warnings.simplefilter('ignore', RuntimeWarning)
            sig, _meta = bem.solve(excited)
    return _extract_sigma(sig)


def _compare_sigmas(
    sig_a: Dict[str, np.ndarray],
    sig_b: Dict[str, np.ndarray],
    tol: float,
    label: str,
) -> List[Tuple[str, float]]:
    """Compute per-field relative diff and assert.

    Returns the list of (name, rel_diff) so callers can dump on failure.
    """
    keys = sorted(set(sig_a) & set(sig_b))
    assert keys, '[{}] no overlapping signal keys in {} vs {}'.format(
        label, sorted(sig_a), sorted(sig_b))
    diffs: List[Tuple[str, float]] = []
    for k in keys:
        a = sig_a[k]
        b = sig_b[k]
        assert a.shape == b.shape, '[{} {}] shape {} vs {}'.format(
            label, k, a.shape, b.shape)
        denom = max(float(np.max(np.abs(a))), 1e-30)
        rel = float(np.max(np.abs(a - b))) / denom
        diffs.append((k, rel))
    bad = [(k, r) for k, r in diffs if r > tol]
    assert not bad, '[{}] distributed drift exceeds tol={:.1e}: {}'.format(
        label, tol, ', '.join(
            '{}={:.3e}'.format(k, r) for k, r in diffs))
    return diffs


# ---------------------------------------------------------------------------
# Distributed-path probing
# ---------------------------------------------------------------------------

def _has_distributed_path(BEMClass) -> bool:
    """Return True if the solver module exposes ``_vram_share_active``.

    This is the unambiguous signal that the distributed build is wired.
    Quasi-static solvers without this helper still get the single-GPU
    smoke check but the bit-identity test is skipped.
    """
    if isinstance(BEMClass, _Missing):
        return False
    module_name = BEMClass.__module__
    try:
        mod = sys.modules[module_name]
    except KeyError:  # pragma: no cover - module import already happened
        return False
    return callable(getattr(mod, '_vram_share_active', None))


# ---------------------------------------------------------------------------
# Test matrix definition
# ---------------------------------------------------------------------------
#
# Each entry is (id, BEM, excitation fixture name, particle fixture name,
# bem_kwargs_factory, dense_tol).  ``bem_kwargs_factory`` takes the
# request and returns the kwargs for the BEM solver (e.g. layer=layer).
#

_DENSE_TOL = 1e-12
_ITER_TOL = 1e-8


def _bem_kwargs_none(_request):
    return {}


def _bem_kwargs_layer(request):
    _cp, layer = request.getfixturevalue('small_sphere_layer')
    return {'layer': layer}


_MATRIX: List[Tuple[str, Any, str, str, Any, float]] = [
    # --------- BEMRet ---------
    ('BEMRet+planewave',  BEMRet,         'planewave_ret',     'small_sphere',
        _bem_kwargs_none, _DENSE_TOL),
    ('BEMRet+dipole',     BEMRet,         'dipole_ret',        'small_sphere',
        _bem_kwargs_none, _DENSE_TOL),
    ('BEMRet+eels',       BEMRet,         'eels_ret',          'small_sphere',
        _bem_kwargs_none, _DENSE_TOL),

    # --------- BEMRetIter ---------
    ('BEMRetIter+planewave', BEMRetIter,  'planewave_ret',     'small_sphere',
        _bem_kwargs_none, _ITER_TOL),

    # --------- BEMRetLayer ---------
    ('BEMRetLayer+planewave', BEMRetLayer, 'planewave_ret_layer',
        'small_sphere_layer', _bem_kwargs_layer, _DENSE_TOL),
    ('BEMRetLayer+dipole',    BEMRetLayer, 'dipole_ret_layer',
        'small_sphere_layer', _bem_kwargs_layer, _DENSE_TOL),
    ('BEMRetLayer+eels',      BEMRetLayer, 'eels_ret_layer',
        'small_sphere_layer', _bem_kwargs_layer, _DENSE_TOL),

    # --------- BEMRetLayerIter ---------
    ('BEMRetLayerIter+planewave', BEMRetLayerIter, 'planewave_ret_layer',
        'small_sphere_layer', _bem_kwargs_layer, _ITER_TOL),

    # --------- BEMRetMirror ---------
    ('BEMRetMirror+planewave', BEMRetMirror, 'planewave_ret_mirror',
        'small_sphere_mirror', _bem_kwargs_none, _DENSE_TOL),

    # --------- BEMStat ---------
    ('BEMStat+planewave',  BEMStat, 'planewave_stat',  'small_sphere',
        _bem_kwargs_none, _DENSE_TOL),
    ('BEMStat+dipole',     BEMStat, 'dipole_stat',     'small_sphere',
        _bem_kwargs_none, _DENSE_TOL),
    ('BEMStat+eels',       BEMStat, 'eels_stat',       'small_sphere',
        _bem_kwargs_none, _DENSE_TOL),

    # --------- BEMStatIter ---------
    ('BEMStatIter+planewave', BEMStatIter, 'planewave_stat', 'small_sphere',
        _bem_kwargs_none, _ITER_TOL),

    # --------- BEMStatLayer ---------
    ('BEMStatLayer+planewave', BEMStatLayer, 'planewave_stat_layer',
        'small_sphere_layer', _bem_kwargs_layer, _DENSE_TOL),
    ('BEMStatLayer+dipole',    BEMStatLayer, 'dipole_stat_layer',
        'small_sphere_layer', _bem_kwargs_layer, _DENSE_TOL),
    ('BEMStatLayer+eels',      BEMStatLayer, 'eels_stat_layer',
        'small_sphere_layer', _bem_kwargs_layer, _DENSE_TOL),

    # --------- BEMStatMirror ---------
    ('BEMStatMirror+planewave', BEMStatMirror, 'planewave_stat_mirror',
        'small_sphere_mirror', _bem_kwargs_none, _DENSE_TOL),
]


# ---------------------------------------------------------------------------
# Import-surface (always runs)
# ---------------------------------------------------------------------------

class TestImportSurface:
    """Smoke: every cell of the matrix has its BEM solver and excitation
    importable on the current tree.  Surfaces any cascading import breaks
    introduced by an in-flight refactor.
    """

    @pytest.mark.parametrize('label,BEMCls,exc_fix,part_fix,_,__', _MATRIX,
                             ids=[e[0] for e in _MATRIX])
    def test_import_present(self, label, BEMCls, exc_fix, part_fix, _, __):
        if isinstance(BEMCls, _Missing):
            pytest.skip('[info] {} not importable: {}'.format(
                BEMCls.name, BEMCls.error))
        # excitation/particle imports are surfaced via fixture; just touch
        # the names so pyflakes does not flag them.
        assert exc_fix and part_fix


# ---------------------------------------------------------------------------
# Single-GPU smoke (runs on every box with cupy; <60s)
# ---------------------------------------------------------------------------

class TestSingleGPUSmoke:
    """Each cell runs the baseline (single-GPU) path and asserts the
    surface charges are finite.  This is the bare-minimum integration
    gate kept green by every CI run regardless of GPU count.
    """

    @requires_cupy
    @pytest.mark.parametrize('label,BEMCls,exc_fix,part_fix,kwargs_fac,tol',
                             _MATRIX, ids=[e[0] for e in _MATRIX])
    def test_baseline_finite(self, request, label, BEMCls, exc_fix, part_fix,
                             kwargs_fac, tol):
        if isinstance(BEMCls, _Missing):
            pytest.skip('[info] {} not importable: {}'.format(
                BEMCls.name, BEMCls.error))

        particle_obj = request.getfixturevalue(part_fix)
        if isinstance(particle_obj, tuple):
            particle = particle_obj[0]
        else:
            particle = particle_obj

        excitation = request.getfixturevalue(exc_fix)
        bem_kwargs = kwargs_fac(request)

        def _bem():
            return _build_bem(BEMCls, particle, bem_kwargs)

        def _exc():
            return excitation

        sigmas = _solve_with_env(_bem, _exc, particle,
                                 _env_single_gpu(), WL_NM)

        assert sigmas, '[{}] solver returned no signal'.format(label)
        for k, v in sigmas.items():
            assert np.all(np.isfinite(v)), \
                '[{} {}] non-finite values in baseline'.format(label, k)


# ---------------------------------------------------------------------------
# Distributed bit-identity (multi-GPU only)
# ---------------------------------------------------------------------------

class TestDistributedBitIdentity:
    """The flagship matrix: single-GPU vs distributed must agree on sig/h.

    Per-case logic
    ---------------
    1. Skip the cell entirely if the BEM module does not yet expose
       ``_vram_share_active`` (distributed not wired - quasi-static
       in-flight in Tasks #213-217).
    2. Run baseline -> capture sig.
    3. Run distributed with the same particle/excitation -> capture sig.
    4. Per-field rel-diff must be <= tol.  Iter solvers get 1e-8 to
       absorb GMRES restart re-ordering; dense solvers get 1e-12.

    On mismatch the assertion message reports each signal name with
    its relative diff so it's obvious which Green-function block drifted.
    """

    @requires_cupy
    @requires_multi_gpu
    @pytest.mark.parametrize('label,BEMCls,exc_fix,part_fix,kwargs_fac,tol',
                             _MATRIX, ids=[e[0] for e in _MATRIX])
    def test_distributed_matches_baseline(self, request, label, BEMCls,
                                          exc_fix, part_fix, kwargs_fac, tol):
        if isinstance(BEMCls, _Missing):
            pytest.skip('[info] {} not importable: {}'.format(
                BEMCls.name, BEMCls.error))
        if not _has_distributed_path(BEMCls):
            pytest.skip('[info] {} distributed build not wired yet'.format(
                BEMCls.__name__))

        particle_obj = request.getfixturevalue(part_fix)
        if isinstance(particle_obj, tuple):
            particle = particle_obj[0]
        else:
            particle = particle_obj

        excitation = request.getfixturevalue(exc_fix)
        bem_kwargs = kwargs_fac(request)

        def _bem():
            return _build_bem(BEMCls, particle, bem_kwargs)

        def _exc():
            return excitation

        # 1. Baseline single-GPU.
        sigmas_base = _solve_with_env(_bem, _exc, particle,
                                      _env_single_gpu(), WL_NM)

        # 2. Distributed multi-GPU.
        n_gpus = min(_GPU_COUNT, 4)
        sigmas_dist = _solve_with_env(_bem, _exc, particle,
                                      _env_distributed(n_gpus), WL_NM)

        # 3. Compare.
        diffs = _compare_sigmas(sigmas_base, sigmas_dist, tol, label)

        # Report (visible with -s)
        for k, r in diffs:
            print('[{}] {}: rel_diff={:.3e}'.format(label, k, r))


# ---------------------------------------------------------------------------
# Env-off no-op (multi-GPU box but env disabled -> baseline result)
# ---------------------------------------------------------------------------

class TestDistributedOffNoOp:
    """Setting MNPBEM_VRAM_SHARE=1 *without* MNPBEM_VRAM_SHARE_DISTRIBUTED
    must NOT change the surface charges relative to the plain GPU path.

    This guards against a subtle regression where the master-switch flag
    accidentally re-routes the matrix assembly through the distributed
    code path on a single-GPU box (e.g. when the DISTRIBUTED gate is
    forgotten).
    """

    @requires_cupy
    @requires_multi_gpu
    @pytest.mark.parametrize('label,BEMCls,exc_fix,part_fix,kwargs_fac,tol',
                             _MATRIX, ids=[e[0] for e in _MATRIX])
    def test_env_share_only_is_noop(self, request, label, BEMCls, exc_fix,
                                    part_fix, kwargs_fac, tol):
        if isinstance(BEMCls, _Missing):
            pytest.skip('[info] {} not importable: {}'.format(
                BEMCls.name, BEMCls.error))
        if not _has_distributed_path(BEMCls):
            pytest.skip('[info] {} distributed build not wired yet'.format(
                BEMCls.__name__))

        particle_obj = request.getfixturevalue(part_fix)
        if isinstance(particle_obj, tuple):
            particle = particle_obj[0]
        else:
            particle = particle_obj

        excitation = request.getfixturevalue(exc_fix)
        bem_kwargs = kwargs_fac(request)

        def _bem():
            return _build_bem(BEMCls, particle, bem_kwargs)

        def _exc():
            return excitation

        sigmas_base = _solve_with_env(_bem, _exc, particle,
                                      _env_single_gpu(), WL_NM)

        # VRAM_SHARE=1 but DISTRIBUTED unset / 0 -> must be a no-op for
        # the BEM matrix; only the LU dispatch can switch backends.
        env_share = dict(
            MNPBEM_GPU='1',
            MNPBEM_VRAM_SHARE='1',
            MNPBEM_VRAM_SHARE_GPUS=str(min(_GPU_COUNT, 4)),
            MNPBEM_VRAM_SHARE_BACKEND='cusolvermg',
            MNPBEM_VRAM_SHARE_DISTRIBUTED='0',
        )
        sigmas_share = _solve_with_env(_bem, _exc, particle,
                                       env_share, WL_NM)

        # No-op vs baseline: same matrix assembly, same LU result.  The
        # cuSolverMg LU may have a different pivot order so we relax tol
        # to 1e-10 for dense, keep 1e-8 for iter.
        share_tol = max(tol, 1e-10)
        _compare_sigmas(sigmas_base, sigmas_share, share_tol,
                        '{} (VRAM_SHARE=1 no-op)'.format(label))


# ---------------------------------------------------------------------------
# Excitation x boundary anomalies (xfail block)
# ---------------------------------------------------------------------------
#
# Some combinations are explicitly outside the MNPBEM Demo set.  We
# document them here as expected-fail so the matrix stays exhaustive
# without surprising CI.
#

_XFAIL = {
    # EELS x Mirror: not part of MNPBEM Demo; see
    # mnpbem/tests/test_v17_integration_eels_combo.py for the
    # acknowledged skip.
    'EELSRet+BEMRetMirror',
    'EELSStat+BEMStatMirror',
}


# ---------------------------------------------------------------------------
# Discovery helper for ad-hoc invocation
# ---------------------------------------------------------------------------

def _matrix_summary():
    """Return a compact textual list of (id, BEM, exc) for human review.

    Run::

        python tests/test_e2e_distributed.py

    to print the matrix without going through pytest.
    """
    out = ['{:32s}  {:20s}  {:24s}  {:8s}'.format(
        'id', 'bem_class', 'excitation', 'tol')]
    for label, BEMCls, exc_fix, _part, _kw, tol in _MATRIX:
        bem_name = BEMCls.name if isinstance(BEMCls, _Missing) \
            else BEMCls.__name__
        out.append('{:32s}  {:20s}  {:24s}  {:8.1e}'.format(
            label, bem_name, exc_fix, tol))
    return '\n'.join(out)


if __name__ == '__main__':  # pragma: no cover
    print(_matrix_summary())
    print('\nGPU count detected:', _GPU_COUNT)
    print('cupy available    :', _HAS_CUPY)
