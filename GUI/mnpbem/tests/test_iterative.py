"""
Comprehensive tests for Iterative BEM module (MATLAB -> Python conversion).

Tests for:
  - BEMIter: construction, options parsing, _iter_solve iteration,
             _set_iter/_set_stat, info/hinfo statistics, _print_stat
  - BEMStatIter: construction, _init_green, _init_matrices, solve (mldivide),
                 potential, field, clear, _afun/_mfun callbacks
  - BEMRetIter: construction, _init_green, _init_matrices, _init_precond,
                _excitation, solve (mldivide), potential, field, clear,
                _pack/_unpack, _inner/_outer, _afun/_mfun callbacks
  - BEMRetLayerIter: construction, _init_green, _init_matrices, _init_precond,
                     _excitation, solve (mldivide), potential, field, clear,
                     _pack/_unpack, _inner/_outer, _decorate_gamma

MATLAB reference:
  BEM/@bemiter, BEM/@bemstatiter, BEM/@bemretiter, BEM/@bemretlayeriter

Discrepancies between MATLAB and Python noted in individual test docstrings.
"""

import sys
import os
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from mnpbem.bem.bem_iter import BEMIter
from mnpbem.greenfun import CompStruct


# ---------------------------------------------------------------------------
# Mock particle objects
# ---------------------------------------------------------------------------

class MockParticle(object):
    """
    Simplified particle mock for iterative BEM testing.

    Provides n faces (octahedron-like), nvec, eps1/eps2 methods, pos, area.
    """

    def __init__(self, n=8, eps1_val=1.0, eps2_val=-10.0+1.0j):
        self.n = n
        # Random but reproducible normal vectors (unit-normalized)
        rng = np.random.RandomState(42)
        raw = rng.randn(n, 3)
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        self.nvec = raw / norms

        # Face positions
        self.pos = rng.randn(n, 3) * 5.0

        # Face areas
        self.area = np.ones(n) * 10.0

        # Store dielectric values
        self._eps1_val = eps1_val
        self._eps2_val = eps2_val

        # Dummy verts/faces for interp/deriv calls
        self.verts = rng.randn(n + 2, 3) * 5.0
        self.faces = np.zeros((n, 4))
        for i in range(n):
            self.faces[i, :3] = [(i % (n + 2)),
                                 ((i + 1) % (n + 2)),
                                 ((i + 2) % (n + 2))]
            self.faces[i, 3] = np.nan

    def eps1(self, enei):
        """Return inside dielectric function (scalar)."""
        return self._eps1_val

    def eps2(self, enei):
        """Return outside dielectric function (scalar)."""
        return self._eps2_val

    def interp(self, values):
        """Mock interpolation -- return values unchanged."""
        return values

    def deriv(self, values):
        """
        Mock derivative computation.
        Returns (d1, d2, t1, t2) where t1, t2 are tangent vectors.
        """
        n = self.n
        if values.ndim == 1:
            d1 = values.copy()
            d2 = values.copy()
        elif values.ndim == 2:
            d1 = values.copy()
            d2 = values.copy()
        else:
            d1 = values.copy()
            d2 = values.copy()

        # Tangent vectors
        rng = np.random.RandomState(99)
        t1 = rng.randn(n, 3)
        t2 = rng.randn(n, 3)
        return d1, d2, t1, t2


class MockGreenFunction(object):
    """
    Mock Green function object for static BEM testing.
    Provides G, F, H1, H2 matrices and potential method.
    """

    def __init__(self, n):
        rng = np.random.RandomState(123)
        # Symmetric Green function matrices, well-conditioned
        G = rng.randn(n, n) + 1j * rng.randn(n, n)
        G = (G + G.T) / 2
        G += n * np.eye(n)
        self.G = G
        self.F = rng.randn(n, n)
        self.H1 = rng.randn(n, n) + 1j * rng.randn(n, n)
        self.H2 = rng.randn(n, n) + 1j * rng.randn(n, n)

    def potential(self, sig, inout):
        n = self.G.shape[0]
        if inout == 1:
            return CompStruct(sig.p, sig.enei,
                              phi1=np.ones(n, dtype=complex),
                              phi1p=np.ones(n, dtype=complex))
        else:
            return CompStruct(sig.p, sig.enei,
                              phi2=np.ones(n, dtype=complex),
                              phi2p=np.ones(n, dtype=complex))


class MockGreenFunctionRet(object):
    """
    Mock Green function for retarded solver.
    Provides eval() for G/H1/H2 matrices and potential method.
    """

    def __init__(self, n):
        self._n = n
        rng = np.random.RandomState(123)
        self.G = rng.randn(n, n) + 1j * rng.randn(n, n)
        self.G = (self.G + self.G.T) / 2 + n * np.eye(n)

    def eval(self, i, j, field, enei):
        n = self._n
        rng = np.random.RandomState(hash((i, j, field)) % (2**31))
        mat = rng.randn(n, n) + 1j * rng.randn(n, n)
        mat = (mat + mat.T) / 2
        mat += n * np.eye(n)
        return mat

    def potential(self, sig, inout):
        n = self._n
        if inout == 1:
            return CompStruct(sig.p, sig.enei,
                              phi1=np.ones(n, dtype=complex),
                              phi1p=np.ones(n, dtype=complex),
                              a1=np.ones((n, 3), dtype=complex),
                              a1p=np.ones((n, 3), dtype=complex))
        else:
            return CompStruct(sig.p, sig.enei,
                              phi2=np.ones(n, dtype=complex),
                              phi2p=np.ones(n, dtype=complex),
                              a2=np.ones((n, 3), dtype=complex),
                              a2p=np.ones((n, 3), dtype=complex))


class MockLayerGreenFunction(MockGreenFunctionRet):
    """
    Mock Green function for retarded layer solver.
    eval() returns structured objects for outer-surface pairs.
    """

    def __init__(self, n):
        super(MockLayerGreenFunction, self).__init__(n)

    def eval(self, i, j, field, enei):
        n = self._n
        rng = np.random.RandomState(hash((i, j, field)) % (2**31))
        mat = rng.randn(n, n) + 1j * rng.randn(n, n)
        mat = (mat + mat.T) / 2
        mat += n * np.eye(n)

        # For (1,1) = outer-outer, return object with ss, hh, p, sh, hs
        if i == 1 and j == 1:
            result = _MockLayerResult()
            result.ss = mat.copy()
            result.hh = mat.copy() * 0.9
            result.p = mat.copy() * 1.1
            result.sh = rng.randn(n, n) * 0.1 + 1j * rng.randn(n, n) * 0.1
            result.hs = rng.randn(n, n) * 0.1 + 1j * rng.randn(n, n) * 0.1
            return result
        else:
            return mat


class _MockLayerResult(object):
    """Helper object with ss, hh, p, sh, hs attributes."""
    pass


# ---------------------------------------------------------------------------
# Helper: create BEM objects without triggering _init_green
# ---------------------------------------------------------------------------

def _make_stat_iter(particle, green_func=None, **options):
    """
    Create a BEMStatIter with mock Green function, bypassing _init_green.
    """
    from mnpbem.bem.bem_stat_iter import BEMStatIter

    class _TestStatIter(BEMStatIter):
        def _init_green(self, p, **opts):
            pass

    defaults = dict(solver='gmres', tol=1e-4, maxit=200, precond='hmat')
    defaults.update(options)
    bem = _TestStatIter(particle, **defaults)

    if green_func is None:
        green_func = MockGreenFunction(particle.n)
    bem._g = green_func
    bem.F = green_func.F
    return bem


def _make_ret_iter(particle, green_func=None, **options):
    """
    Create a BEMRetIter with mock Green function, bypassing _init_green.
    """
    from mnpbem.bem.bem_ret_iter import BEMRetIter

    class _TestRetIter(BEMRetIter):
        def _init_green(self, p, **opts):
            pass

    defaults = dict(solver='gmres', tol=1e-4, maxit=200, precond='hmat')
    defaults.update(options)
    bem = _TestRetIter(particle, **defaults)

    if green_func is None:
        green_func = MockGreenFunctionRet(particle.n)
    bem.g = green_func
    return bem


def _make_ret_layer_iter(particle, green_func=None, **options):
    """
    Create a BEMRetLayerIter with mock Green function, bypassing _init_green.
    """
    from mnpbem.bem.bem_ret_layer_iter import BEMRetLayerIter

    class _TestRetLayerIter(BEMRetLayerIter):
        def _init_green(self, p, **opts):
            pass

    defaults = dict(solver='gmres', tol=1e-4, maxit=200, precond='hmat')
    defaults.update(options)
    bem = _TestRetLayerIter(particle, **defaults)

    if green_func is None:
        green_func = MockLayerGreenFunction(particle.n)
    bem.g = green_func
    return bem


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def particle():
    return MockParticle(n=8)


# ===========================================================================
# BEMIter tests
# ===========================================================================

class TestBEMIter(object):
    """Tests for BEMIter base class (MATLAB: @bemiter)."""

    def test_default_construction(self):
        """BEMIter default parameters match MATLAB @bemiter defaults."""
        it = BEMIter()
        assert it.solver == 'gmres'
        assert it.tol == 1e-4
        assert it.maxit == 200
        assert it.restart is None
        assert it.precond == 'hmat'
        assert it.output == 0
        assert it._flag is None
        assert it._relres is None
        assert it._iter is None

    def test_custom_construction(self):
        """BEMIter accepts custom parameters (MATLAB: @bemiter/init.m)."""
        it = BEMIter(solver='cgs', tol=1e-8, maxit=500, precond='full', output=1)
        assert it.solver == 'cgs'
        assert it.tol == 1e-8
        assert it.maxit == 500
        assert it.precond == 'full'
        assert it.output == 1

    def test_bicgstab_construction(self):
        """BEMIter supports bicgstab solver."""
        it = BEMIter(solver='bicgstab', tol=1e-6, maxit=150)
        assert it.solver == 'bicgstab'

    def test_options_defaults(self):
        """
        BEMIter.options() returns correct default dict.
        MATLAB: bemiter.options() returns struct with solver, tol, maxit, etc.
        """
        op = BEMIter.options()
        assert op['solver'] == 'gmres'
        assert op['tol'] == 1e-6
        assert op['maxit'] == 100
        assert op['restart'] is None
        assert op['precond'] == 'hmat'
        assert op['output'] == 0
        assert op['cleaf'] == 200
        assert op['htol'] == 1e-6
        assert op['kmax'] == [4, 100]
        assert callable(op['fadmiss'])

    def test_options_custom(self):
        """BEMIter.options() merges user overrides (MATLAB: bemiter/options.m)."""
        op = BEMIter.options(solver='cgs', tol=1e-8, maxit=500)
        assert op['solver'] == 'cgs'
        assert op['tol'] == 1e-8
        assert op['maxit'] == 500
        assert op['precond'] == 'hmat'

    def test_options_fadmiss(self):
        """Test default fadmiss function from options (MATLAB: fadmiss)."""
        op = BEMIter.options()
        assert op['fadmiss'](1, 2, 10) is True
        assert op['fadmiss'](5, 5, 10) is False

    def test_set_iter(self):
        """_set_iter accumulates solver statistics (MATLAB: bemiter/setiter.m)."""
        it = BEMIter()
        it._set_iter(0, 1e-6, np.array([10, 3]))
        assert it._flag == [0]
        assert it._relres == [1e-6]
        assert len(it._iter) == 1
        np.testing.assert_array_equal(it._iter[0], np.array([10, 3]))

        it._set_iter(1, 1e-3, np.array([20, 5]))
        assert it._flag == [0, 1]
        assert it._relres == [1e-6, 1e-3]
        assert len(it._iter) == 2

    def test_info(self):
        """info() returns (flag, relres, iter) tuples (MATLAB: bemiter/info.m)."""
        it = BEMIter()
        it._set_iter(0, 1e-7, np.array([50, 10]))
        it._set_iter(1, 2e-4, np.array([100, 20]))
        flag, relres, iters = it.info()
        assert flag == [0, 1]
        assert relres == [1e-7, 2e-4]
        assert len(iters) == 2

    def test_set_stat(self):
        """_set_stat accumulates H-matrix compression stats (MATLAB: bemiter/setstat.m)."""
        it = BEMIter()
        mock_hmat = MagicMock()
        mock_hmat.compression.return_value = 0.15
        it._set_stat('G', mock_hmat)
        assert 'G' in it._stat['compression']
        assert it._stat['compression']['G'] == [0.15]

        mock_hmat.compression.return_value = 0.12
        it._set_stat('G', mock_hmat)
        assert it._stat['compression']['G'] == [0.15, 0.12]

    def test_set_stat_no_compression_attr(self):
        """_set_stat handles objects without compression attribute."""
        it = BEMIter()
        mock_hmat = MagicMock(spec=[])
        it._set_stat('F', mock_hmat)
        assert 'F' in it._stat['compression']
        assert it._stat['compression']['F'] == []

    def test_hinfo_empty(self):
        """hinfo() does nothing when no stats collected (MATLAB: bemiter/hinfo.m)."""
        it = BEMIter()
        it.hinfo()

    def test_hinfo_with_stats(self, capsys):
        """hinfo() prints compression info (MATLAB: bemiter/hinfo.m)."""
        it = BEMIter()
        mock_hmat = MagicMock()
        mock_hmat.compression.return_value = 0.25
        it._set_stat('G', mock_hmat)
        it._set_stat('G1', mock_hmat)
        mock_hmat2 = MagicMock()
        mock_hmat2.compression.return_value = 0.15
        it._set_stat('mat', mock_hmat2)

        it.hinfo()
        captured = capsys.readouterr()
        assert 'Compression Green functions' in captured.out
        assert 'Compression auxiliary matrices' in captured.out

    def test_iter_solve_gmres_identity(self):
        """
        _iter_solve with identity-like system converges.
        MATLAB: bemiter/solve.m with gmres.
        """
        it = BEMIter(solver='gmres', tol=1e-10, maxit=50, precond=None)
        n = 10
        A = np.eye(n, dtype=complex)
        b = np.ones(n, dtype=complex)
        afun = lambda x: A @ x
        x, _ = it._iter_solve(None, b, afun, None)
        np.testing.assert_allclose(x, b, atol=1e-8)

    def test_iter_solve_cgs(self):
        """_iter_solve with cgs solver (MATLAB: solve.m, case 'cgs')."""
        it = BEMIter(solver='cgs', tol=1e-10, maxit=50, precond=None)
        n = 10
        rng = np.random.RandomState(7)
        A = rng.randn(n, n) + 1j * rng.randn(n, n)
        A = A @ A.conj().T + n * np.eye(n)
        b = rng.randn(n) + 1j * rng.randn(n)
        afun = lambda x: A @ x
        x, _ = it._iter_solve(None, b, afun, None)
        np.testing.assert_allclose(A @ x, b, atol=1e-6)

    def test_iter_solve_bicgstab(self):
        """_iter_solve with bicgstab solver (MATLAB: solve.m, case 'bicgstab')."""
        it = BEMIter(solver='bicgstab', tol=1e-10, maxit=100, precond=None)
        n = 10
        rng = np.random.RandomState(13)
        A = rng.randn(n, n) + 1j * rng.randn(n, n)
        A = A @ A.conj().T + n * np.eye(n)
        b = rng.randn(n) + 1j * rng.randn(n)
        afun = lambda x: A @ x
        x, _ = it._iter_solve(None, b, afun, None)
        np.testing.assert_allclose(A @ x, b, atol=1e-6)

    def test_iter_solve_with_preconditioner(self):
        """_iter_solve uses preconditioner when provided (MATLAB: solve.m M argument)."""
        it = BEMIter(solver='gmres', tol=1e-10, maxit=50, precond='hmat')
        n = 10
        rng = np.random.RandomState(11)
        A = rng.randn(n, n) + 1j * rng.randn(n, n)
        A = A @ A.conj().T + n * np.eye(n)
        Ainv = np.linalg.inv(A)
        b = rng.randn(n) + 1j * rng.randn(n)
        afun = lambda x: A @ x
        mfun = lambda x: Ainv @ x
        x, _ = it._iter_solve(None, b, afun, mfun)
        np.testing.assert_allclose(A @ x, b, atol=1e-8)

    def test_iter_solve_maxit_zero_uses_precond(self):
        """
        When maxit=0, _iter_solve uses only the preconditioner.
        MATLAB: solve.m lines 15-19.
        """
        it = BEMIter(solver='gmres', tol=1e-10, maxit=0, precond='hmat')
        n = 5
        b = np.ones(n, dtype=complex)
        mfun = lambda x: x * 2.0
        x, _ = it._iter_solve(None, b, None, mfun)
        np.testing.assert_allclose(x, b * 2.0, atol=1e-14)

    def test_iter_solve_unknown_solver_raises(self):
        """Unknown solver raises ValueError (MATLAB: solve.m 'otherwise' case)."""
        it = BEMIter(solver='unknown_solver', tol=1e-4, maxit=50)
        b = np.ones(5, dtype=complex)
        with pytest.raises(ValueError, match='iterative solver not known'):
            it._iter_solve(None, b, lambda x: x, None)

    def test_iter_solve_records_statistics(self):
        """_iter_solve records flag, relres, iter (MATLAB: solve.m line 42)."""
        it = BEMIter(solver='gmres', tol=1e-10, maxit=10, precond=None)
        n = 5
        b = np.ones(n, dtype=complex)
        afun = lambda x: x
        it._iter_solve(None, b, afun, None)
        assert it._flag is not None
        assert len(it._flag) == 1
        assert len(it._relres) == 1

    def test_iter_solve_prints_with_output(self, capsys):
        """_iter_solve prints statistics when output=1 (MATLAB: solve.m line 44)."""
        it = BEMIter(solver='gmres', tol=1e-10, maxit=10, precond=None, output=1)
        n = 5
        b = np.ones(n, dtype=complex)
        afun = lambda x: x
        it._iter_solve(None, b, afun, None)
        captured = capsys.readouterr()
        assert 'gmres' in captured.out

    def test_print_stat_cgs(self, capsys):
        """_print_stat for cgs format (MATLAB: bemiter/printstat.m case 'cgs')."""
        it = BEMIter(solver='cgs', maxit=100)
        it._print_stat(0, 1e-6, np.array([50, 0]))
        captured = capsys.readouterr()
        assert 'cgs(100)' in captured.out

    def test_print_stat_bicgstab(self, capsys):
        """_print_stat for bicgstab (MATLAB: bemiter/printstat.m case 'bicgstab')."""
        it = BEMIter(solver='bicgstab', maxit=200)
        it._print_stat(0, 1e-5, np.array([75, 0]))
        captured = capsys.readouterr()
        assert 'bicgstab(200)' in captured.out

    def test_print_stat_gmres(self, capsys):
        """_print_stat for gmres (MATLAB: bemiter/printstat.m case 'gmres')."""
        it = BEMIter(solver='gmres', maxit=100)
        it._print_stat(0, 1e-7, np.array([5, 30]))
        captured = capsys.readouterr()
        assert 'gmres(100)' in captured.out

    def test_repr(self):
        """__repr__ returns useful string."""
        it = BEMIter(solver='cgs', tol=1e-6, maxit=100, precond='full')
        r = repr(it)
        assert 'BEMIter' in r
        assert 'cgs' in r


# ===========================================================================
# BEMStatIter tests
# ===========================================================================

class TestBEMStatIter(object):
    """
    Tests for BEMStatIter (MATLAB: @bemstatiter).

    Discrepancy note: MATLAB bemstatiter/mtimes.m adds potential(sig,1) + potential(sig,2)
    into a single CompStruct. Python stores phi1, phi1p, phi2, phi2p separately.
    This is an intentional API difference in the Python conversion.
    """

    def test_construction(self, particle):
        """
        BEMStatIter construction sets particle and solver params.
        MATLAB: bemstatiter(p, op).
        """
        bem = _make_stat_iter(particle, solver='gmres', tol=1e-6)
        assert bem.p is particle
        assert bem.solver == 'gmres'
        assert bem.tol == 1e-6
        assert bem.enei is None
        assert bem.F is not None

    def test_construction_inherits_bemiter(self):
        """BEMStatIter inherits from BEMIter (MATLAB: bemstatiter < bemiter)."""
        from mnpbem.bem.bem_stat_iter import BEMStatIter
        assert issubclass(BEMStatIter, BEMIter)

    def test_init_matrices_lambda(self, particle):
        """
        _init_matrices computes lambda = 2*pi*(eps1+eps2)/(eps1-eps2).
        MATLAB: bemstatiter/private/initmat.m, Eq. (23) from Garcia de Abajo.
        """
        bem = _make_stat_iter(particle, solver='gmres')
        bem._init_matrices(500.0)

        eps1 = particle.eps1(500.0)
        eps2 = particle.eps2(500.0)
        expected_lambda = 2 * np.pi * (eps1 + eps2) / (eps1 - eps2)
        np.testing.assert_allclose(bem._lambda, expected_lambda)

    def test_init_matrices_caching(self, particle):
        """
        _init_matrices skips recomputation if enei unchanged.
        MATLAB: initmat.m line 5: if isempty(obj.enei) || obj.enei ~= enei.
        """
        bem = _make_stat_iter(particle, solver='gmres')
        bem._init_matrices(500.0)
        lambda1 = bem._lambda
        mat1 = bem._mat_lu

        bem._init_matrices(500.0)
        assert bem._lambda is lambda1
        assert bem._mat_lu is mat1

    def test_init_matrices_precond_hmat(self, particle):
        """
        With precond='hmat', _init_matrices computes inv(-Lambda - F).
        MATLAB: initmat.m, case 'hmat': obj.mat = lu(-lambda - F).
        """
        bem = _make_stat_iter(particle, solver='gmres', precond='hmat')
        bem._init_matrices(500.0)
        assert bem._mat_lu is not None
        # dispatch package format: (tag, lu_matrix, piv)
        assert bem._mat_lu[0] in ('cpu', 'gpu')
        assert bem._mat_lu[1].shape == (particle.n, particle.n)

    def test_init_matrices_precond_full(self, particle):
        """
        With precond='full', computes inv(-Lambda - F).
        MATLAB: initmat.m, case 'full': obj.mat = inv(-lambda - full(F)).
        """
        bem = _make_stat_iter(particle, solver='gmres', precond='full')
        bem._init_matrices(500.0)
        assert bem._mat_lu is not None

    def test_init_matrices_precond_unknown_raises(self, particle):
        """
        Unknown preconditioner raises ValueError.
        MATLAB: initmat.m, otherwise: error('preconditioner not known').
        """
        bem = _make_stat_iter(particle, solver='gmres', precond='unknown')
        with pytest.raises(ValueError, match='preconditioner not known'):
            bem._init_matrices(500.0)

    def test_init_matrices_new_enei(self, particle):
        """
        _init_matrices recomputes when enei changes.
        """
        bem = _make_stat_iter(particle, solver='gmres')
        bem._init_matrices(500.0)
        lambda1 = bem._lambda

        bem._init_matrices(600.0)
        assert bem.enei == 600.0
        # lambda should differ for different enei (since eps depends on enei)
        # But our mock returns constant eps, so lambda will be same
        # Still, the method ran to completion
        assert bem._lambda is not None

    def test_afun(self, particle):
        """
        _afun computes -(F*vec + vec*lambda).
        MATLAB: bemstatiter/private/afun.m.
        """
        bem = _make_stat_iter(particle, solver='gmres', precond='hmat')
        bem._init_matrices(500.0)

        n = particle.n
        vec = np.ones(n, dtype=complex)
        result = bem._afun(vec)
        assert result.shape == (n,)

        vec_2d = vec.reshape(n, -1)
        # Handle scalar lambda (MockParticle returns scalar eps)
        lam = bem._lambda
        if np.isscalar(lam) or (isinstance(lam, np.ndarray) and lam.ndim == 0):
            expected = -(bem.F @ vec_2d + vec_2d * lam)
        else:
            expected = -(bem.F @ vec_2d + vec_2d * lam[:, np.newaxis])
        np.testing.assert_allclose(result, expected.reshape(-1))

    def test_mfun_hmat(self, particle):
        """
        _mfun applies preconditioner: mat @ vec.
        MATLAB: bemstatiter/private/mfun.m, case 'hmat': vec = solve(obj.mat, vec).
        """
        bem = _make_stat_iter(particle, solver='gmres', precond='hmat')
        bem._init_matrices(500.0)

        n = particle.n
        vec = np.ones(n, dtype=complex)
        result = bem._mfun(vec)
        assert result.shape == (n,)
        # result = lu_solve_dispatch(mat_lu, vec)
        from mnpbem.utils.gpu import lu_solve_dispatch
        expected = lu_solve_dispatch(bem._mat_lu, vec.reshape(n, -1))
        np.testing.assert_allclose(result, expected.reshape(-1))

    def test_mfun_full(self, particle):
        """_mfun with precond='full' (MATLAB: mfun.m, case 'full')."""
        bem = _make_stat_iter(particle, solver='gmres', precond='full')
        bem._init_matrices(500.0)

        n = particle.n
        vec = np.ones(n, dtype=complex)
        result = bem._mfun(vec)
        assert result.shape == (n,)

    def test_mfun_no_precond(self, particle):
        """_mfun without preconditioner returns vec unchanged."""
        bem = _make_stat_iter(particle, solver='gmres', precond=None)
        bem._init_matrices(500.0)

        n = particle.n
        vec = np.ones(n, dtype=complex)
        # precond is None, so _mfun should not be called in solve
        # but if called directly, it goes to else branch
        result = bem._mfun(vec)
        np.testing.assert_allclose(result, vec)

    def test_solve(self, particle):
        """
        solve() performs iterative solution.
        MATLAB: bemstatiter/solve.m.
        """
        bem = _make_stat_iter(particle, solver='gmres', tol=1e-4, maxit=50,
                              precond='hmat')

        n = particle.n
        exc = CompStruct(particle, 500.0,
                         phip=np.ones(n, dtype=complex))
        sig, bem_out = bem.solve(exc)
        assert hasattr(sig, 'sig')
        assert sig.sig.shape == (n,)
        assert sig.enei == 500.0

    def test_truediv_calls_solve(self, particle):
        """
        __truediv__ delegates to solve.
        MATLAB: bemstatiter/mldivide.m.
        """
        bem = _make_stat_iter(particle, solver='gmres', tol=1e-4, maxit=50,
                              precond='hmat')
        n = particle.n
        exc = CompStruct(particle, 500.0,
                         phip=np.ones(n, dtype=complex))
        sig, _ = bem / exc
        assert hasattr(sig, 'sig')

    def test_potential(self, particle):
        """
        potential() delegates to Green function.
        MATLAB: bemstatiter/potential.m.
        """
        bem = _make_stat_iter(particle, solver='gmres')
        n = particle.n
        sig = CompStruct(particle, 500.0,
                         sig=np.ones(n, dtype=complex))
        pot = bem.potential(sig, 2)
        assert hasattr(pot, 'phi2')

    def test_potential_inout1(self, particle):
        """potential() with inout=1 returns inside potential."""
        bem = _make_stat_iter(particle, solver='gmres')
        n = particle.n
        sig = CompStruct(particle, 500.0,
                         sig=np.ones(n, dtype=complex))
        pot = bem.potential(sig, 1)
        assert hasattr(pot, 'phi1')

    def test_clear(self, particle):
        """
        clear() resets _mat to None.
        MATLAB: bemstatiter/clear.m: obj.mat = [].
        """
        bem = _make_stat_iter(particle, solver='gmres', precond='hmat')
        bem._init_matrices(500.0)
        assert bem._mat_lu is not None
        bem.clear()
        assert bem._mat_lu is None

    def test_call_init_matrices(self, particle):
        """
        __call__ triggers _init_matrices.
        MATLAB: bemstatiter/subsref.m with '()'.
        """
        bem = _make_stat_iter(particle, solver='gmres', precond='hmat')
        result = bem(500.0)
        assert result.enei == 500.0

    def test_repr(self, particle):
        """__repr__ shows useful info."""
        bem = _make_stat_iter(particle, solver='gmres')
        r = repr(bem)
        assert 'BEMStatIter' in r
        assert 'gmres' in r

    def test_repr_initialized(self, particle):
        """__repr__ shows enei when initialized."""
        bem = _make_stat_iter(particle, solver='gmres')
        bem._init_matrices(500.0)
        r = repr(bem)
        assert '500.0' in r

    def test_mtimes(self, particle):
        """
        __mul__ computes potentials at both inside and outside.
        MATLAB: bemstatiter/mtimes.m.

        Discrepancy: MATLAB adds potential(sig,1)+potential(sig,2).
        Python stores phi1, phi1p, phi2, phi2p separately.
        """
        bem = _make_stat_iter(particle, solver='gmres')
        n = particle.n
        sig = CompStruct(particle, 500.0,
                         sig=np.ones(n, dtype=complex))
        phi = bem * sig
        assert hasattr(phi, 'phi1')
        assert hasattr(phi, 'phi2')

    def test_name_property(self):
        """BEMStatIter.name is 'bemsolver' (MATLAB: Constant property)."""
        from mnpbem.bem.bem_stat_iter import BEMStatIter
        assert BEMStatIter.name == 'bemsolver'

    def test_needs_property(self):
        """BEMStatIter.needs has sim='stat' (MATLAB: Constant property)."""
        from mnpbem.bem.bem_stat_iter import BEMStatIter
        assert BEMStatIter.needs['sim'] == 'stat'


# ===========================================================================
# BEMRetIter tests
# ===========================================================================

class TestBEMRetIter(object):
    """
    Tests for BEMRetIter (MATLAB: @bemretiter).

    Discrepancy note: MATLAB bemretiter/mtimes.m adds potential(sig,1)+potential(sig,2).
    Python stores all potential fields separately.
    """

    def test_construction(self, particle):
        """
        BEMRetIter construction.
        MATLAB: bemretiter(p, op).
        """
        bem = _make_ret_iter(particle, solver='gmres', tol=1e-6, precond=None)
        assert bem.p is particle
        assert bem.solver == 'gmres'
        assert bem.enei is None
        assert bem.g is not None
        assert bem._nvec is not None

    def test_init_matrices_wavenumber(self, particle):
        """
        _init_matrices sets k = 2*pi/enei.
        MATLAB: bemretiter/private/initmat.m: obj.k = 2*pi/enei.
        """
        bem = _make_ret_iter(particle, solver='gmres', precond=None)
        bem._init_matrices(500.0)
        assert bem._k == pytest.approx(2 * np.pi / 500.0)

    def test_init_matrices_dielectric(self, particle):
        """_init_matrices stores eps1, eps2 from particle."""
        bem = _make_ret_iter(particle, solver='gmres', precond=None)
        bem._init_matrices(500.0)
        assert bem._eps1 == particle.eps1(500.0)
        assert bem._eps2 == particle.eps2(500.0)

    def test_init_matrices_caching(self, particle):
        """_init_matrices skips recomputation if enei unchanged."""
        bem = _make_ret_iter(particle, solver='gmres', precond=None)
        bem._init_matrices(500.0)
        G1_ref = bem._G1
        bem._init_matrices(500.0)
        assert bem._G1 is G1_ref

    def test_init_matrices_green_shapes(self, particle):
        """
        _init_matrices evaluates G1, G2, H1, H2 with correct shapes.
        MATLAB: initmat.m lines 18-22.
        """
        bem = _make_ret_iter(particle, solver='gmres', precond=None)
        bem._init_matrices(500.0)
        n = particle.n
        assert bem._G1.shape == (n, n)
        assert bem._G2.shape == (n, n)
        assert bem._H1.shape == (n, n)
        assert bem._H2.shape == (n, n)

    def test_init_matrices_green_difference(self, particle):
        """
        G1 = g{1,1}.G(enei) - g{2,1}.G(enei), etc.
        MATLAB: initmat.m.
        """
        gf = MockGreenFunctionRet(particle.n)
        bem = _make_ret_iter(particle, green_func=gf, solver='gmres', precond=None)
        bem._init_matrices(500.0)

        G11 = gf.eval(0, 0, 'G', 500.0)
        G21 = gf.eval(1, 0, 'G', 500.0)
        expected_G1 = G11 - G21
        np.testing.assert_allclose(bem._G1, expected_G1)

    def test_init_precond(self, particle):
        """
        _init_precond builds preconditioner matrices.
        MATLAB: bemretiter/private/initprecond.m.
        """
        bem = _make_ret_iter(particle, solver='gmres', precond='full')
        bem._init_matrices(500.0)
        assert bem._sav is not None
        assert 'G1_lu' in bem._sav
        assert 'G2_lu' in bem._sav
        assert 'Sigma1' in bem._sav
        assert 'Delta_lu' in bem._sav
        assert 'Sigma_lu' in bem._sav
        assert 'k' in bem._sav
        assert 'nvec' in bem._sav
        assert 'eps1' in bem._sav
        assert 'eps2' in bem._sav

    def test_init_precond_sigma1_shape(self, particle):
        """Sigma1 has correct shape (n, n)."""
        bem = _make_ret_iter(particle, solver='gmres', precond='full')
        bem._init_matrices(500.0)
        n = particle.n
        assert bem._sav['Sigma1'].shape == (n, n)

    def test_decorate_deltai(self):
        """
        _decorate_deltai computes sum_i nvec_i * Deltai * nvec_i.
        MATLAB: initprecond.m, fun(Deltai, nvec) using all 3 components.
        """
        from mnpbem.bem.bem_ret_iter import BEMRetIter
        n = 4
        rng = np.random.RandomState(77)
        Deltai = rng.randn(n, n) + 1j * rng.randn(n, n)
        nvec = rng.randn(n, 3)
        nvec = nvec / np.linalg.norm(nvec, axis=1, keepdims=True)

        result = BEMRetIter._decorate_deltai(Deltai, nvec)

        expected = np.zeros((n, n), dtype=Deltai.dtype)
        for i in range(3):
            nvec_i = np.diag(nvec[:, i])
            expected += nvec_i @ Deltai @ nvec_i
        np.testing.assert_allclose(result, expected)

    def test_pack_unpack_roundtrip(self, particle):
        """
        _pack and _unpack are inverses.
        MATLAB: bemretiter/private/pack.m, unpack.m.
        """
        bem = _make_ret_iter(particle, solver='gmres', precond=None)

        n = particle.n
        rng = np.random.RandomState(5)
        phi = rng.randn(n) + 1j * rng.randn(n)
        a = rng.randn(n, 3) + 1j * rng.randn(n, 3)
        phip = rng.randn(n) + 1j * rng.randn(n)
        ap = rng.randn(n, 3) + 1j * rng.randn(n, 3)

        vec = bem._pack(phi, a, phip, ap)
        assert vec.shape == (8 * n,)

        phi2, a2, phip2, ap2 = bem._unpack(vec)
        np.testing.assert_allclose(phi2, phi)
        np.testing.assert_allclose(a2, a)
        np.testing.assert_allclose(phip2, phip)
        np.testing.assert_allclose(ap2, ap)

    def test_pack_format(self, particle):
        """
        _pack concatenates [phi(:); a(:); phip(:); ap(:)].
        MATLAB: pack.m: vec = [phi(:); a(:); phip(:); ap(:)].
        """
        bem = _make_ret_iter(particle, solver='gmres', precond=None)
        n = particle.n
        phi = np.arange(n, dtype=complex)
        a = np.arange(3 * n, dtype=complex).reshape(n, 3)
        phip = np.arange(n, dtype=complex) + 100
        ap = np.arange(3 * n, dtype=complex).reshape(n, 3) + 200

        vec = bem._pack(phi, a, phip, ap)
        # MATLAB uses column-major (:) flatten, so expected uses order='F'
        expected = np.hstack([
            phi.ravel(order = 'F'), a.ravel(order = 'F'),
            phip.ravel(order = 'F'), ap.ravel(order = 'F')])
        np.testing.assert_array_equal(vec, expected)

    def test_inner_1d(self):
        """
        _inner computes dot product of nvec and a.
        MATLAB: bemretiter/private/inner.m.
        """
        from mnpbem.bem.bem_ret_iter import BEMRetIter
        n = 4
        rng = np.random.RandomState(10)
        nvec = rng.randn(n, 3)
        a = rng.randn(n, 3)
        result = BEMRetIter._inner(nvec, a)
        expected = np.sum(a * nvec, axis=1)
        np.testing.assert_allclose(result, expected)

    def test_inner_with_mul(self):
        """_inner with mul argument (MATLAB: inner.m, mul line)."""
        from mnpbem.bem.bem_ret_iter import BEMRetIter
        n = 4
        rng = np.random.RandomState(10)
        nvec = rng.randn(n, 3)
        a = rng.randn(n, 3)
        mul = rng.randn(n)
        result = BEMRetIter._inner(nvec, a, mul)
        expected = np.sum(a * nvec, axis=1) * mul
        np.testing.assert_allclose(result, expected)

    def test_inner_zero_input(self):
        """_inner returns 0 for zero input (MATLAB: inner.m line 4)."""
        from mnpbem.bem.bem_ret_iter import BEMRetIter
        nvec = np.ones((4, 3))
        result = BEMRetIter._inner(nvec, 0)
        assert result == 0

    def test_inner_3d(self):
        """_inner for 3D arrays (n, 3, siz) (MATLAB: inner.m squeeze)."""
        from mnpbem.bem.bem_ret_iter import BEMRetIter
        n, siz = 4, 2
        rng = np.random.RandomState(20)
        nvec = rng.randn(n, 3)
        a = rng.randn(n, 3, siz)
        result = BEMRetIter._inner(nvec, a)
        expected = np.sum(a * nvec[:, :, np.newaxis], axis=1)
        np.testing.assert_allclose(result, expected)

    def test_outer_1d(self):
        """
        _outer computes outer product of nvec and val.
        MATLAB: bemretiter/private/outer.m.
        """
        from mnpbem.bem.bem_ret_iter import BEMRetIter
        n = 4
        rng = np.random.RandomState(30)
        nvec = rng.randn(n, 3)
        val = rng.randn(n)
        result = BEMRetIter._outer(nvec, val)
        expected = nvec * val[:, np.newaxis]
        np.testing.assert_allclose(result, expected)

    def test_outer_with_mul(self):
        """_outer with mul argument (MATLAB: outer.m, mul line)."""
        from mnpbem.bem.bem_ret_iter import BEMRetIter
        n = 4
        rng = np.random.RandomState(30)
        nvec = rng.randn(n, 3)
        val = rng.randn(n)
        mul = rng.randn(n)
        result = BEMRetIter._outer(nvec, val, mul)
        expected = nvec * (val * mul)[:, np.newaxis]
        np.testing.assert_allclose(result, expected)

    def test_outer_zero_input(self):
        """_outer returns 0 for zero input (MATLAB: outer.m line 4)."""
        from mnpbem.bem.bem_ret_iter import BEMRetIter
        nvec = np.ones((4, 3))
        result = BEMRetIter._outer(nvec, 0)
        assert result == 0

    def test_outer_2d(self):
        """_outer for 2D val (n, siz) produces (n, 3, siz) result."""
        from mnpbem.bem.bem_ret_iter import BEMRetIter
        n, siz = 4, 3
        rng = np.random.RandomState(31)
        nvec = rng.randn(n, 3)
        val = rng.randn(n, siz)
        result = BEMRetIter._outer(nvec, val)
        assert result.shape == (n, 3, siz)
        for i in range(3):
            np.testing.assert_allclose(result[:, i, :], val * nvec[:, i:i+1])

    def test_subtract(self):
        """_subtract handles mixed array/scalar subtraction."""
        from mnpbem.bem.bem_ret_iter import BEMRetIter
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([0.5, 1.0, 1.5])
        np.testing.assert_allclose(BEMRetIter._subtract(a, b), a - b)
        np.testing.assert_allclose(BEMRetIter._subtract(a, 0), a)
        np.testing.assert_allclose(BEMRetIter._subtract(0, b), -b)
        assert BEMRetIter._subtract(0, 0) == 0

    def test_excitation(self, particle):
        """
        _excitation extracts excitation variables from CompStruct.
        MATLAB: bemretiter/private/excitation.m.
        Returns (phi, a, De, alpha) from Eqs. (10,11,15,18).
        """
        bem = _make_ret_iter(particle, solver='gmres', precond=None)
        bem._init_matrices(500.0)

        n = particle.n
        rng = np.random.RandomState(33)
        exc = CompStruct(particle, 500.0,
                         phi1=rng.randn(n) + 0j,
                         phi2=rng.randn(n) + 0j,
                         phi1p=rng.randn(n) + 0j,
                         phi2p=rng.randn(n) + 0j,
                         a1=rng.randn(n, 3) + 0j,
                         a2=rng.randn(n, 3) + 0j,
                         a1p=rng.randn(n, 3) + 0j,
                         a2p=rng.randn(n, 3) + 0j)
        phi, a, De, alpha = bem._excitation(exc)
        assert isinstance(phi, np.ndarray)
        assert isinstance(a, np.ndarray)
        assert phi.shape == (n,)
        assert a.shape == (n, 3)

    def test_excitation_eq10(self, particle):
        """
        _excitation Eq. (10): phi = phi2 - phi1.
        MATLAB: excitation.m line 22.
        """
        bem = _make_ret_iter(particle, solver='gmres', precond=None)
        bem._init_matrices(500.0)
        n = particle.n
        phi1 = np.ones(n, dtype=complex) * 2.0
        phi2 = np.ones(n, dtype=complex) * 5.0
        exc = CompStruct(particle, 500.0,
                         phi1=phi1, phi2=phi2,
                         a1=np.zeros((n, 3), dtype=complex),
                         a2=np.zeros((n, 3), dtype=complex),
                         phi1p=np.zeros(n, dtype=complex),
                         phi2p=np.zeros(n, dtype=complex),
                         a1p=np.zeros((n, 3), dtype=complex),
                         a2p=np.zeros((n, 3), dtype=complex))
        phi, a, De, alpha = bem._excitation(exc)
        np.testing.assert_allclose(phi, phi2 - phi1)

    def test_excitation_zero_defaults(self, particle):
        """
        _excitation handles missing fields (default to zero).
        MATLAB: excitation.m lines 5-8.
        """
        bem = _make_ret_iter(particle, solver='gmres', precond=None)
        bem._init_matrices(500.0)
        n = particle.n
        exc = CompStruct(particle, 500.0,
                         phi1=np.ones(n, dtype=complex))
        phi, a, De, alpha = bem._excitation(exc)
        np.testing.assert_allclose(phi, -np.ones(n, dtype=complex))

    def test_afun_shape(self, particle):
        """
        _afun maps vec -> result with consistent shapes.
        MATLAB: bemretiter/private/afun.m.

        Input vec is split in half: first half = (sig1, h1) packed, second = (sig2, h2).
        Each half has 4*n elements for single-excitation (siz=1).
        Total input size = 8*n.
        Output is _pack(phi, a, De, alpha) = 8*n elements.
        """
        bem = _make_ret_iter(particle, solver='gmres', precond=None)
        bem._init_matrices(500.0)
        n = particle.n

        # Input: packed (sig1, h1, sig2, h2) with siz=1 -> 8*n total
        total = 8 * n
        rng = np.random.RandomState(44)
        vec = rng.randn(total) + 1j * rng.randn(total)
        result = bem._afun(vec)
        assert result.shape[0] == 8 * n
        assert np.all(np.isfinite(result))

    def test_mfun_shape(self, particle):
        """
        _mfun applies preconditioner to packed vector.
        MATLAB: bemretiter/private/mfun.m.
        """
        bem = _make_ret_iter(particle, solver='gmres', precond='full')
        bem._init_matrices(500.0)
        n = particle.n
        rng = np.random.RandomState(55)
        vec = rng.randn(8 * n) + 1j * rng.randn(8 * n)
        result = bem._mfun(vec)
        assert result.shape == (8 * n,)

    def test_mfun_contents(self, particle):
        """
        _mfun uses correct preconditioner variables.
        MATLAB: mfun.m uses sav.G1i, sav.G2i, sav.Sigma1, etc.
        """
        bem = _make_ret_iter(particle, solver='gmres', precond='full')
        bem._init_matrices(500.0)
        n = particle.n
        rng = np.random.RandomState(56)
        vec = rng.randn(8 * n) + 1j * rng.randn(8 * n)
        result = bem._mfun(vec)
        # Just verify it runs and produces finite values
        assert np.all(np.isfinite(result))

    def test_truediv_calls_solve(self, particle):
        """
        __truediv__ delegates to solve.
        MATLAB: bemretiter/mldivide.m.
        """
        bem = _make_ret_iter(particle, solver='gmres', precond=None)
        bem.solve = MagicMock(return_value=(
            CompStruct(particle, 500.0, sig1=np.zeros(8)),
            bem
        ))
        exc = CompStruct(particle, 500.0)
        sig, _ = bem / exc
        bem.solve.assert_called_once_with(exc)

    def test_potential(self, particle):
        """
        potential() delegates to Green function.
        MATLAB: bemretiter/potential.m.
        """
        bem = _make_ret_iter(particle, solver='gmres', precond=None)
        n = particle.n
        sig = CompStruct(particle, 500.0,
                         sig1=np.ones(n, dtype=complex),
                         sig2=np.ones(n, dtype=complex),
                         h1=np.ones((n, 3), dtype=complex),
                         h2=np.ones((n, 3), dtype=complex))
        pot = bem.potential(sig, 2)
        assert hasattr(pot, 'phi2')

    def test_clear(self, particle):
        """
        clear() resets Green function matrices and preconditioner.
        MATLAB: bemretiter/clear.m.
        """
        bem = _make_ret_iter(particle, solver='gmres', precond='full')
        bem._init_matrices(500.0)
        assert bem._G1 is not None
        bem.clear()
        assert bem._G1 is None
        assert bem._H1 is None
        assert bem._G2 is None
        assert bem._H2 is None
        assert bem._sav is None

    def test_call_init_matrices(self, particle):
        """__call__ triggers _init_matrices (MATLAB: subsref)."""
        bem = _make_ret_iter(particle, solver='gmres', precond=None)
        result = bem(600.0)
        assert result.enei == 600.0

    def test_repr(self, particle):
        """__repr__ shows useful info."""
        bem = _make_ret_iter(particle, solver='gmres', precond=None)
        r = repr(bem)
        assert 'BEMRetIter' in r

    def test_repr_initialized(self, particle):
        """__repr__ shows enei after initialization."""
        bem = _make_ret_iter(particle, solver='gmres', precond=None)
        bem._init_matrices(500.0)
        r = repr(bem)
        assert '500.0' in r

    def test_needs_property(self):
        """needs specifies retarded simulation type."""
        from mnpbem.bem.bem_ret_iter import BEMRetIter
        assert BEMRetIter.needs == {'sim': 'ret'}

    def test_name_property(self):
        """name is 'bemsolver'."""
        from mnpbem.bem.bem_ret_iter import BEMRetIter
        assert BEMRetIter.name == 'bemsolver'

    def test_mtimes(self, particle):
        """
        __mul__ computes potentials at both inside and outside.
        MATLAB: bemretiter/mtimes.m.

        Discrepancy: MATLAB adds potential(sig,1)+potential(sig,2).
        Python stores phi1, a1, phi2, a2 separately.
        """
        bem = _make_ret_iter(particle, solver='gmres', precond=None)
        n = particle.n
        sig = CompStruct(particle, 500.0,
                         sig1=np.ones(n, dtype=complex),
                         sig2=np.ones(n, dtype=complex),
                         h1=np.ones((n, 3), dtype=complex),
                         h2=np.ones((n, 3), dtype=complex))
        phi = bem * sig
        assert hasattr(phi, 'phi1')
        assert hasattr(phi, 'phi2')
        assert hasattr(phi, 'a1')
        assert hasattr(phi, 'a2')

    def test_solve_full(self, particle):
        """
        solve() performs full iterative solution.
        MATLAB: bemretiter/solve.m.
        """
        bem = _make_ret_iter(particle, solver='gmres', tol=1e-4, maxit=50,
                             precond='full')

        n = particle.n
        exc = CompStruct(particle, 500.0,
                         phi1=np.ones(n, dtype=complex),
                         phi2=np.zeros(n, dtype=complex),
                         a1=np.zeros((n, 3), dtype=complex),
                         a2=np.zeros((n, 3), dtype=complex),
                         phi1p=np.zeros(n, dtype=complex),
                         phi2p=np.zeros(n, dtype=complex),
                         a1p=np.zeros((n, 3), dtype=complex),
                         a2p=np.zeros((n, 3), dtype=complex))
        sig, _ = bem.solve(exc)
        assert hasattr(sig, 'sig1')
        assert hasattr(sig, 'sig2')
        assert hasattr(sig, 'h1')
        assert hasattr(sig, 'h2')


# ===========================================================================
# BEMRetLayerIter tests
# ===========================================================================

class TestBEMRetLayerIter(object):
    """
    Tests for BEMRetLayerIter (MATLAB: @bemretlayeriter).

    Layer structure: ss, hh, p, sh, hs components for outer-surface Green
    functions. See Waxenegger et al., CPC 193, 138 (2015).
    """

    def test_construction(self, particle):
        """
        BEMRetLayerIter construction.
        MATLAB: bemretlayeriter(p, op).
        """
        mock_layer = MagicMock()
        bem = _make_ret_layer_iter(particle, solver='gmres', tol=1e-6,
                                   precond=None, layer=mock_layer)
        assert bem.p is particle
        assert bem.layer is mock_layer
        assert bem.solver == 'gmres'
        assert bem.enei is None

    def test_name_and_needs(self):
        """Class attributes match MATLAB Constant properties."""
        from mnpbem.bem.bem_ret_layer_iter import BEMRetLayerIter
        assert BEMRetLayerIter.name == 'bemsolver'
        assert BEMRetLayerIter.needs == {'sim': 'ret'}

    def test_decorate_gamma(self):
        """
        _decorate_gamma uses only parallel (x,y) normal vector components.
        MATLAB: initprecond.m, fun(Gamma, nvec) with nvec(:,1:2).
        """
        from mnpbem.bem.bem_ret_layer_iter import BEMRetLayerIter
        n = 4
        rng = np.random.RandomState(88)
        Gamma = rng.randn(n, n) + 1j * rng.randn(n, n)
        nvec = rng.randn(n, 3)
        nvec = nvec / np.linalg.norm(nvec, axis=1, keepdims=True)

        result = BEMRetLayerIter._decorate_gamma(Gamma, nvec)

        expected = np.zeros((n, n), dtype=Gamma.dtype)
        for i in range(2):  # only x, y
            nvec_i = np.diag(nvec[:, i])
            expected += nvec_i @ Gamma @ nvec_i
        np.testing.assert_allclose(result, expected)

    def test_decorate_gamma_vs_deltai(self):
        """
        _decorate_gamma uses 2 components (parallel), while BEMRetIter._decorate_deltai
        uses 3 (full). This is the key difference for layer structure.
        """
        from mnpbem.bem.bem_ret_iter import BEMRetIter
        from mnpbem.bem.bem_ret_layer_iter import BEMRetLayerIter
        n = 4
        rng = np.random.RandomState(90)
        M = rng.randn(n, n) + 1j * rng.randn(n, n)
        nvec = rng.randn(n, 3)
        nvec = nvec / np.linalg.norm(nvec, axis=1, keepdims=True)

        gamma_result = BEMRetLayerIter._decorate_gamma(M, nvec)
        deltai_result = BEMRetIter._decorate_deltai(M, nvec)

        # They should differ because gamma only uses x,y while deltai uses x,y,z
        assert not np.allclose(gamma_result, deltai_result)

    def test_pack_unpack_4arg_roundtrip(self, particle):
        """
        _pack/_unpack with 4 arguments: roundtrip.
        MATLAB: bemretlayeriter/private/pack.m (4 args), unpack.m (nargout==4).
        """
        bem = _make_ret_layer_iter(particle, solver='gmres', precond=None)
        n = particle.n
        rng = np.random.RandomState(15)
        phi = rng.randn(n) + 1j * rng.randn(n)
        a = rng.randn(n, 3) + 1j * rng.randn(n, 3)
        phip = rng.randn(n) + 1j * rng.randn(n)
        ap = rng.randn(n, 3) + 1j * rng.randn(n, 3)

        vec = bem._pack(phi, a, phip, ap)
        assert vec.shape == (8 * n,)

        phi2, a2, phip2, ap2 = bem._unpack(vec, nout=4)
        np.testing.assert_allclose(phi2, phi)
        np.testing.assert_allclose(a2, a)
        np.testing.assert_allclose(phip2, phip)
        np.testing.assert_allclose(ap2, ap)

    def test_unpack_6_outputs(self, particle):
        """
        _unpack with nout=6 decomposes a into (apar, aperp).
        MATLAB: unpack.m with nargout==6.
        """
        bem = _make_ret_layer_iter(particle, solver='gmres', precond=None)
        n = particle.n
        rng = np.random.RandomState(16)
        phi = rng.randn(n) + 1j * rng.randn(n)
        a = rng.randn(n, 3) + 1j * rng.randn(n, 3)
        phip = rng.randn(n) + 1j * rng.randn(n)
        ap = rng.randn(n, 3) + 1j * rng.randn(n, 3)

        vec = bem._pack(phi, a, phip, ap)
        phi_r, apar, aperp, phip_r, appar, apperp = bem._unpack(vec, nout=6)

        np.testing.assert_allclose(phi_r, phi)
        np.testing.assert_allclose(phip_r, phip)
        np.testing.assert_allclose(apar, a[:, :2])
        np.testing.assert_allclose(aperp, a[:, 2])
        np.testing.assert_allclose(appar, ap[:, :2])
        np.testing.assert_allclose(apperp, ap[:, 2])

    def test_inner_2d_nvec(self):
        """
        _inner handles 2D normal vectors (parallel only).
        MATLAB: bemretlayeriter/private/inner.m with nvec 2 columns.
        """
        from mnpbem.bem.bem_ret_layer_iter import BEMRetLayerIter
        n = 4
        rng = np.random.RandomState(25)
        nvec = rng.randn(n, 2)
        a = rng.randn(n, 2)
        result = BEMRetLayerIter._inner(nvec, a)
        expected = np.sum(a * nvec, axis=1)
        np.testing.assert_allclose(result, expected)

    def test_inner_3d_nvec(self):
        """_inner handles 3D normal vectors (full)."""
        from mnpbem.bem.bem_ret_layer_iter import BEMRetLayerIter
        n = 4
        rng = np.random.RandomState(25)
        nvec = rng.randn(n, 3)
        a = rng.randn(n, 3)
        result = BEMRetLayerIter._inner(nvec, a)
        expected = np.sum(a * nvec, axis=1)
        np.testing.assert_allclose(result, expected)

    def test_inner_zero(self):
        """_inner returns 0 for zero input."""
        from mnpbem.bem.bem_ret_layer_iter import BEMRetLayerIter
        nvec = np.ones((4, 3))
        assert BEMRetLayerIter._inner(nvec, 0) == 0

    def test_inner_with_mul(self):
        """_inner with mul argument."""
        from mnpbem.bem.bem_ret_layer_iter import BEMRetLayerIter
        n = 4
        rng = np.random.RandomState(26)
        nvec = rng.randn(n, 3)
        a = rng.randn(n, 3)
        mul = rng.randn(n)
        result = BEMRetLayerIter._inner(nvec, a, mul)
        expected = np.sum(a * nvec, axis=1) * mul
        np.testing.assert_allclose(result, expected)

    def test_outer_2d_nvec(self):
        """
        _outer with 2D nvec returns (n, 2) result.
        MATLAB: bemretlayeriter/private/outer.m with nvec 2 columns.
        """
        from mnpbem.bem.bem_ret_layer_iter import BEMRetLayerIter
        n = 4
        rng = np.random.RandomState(35)
        nvec = rng.randn(n, 2)
        val = rng.randn(n)
        result = BEMRetLayerIter._outer(nvec, val)
        assert result.shape == (n, 2)
        for i in range(2):
            np.testing.assert_allclose(result[:, i], val * nvec[:, i])

    def test_outer_3d_nvec(self):
        """_outer with 3D nvec returns (n, 3) result."""
        from mnpbem.bem.bem_ret_layer_iter import BEMRetLayerIter
        n = 4
        rng = np.random.RandomState(35)
        nvec = rng.randn(n, 3)
        val = rng.randn(n)
        result = BEMRetLayerIter._outer(nvec, val)
        assert result.shape == (n, 3)

    def test_outer_zero(self):
        """_outer returns 0 for zero input."""
        from mnpbem.bem.bem_ret_layer_iter import BEMRetLayerIter
        nvec = np.ones((4, 3))
        assert BEMRetLayerIter._outer(nvec, 0) == 0

    def test_outer_2d_val(self):
        """_outer for 2D val (n, siz) produces (n, ndim, siz) result."""
        from mnpbem.bem.bem_ret_layer_iter import BEMRetLayerIter
        n, siz = 4, 3
        rng = np.random.RandomState(36)
        nvec = rng.randn(n, 3)
        val = rng.randn(n, siz)
        result = BEMRetLayerIter._outer(nvec, val)
        assert result.shape == (n, 3, siz)

    def test_subtract(self):
        """_subtract handles mixed scalar/array."""
        from mnpbem.bem.bem_ret_layer_iter import BEMRetLayerIter
        a = np.array([1.0, 2.0])
        b = np.array([0.5, 1.0])
        np.testing.assert_allclose(BEMRetLayerIter._subtract(a, b), a - b)
        np.testing.assert_allclose(BEMRetLayerIter._subtract(a, 0), a)
        np.testing.assert_allclose(BEMRetLayerIter._subtract(0, b), -b)

    def test_solve_block_lu(self):
        """
        _solve_block_lu performs block LU solve.
        MATLAB: bemretlayeriter/private/mfun.m, function fun(M, b1, b2).
        """
        from mnpbem.bem.bem_ret_layer_iter import BEMRetLayerIter
        n = 4
        rng = np.random.RandomState(66)
        im11 = rng.randn(n, n) + 1j * rng.randn(n, n)
        im12 = rng.randn(n, n) + 1j * rng.randn(n, n)
        im21 = rng.randn(n, n) + 1j * rng.randn(n, n)
        im22 = rng.randn(n, n) + 1j * rng.randn(n, n)
        im = [[im11, im12], [im21, im22]]

        b1 = rng.randn(n) + 1j * rng.randn(n)
        b2 = rng.randn(n) + 1j * rng.randn(n)

        x1, x2 = BEMRetLayerIter._solve_block_lu(im, b1, b2)
        assert x1.shape == (n,)
        assert x2.shape == (n,)
        # Verify finite values
        assert np.all(np.isfinite(x1))
        assert np.all(np.isfinite(x2))

    def test_clear(self, particle):
        """
        clear() resets matrices.
        MATLAB: bemretlayeriter/clear.m.
        """
        bem = _make_ret_layer_iter(particle, solver='gmres', precond=None)
        bem._G1 = np.ones((4, 4))
        bem._H1 = np.ones((4, 4))
        bem._G2 = np.ones((4, 4))
        bem._H2 = np.ones((4, 4))
        bem._sav = {'k': 1.0}

        result = bem.clear()
        assert result._G1 is None
        assert result._H1 is None
        assert result._G2 is None
        assert result._H2 is None
        assert result._sav is None

    def test_call_init_matrices(self, particle):
        """__call__ triggers _init_matrices."""
        bem = _make_ret_layer_iter(particle, solver='gmres', precond=None)
        bem._init_matrices = MagicMock(return_value=bem)
        result = bem(500.0)
        bem._init_matrices.assert_called_once_with(500.0)

    def test_repr(self, particle):
        """__repr__ shows useful info."""
        bem = _make_ret_layer_iter(particle, solver='gmres', precond=None)
        r = repr(bem)
        assert 'BEMRetLayerIter' in r
        assert 'gmres' in r

    def test_truediv_calls_solve(self, particle):
        """
        __truediv__ delegates to solve.
        MATLAB: bemretlayeriter/mldivide.m.
        """
        bem = _make_ret_layer_iter(particle, solver='gmres', precond=None)
        bem.solve = MagicMock(return_value=(
            CompStruct(particle, 500.0, sig1=np.zeros(8)),
            bem
        ))
        exc = CompStruct(particle, 500.0)
        sig, _ = bem / exc
        bem.solve.assert_called_once_with(exc)

    def test_potential(self, particle):
        """
        potential() delegates to Green function.
        MATLAB: bemretlayeriter/potential.m.
        """
        bem = _make_ret_layer_iter(particle, solver='gmres', precond=None)
        n = particle.n
        sig = CompStruct(particle, 500.0,
                         sig1=np.ones(n, dtype=complex),
                         sig2=np.ones(n, dtype=complex),
                         h1=np.ones((n, 3), dtype=complex),
                         h2=np.ones((n, 3), dtype=complex))
        pot = bem.potential(sig, 2)
        assert hasattr(pot, 'phi2')

    def test_excitation(self, particle):
        """
        _excitation extracts and computes excitation variables.
        MATLAB: bemretlayeriter/private/excitation.m.
        """
        bem = _make_ret_layer_iter(particle, solver='gmres', precond=None)
        bem._k = 2 * np.pi / 500.0
        bem._eps1 = particle.eps1(500.0)
        bem._eps2 = particle.eps2(500.0)

        n = particle.n
        rng = np.random.RandomState(40)
        exc = CompStruct(particle, 500.0,
                         phi1=rng.randn(n) + 0j,
                         phi2=rng.randn(n) + 0j,
                         phi1p=rng.randn(n) + 0j,
                         phi2p=rng.randn(n) + 0j,
                         a1=rng.randn(n, 3) + 0j,
                         a2=rng.randn(n, 3) + 0j,
                         a1p=rng.randn(n, 3) + 0j,
                         a2p=rng.randn(n, 3) + 0j)
        phi, a, alpha, De = bem._excitation(exc)
        assert isinstance(phi, np.ndarray)
        assert isinstance(a, np.ndarray)
        assert phi.shape == (n,)
        assert a.shape == (n, 3)

    def test_excitation_return_order(self, particle):
        """
        _excitation returns (phi, a, alpha, De) -- note alpha before De.
        MATLAB: function [phi, a, alpha, De] = excitation(obj, exc).
        """
        bem = _make_ret_layer_iter(particle, solver='gmres', precond=None)
        bem._k = 2 * np.pi / 500.0
        bem._eps1 = particle.eps1(500.0)
        bem._eps2 = particle.eps2(500.0)

        n = particle.n
        exc = CompStruct(particle, 500.0,
                         phi1=np.zeros(n, dtype=complex),
                         phi2=np.ones(n, dtype=complex),
                         phi1p=np.zeros(n, dtype=complex),
                         phi2p=np.zeros(n, dtype=complex),
                         a1=np.zeros((n, 3), dtype=complex),
                         a2=np.zeros((n, 3), dtype=complex),
                         a1p=np.zeros((n, 3), dtype=complex),
                         a2p=np.zeros((n, 3), dtype=complex))
        phi, a, alpha, De = bem._excitation(exc)
        # phi = phi2 - phi1 = ones
        np.testing.assert_allclose(phi, np.ones(n, dtype=complex))

    def test_pack_6arg(self, particle):
        """
        _pack with 6 arguments combines par/perp into full 3D vectors.
        MATLAB: pack.m with numel(varargin)==6.
        """
        bem = _make_ret_layer_iter(particle, solver='gmres', precond=None)
        n = particle.n
        rng = np.random.RandomState(17)
        phi = rng.randn(n) + 1j * rng.randn(n)
        apar = rng.randn(n, 2) + 1j * rng.randn(n, 2)
        aperp = rng.randn(n) + 1j * rng.randn(n)
        phip = rng.randn(n) + 1j * rng.randn(n)
        appar = rng.randn(n, 2) + 1j * rng.randn(n, 2)
        apperp = rng.randn(n) + 1j * rng.randn(n)

        vec = bem._pack(phi, apar, aperp, phip, appar, apperp)
        assert vec.shape == (8 * n,)

        # Roundtrip through unpack with nout=6
        phi_r, apar_r, aperp_r, phip_r, appar_r, apperp_r = bem._unpack(vec, nout=6)
        np.testing.assert_allclose(phi_r, phi)
        np.testing.assert_allclose(phip_r, phip)
        np.testing.assert_allclose(apar_r, apar)
        np.testing.assert_allclose(aperp_r, aperp)
        np.testing.assert_allclose(appar_r, appar)
        np.testing.assert_allclose(apperp_r, apperp)

    def test_layer_green_class_exists(self):
        """_LayerGreen helper class exists and has expected attributes."""
        from mnpbem.bem.bem_ret_layer_iter import _LayerGreen
        lg = _LayerGreen()
        assert lg.ss is None
        assert lg.hh is None
        assert lg.p is None
        assert lg.sh is None
        assert lg.hs is None


# ===========================================================================
# Cross-class consistency tests
# ===========================================================================

class TestCrossClassConsistency(object):
    """Tests verifying consistency across the iterative solver hierarchy."""

    def test_bemstatiter_inherits_bemiter(self):
        """BEMStatIter inherits from BEMIter (MATLAB: bemstatiter < bemiter)."""
        from mnpbem.bem.bem_stat_iter import BEMStatIter
        assert issubclass(BEMStatIter, BEMIter)

    def test_bemretiter_inherits_bemiter(self):
        """BEMRetIter inherits from BEMIter (MATLAB: bemretiter < bemiter)."""
        from mnpbem.bem.bem_ret_iter import BEMRetIter
        assert issubclass(BEMRetIter, BEMIter)

    def test_bemretlayeriter_inherits_bemiter(self):
        """BEMRetLayerIter inherits from BEMIter (MATLAB: bemretlayeriter < bemiter)."""
        from mnpbem.bem.bem_ret_layer_iter import BEMRetLayerIter
        assert issubclass(BEMRetLayerIter, BEMIter)

    def test_solver_map_all_entries(self):
        """SOLVER_MAP contains all supported solvers."""
        from scipy.sparse.linalg import gmres, cgs, bicgstab
        assert 'gmres' in BEMIter.SOLVER_MAP
        assert 'cgs' in BEMIter.SOLVER_MAP
        assert 'bicgstab' in BEMIter.SOLVER_MAP
        assert BEMIter.SOLVER_MAP['gmres'] is gmres
        assert BEMIter.SOLVER_MAP['cgs'] is cgs
        assert BEMIter.SOLVER_MAP['bicgstab'] is bicgstab

    def test_all_solvers_have_name_property(self):
        """All iterative solvers have name='bemsolver'."""
        from mnpbem.bem.bem_stat_iter import BEMStatIter
        from mnpbem.bem.bem_ret_iter import BEMRetIter
        from mnpbem.bem.bem_ret_layer_iter import BEMRetLayerIter
        assert BEMStatIter.name == 'bemsolver'
        assert BEMRetIter.name == 'bemsolver'
        assert BEMRetLayerIter.name == 'bemsolver'


# ===========================================================================
# Edge case tests
# ===========================================================================

class TestEdgeCases(object):
    """Edge case and boundary condition tests."""

    def test_bemiter_info_none_returns_none(self):
        """info() returns None lists when no iterations have occurred."""
        it = BEMIter()
        flag, relres, iters = it.info()
        assert flag is None
        assert relres is None
        assert iters is None

    def test_bemiter_multiple_set_iter_calls(self):
        """Multiple _set_iter calls accumulate correctly."""
        it = BEMIter()
        for i in range(10):
            it._set_iter(0, 1e-6 * (i + 1), np.array([i * 10, i]))
        assert len(it._flag) == 10
        assert len(it._relres) == 10
        assert len(it._iter) == 10

    def test_bemiter_set_stat_multiple_names(self):
        """_set_stat accumulates stats for different H-matrix names."""
        it = BEMIter()
        mock_hmat = MagicMock()
        mock_hmat.compression.return_value = 0.1
        it._set_stat('G', mock_hmat)
        it._set_stat('F', mock_hmat)
        it._set_stat('G1', mock_hmat)
        assert len(it._stat['compression']) == 3
        assert 'G' in it._stat['compression']
        assert 'F' in it._stat['compression']
        assert 'G1' in it._stat['compression']

    def test_pack_unpack_single_column(self, particle):
        """Pack/unpack with siz=1 (single excitation column)."""
        bem = _make_ret_iter(particle, solver='gmres', precond=None)
        n = particle.n
        phi = np.ones(n, dtype=complex)
        a = np.ones((n, 3), dtype=complex) * 2.0
        phip = np.ones(n, dtype=complex) * 3.0
        ap = np.ones((n, 3), dtype=complex) * 4.0

        vec = bem._pack(phi, a, phip, ap)
        phi2, a2, phip2, ap2 = bem._unpack(vec)

        np.testing.assert_allclose(phi2, phi)
        np.testing.assert_allclose(a2, a)
        np.testing.assert_allclose(phip2, phip)
        np.testing.assert_allclose(ap2, ap)

    def test_gmres_restart_default(self):
        """When restart is None, gmres uses min(n, 20) as default."""
        it = BEMIter(solver='gmres', tol=1e-10, maxit=50, restart=None,
                     precond=None)
        n = 10
        rng = np.random.RandomState(200)
        A = rng.randn(n, n) + 1j * rng.randn(n, n)
        A = A @ A.conj().T + n * np.eye(n)
        b = rng.randn(n) + 1j * rng.randn(n)
        afun = lambda x: A @ x
        x, _ = it._iter_solve(None, b, afun, None)
        np.testing.assert_allclose(A @ x, b, atol=1e-6)

    def test_gmres_custom_restart(self):
        """When restart is specified, gmres uses it."""
        it = BEMIter(solver='gmres', tol=1e-10, maxit=50, restart=5,
                     precond=None)
        assert it.restart == 5
        n = 10
        rng = np.random.RandomState(201)
        A = rng.randn(n, n) + 1j * rng.randn(n, n)
        A = A @ A.conj().T + n * np.eye(n)
        b = rng.randn(n) + 1j * rng.randn(n)
        afun = lambda x: A @ x
        x, _ = it._iter_solve(None, b, afun, None)
        np.testing.assert_allclose(A @ x, b, atol=1e-6)

    def test_stat_iter_solve_converges(self, particle):
        """BEMStatIter solve produces meaningful output for simple system."""
        bem = _make_stat_iter(particle, solver='gmres', tol=1e-4, maxit=100,
                              precond='hmat')
        n = particle.n
        rng = np.random.RandomState(77)
        exc = CompStruct(particle, 500.0,
                         phip=rng.randn(n) + 0j)
        sig, _ = bem.solve(exc)
        # Result should be finite
        assert np.all(np.isfinite(sig.sig))

    def test_hinfo_with_only_green_stats(self, capsys):
        """hinfo() prints only Green function compression when no aux stats."""
        it = BEMIter()
        mock_hmat = MagicMock()
        mock_hmat.compression.return_value = 0.3
        it._set_stat('G', mock_hmat)
        it._set_stat('H1', mock_hmat)

        it.hinfo()
        captured = capsys.readouterr()
        assert 'Compression Green functions' in captured.out
        # No auxiliary matrices, so that line should not appear
        assert 'auxiliary' not in captured.out

    def test_hinfo_with_only_aux_stats(self, capsys):
        """hinfo() prints only auxiliary compression when no Green stats."""
        it = BEMIter()
        mock_hmat = MagicMock()
        mock_hmat.compression.return_value = 0.2
        it._set_stat('mat', mock_hmat)
        it._set_stat('Sigma1', mock_hmat)

        it.hinfo()
        captured = capsys.readouterr()
        # No Green function stats
        assert 'Green functions' not in captured.out
        assert 'Compression auxiliary matrices' in captured.out
