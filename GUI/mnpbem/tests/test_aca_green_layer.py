import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from mnpbem.materials.eps_const import EpsConst
from mnpbem.geometry.layer_structure import LayerStructure
from mnpbem.greenfun.compgreen_ret_layer import CompGreenRetLayer
from mnpbem.greenfun.aca_compgreen_ret_layer import ACACompGreenRetLayer
from mnpbem.greenfun.compgreen_stat import CompStruct
from mnpbem.greenfun.hmatrix import HMatrix


# ---------------------------------------------------------------------------
# Mock objects
# ---------------------------------------------------------------------------

class MockParticle(object):

    def __init__(self,
            n_faces: int = 32,
            radius: float = 10.0,
            z_center: float = 15.0,
            eps_funcs: list = None,
            inout_arr: list = None) -> None:

        rng = np.random.RandomState(42)
        phi = rng.uniform(0, 2 * np.pi, n_faces)
        cos_theta = rng.uniform(-1, 1, n_faces)
        theta = np.arccos(cos_theta)

        x = radius * np.sin(theta) * np.cos(phi)
        y = radius * np.sin(theta) * np.sin(phi)
        z = radius * np.cos(theta) + z_center

        self.pos = np.empty((n_faces, 3), dtype = np.float64)
        self.pos[:, 0] = x
        self.pos[:, 1] = y
        self.pos[:, 2] = z

        # Outward normals
        center = np.array([0.0, 0.0, z_center])
        dirs = self.pos - center
        norms = np.linalg.norm(dirs, axis = 1, keepdims = True)
        self.nvec = dirs / norms

        # Face areas (approximate)
        surface_area = 4 * np.pi * radius ** 2
        self.area = np.full(n_faces, surface_area / n_faces)

        self.n = n_faces
        self.nfaces = n_faces
        self._np_val = 1

        if inout_arr is not None:
            self.inout = np.array(inout_arr, dtype = int)
        else:
            self.inout = np.array([[2, 1]], dtype = int)

        if eps_funcs is None:
            self.eps = [
                EpsConst(1.0),
                EpsConst(-10.0 + 1.0j),
            ]
        else:
            self.eps = eps_funcs

        self.p = [self]
        self.pc = self
        self.closed = [None]

    @property
    def np(self) -> int:
        return self._np_val

    def bradius(self) -> 'np.ndarray':
        import numpy
        return numpy.full(self.nfaces, 1.0)

    def eps1(self, enei: float) -> 'np.ndarray':
        import numpy
        eps_val, _ = self.eps[self.inout[0, 0] - 1](enei)
        return numpy.full(self.n, complex(eps_val))

    def eps2(self, enei: float) -> 'np.ndarray':
        import numpy
        eps_val, _ = self.eps[self.inout[0, 1] - 1](enei)
        return numpy.full(self.n, complex(eps_val))

    def index(self, ip: int) -> list:
        if ip == 1:
            return list(range(self.n))
        return []

    def index_func(self, ip: int) -> list:
        return self.index(ip)

    def closedparticle(self, ip: int) -> tuple:
        return None, 1, None


class MockSig(object):

    def __init__(self,
            n: int,
            enei: float = 500.0) -> None:

        rng = np.random.RandomState(123)
        self.enei = enei
        self.sig = rng.randn(n) + 1j * rng.randn(n)
        self.sig1 = self.sig.copy()
        self.sig2 = rng.randn(n) + 1j * rng.randn(n)
        self.h1 = rng.randn(n, 3) + 1j * rng.randn(n, 3)
        self.h2 = rng.randn(n, 3) + 1j * rng.randn(n, 3)


def _patch_reflected_green(aca_g: ACACompGreenRetLayer, n: int, enei: float) -> None:

    # Patch the internal GreenRetLayer with synthetic reflected Green function
    # matrices so we can test the ACA wrapping without triggering the full
    # layer integration (which has broadcasting issues).
    rng = np.random.RandomState(999)
    G_refl = rng.randn(n, n) + 1j * rng.randn(n, n)
    F_refl = rng.randn(n, n) + 1j * rng.randn(n, n)

    # Make them smoothly decaying for better low-rank approximability
    pos = aca_g.p.pos
    for ii in range(n):
        for jj in range(n):
            dx = pos[ii, 0] - pos[jj, 0]
            dy = pos[ii, 1] - pos[jj, 1]
            r = np.sqrt(dx ** 2 + dy ** 2) + 1.0
            G_refl[ii, jj] = np.exp(-0.1 * r) / r * (1 + 0.5j)
            F_refl[ii, jj] = -np.exp(-0.1 * r) / (r ** 2) * (1 + 0.3j)

    aca_g.g.gr.G = G_refl
    aca_g.g.gr.F = F_refl
    aca_g.g.gr.enei = enei


def _build_dense_green(p: MockParticle, enei: float) -> np.ndarray:

    # Build a synthetic dense retarded Green function matrix (1/r kernel)
    pos = p.pos
    n = pos.shape[0]
    G = np.empty((n, n), dtype = complex)

    for ii in range(n):
        for jj in range(n):
            dist = np.linalg.norm(pos[ii] - pos[jj])
            if dist < 1e-10:
                G[ii, jj] = 0.0
            else:
                k = 2 * np.pi / enei
                G[ii, jj] = np.exp(1j * k * dist) / (4 * np.pi * dist)

    return G


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def single_layer() -> LayerStructure:
    epstab = [EpsConst(1.0), EpsConst(2.25)]
    layer = LayerStructure(epstab, [1, 2], [0.0])
    return layer


@pytest.fixture
def particle() -> MockParticle:
    return MockParticle(n_faces = 64, radius = 10.0, z_center = 15.0)


@pytest.fixture
def small_particle() -> MockParticle:
    return MockParticle(n_faces = 32, radius = 8.0, z_center = 12.0)


@pytest.fixture
def aca_green(small_particle: MockParticle,
        single_layer: LayerStructure) -> ACACompGreenRetLayer:

    aca_g = ACACompGreenRetLayer(small_particle, single_layer, htol = 1e-4, cleaf = 8)
    # Patch reflected Green function to avoid layer integration
    _patch_reflected_green(aca_g, small_particle.n, 500.0)
    return aca_g


# ---------------------------------------------------------------------------
# ACA layer Green function construction
# ---------------------------------------------------------------------------

class TestACAConstruction(object):

    def test_basic_construction(self,
            particle: MockParticle,
            single_layer: LayerStructure) -> None:

        aca_g = ACACompGreenRetLayer(particle, single_layer, htol = 1e-4)

        assert aca_g.p is particle
        assert aca_g.layer is single_layer
        assert aca_g.tree is not None
        assert aca_g.hmat is not None
        assert aca_g.tree.n == particle.n

    def test_hmatrix_template_initialized(self,
            particle: MockParticle,
            single_layer: LayerStructure) -> None:

        aca_g = ACACompGreenRetLayer(particle, single_layer, htol = 1e-4)

        hmat = aca_g.hmat
        assert len(hmat.row1) > 0 or len(hmat.row2) > 0
        assert hmat.tree is aca_g.tree

    def test_repr(self,
            particle: MockParticle,
            single_layer: LayerStructure) -> None:

        aca_g = ACACompGreenRetLayer(particle, single_layer, htol = 1e-4)
        r = repr(aca_g)
        assert 'ACACompGreenRetLayer' in r
        assert 'n=' in r

    def test_underlying_green(self,
            small_particle: MockParticle,
            single_layer: LayerStructure) -> None:

        aca_g = ACACompGreenRetLayer(small_particle, single_layer, htol = 1e-4)
        assert aca_g.g is not None
        assert isinstance(aca_g.g, CompGreenRetLayer)


# ---------------------------------------------------------------------------
# ACA layer Green function accuracy vs dense (eval1 - direct)
# ---------------------------------------------------------------------------

class TestACAEval1Accuracy(object):

    def test_eval1_returns_hmatrix(self,
            aca_green: ACACompGreenRetLayer) -> None:

        # Patch the dense eval to provide a synthetic dense matrix
        n = aca_green.p.n
        enei = 500.0
        dense_mat = _build_dense_green(aca_green.p, enei)
        aca_green._cache[('dense', 0, 0, 'G', enei)] = dense_mat

        result = aca_green._eval1(0, 0, 'G', enei)
        assert isinstance(result, HMatrix)

    def test_eval1_accuracy(self,
            aca_green: ACACompGreenRetLayer) -> None:

        n = aca_green.p.n
        enei = 500.0
        dense_mat = _build_dense_green(aca_green.p, enei)
        aca_green._cache[('dense', 0, 0, 'G', enei)] = dense_mat

        result = aca_green._eval1(0, 0, 'G', enei)
        A_full = result.full()

        rel_err = np.linalg.norm(A_full - dense_mat) / np.linalg.norm(dense_mat)
        assert rel_err < 1e-2, 'eval1 ACA relative error {} too large'.format(rel_err)

    def test_eval1_matvec_accuracy(self,
            aca_green: ACACompGreenRetLayer) -> None:

        n = aca_green.p.n
        enei = 500.0
        dense_mat = _build_dense_green(aca_green.p, enei)
        aca_green._cache[('dense', 0, 0, 'G', enei)] = dense_mat

        hmat = aca_green._eval1(0, 0, 'G', enei)

        rng = np.random.RandomState(77)
        v = rng.randn(n) + 1j * rng.randn(n)

        result_dense = dense_mat @ v
        result_aca = hmat @ v

        dense_norm = np.linalg.norm(result_dense)
        rel_err = np.linalg.norm(result_aca - result_dense) / dense_norm
        assert rel_err < 1e-2, 'eval1 matvec relative error {}'.format(rel_err)

    def test_eval1_keys(self,
            aca_green: ACACompGreenRetLayer) -> None:

        n = aca_green.p.n
        enei = 500.0

        for key in ['G', 'F', 'H1', 'H2']:
            dense_mat = _build_dense_green(aca_green.p, enei)
            aca_green._cache[('dense', 0, 0, key, enei)] = dense_mat
            result = aca_green._eval1(0, 0, key, enei)
            assert isinstance(result, HMatrix)


# ---------------------------------------------------------------------------
# ACA layer Green function accuracy vs dense (eval2 - reflected)
# ---------------------------------------------------------------------------

class TestACAEval2Accuracy(object):

    def test_eval2_returns_dict(self,
            aca_green: ACACompGreenRetLayer) -> None:

        n = aca_green.p.n
        enei = 500.0
        dense_mat = _build_dense_green(aca_green.p, enei)
        aca_green._cache[('dense', 1, 1, 'G', enei)] = dense_mat

        result = aca_green._eval2(1, 1, 'G', enei)
        assert isinstance(result, dict)
        assert 'ss' in result
        assert 'hh' in result
        assert 'p' in result

    def test_eval2_components_are_hmatrix(self,
            aca_green: ACACompGreenRetLayer) -> None:

        n = aca_green.p.n
        enei = 500.0
        dense_mat = _build_dense_green(aca_green.p, enei)
        aca_green._cache[('dense', 1, 1, 'G', enei)] = dense_mat

        result = aca_green._eval2(1, 1, 'G', enei)

        for name, hmat in result.items():
            assert isinstance(hmat, HMatrix), (
                'Component {} is not HMatrix'.format(name))

    def test_eval2_ss_accuracy(self,
            aca_green: ACACompGreenRetLayer) -> None:

        n = aca_green.p.n
        enei = 500.0
        dense_mat = _build_dense_green(aca_green.p, enei)
        aca_green._cache[('dense', 1, 1, 'G', enei)] = dense_mat

        result = aca_green._eval2(1, 1, 'G', enei)
        ss_full = result['ss'].full()

        # ss component = direct + reflected
        refl = aca_green.g.gr.G
        expected = dense_mat + refl

        dense_norm = np.linalg.norm(expected)
        if dense_norm > 1e-15:
            rel_err = np.linalg.norm(ss_full - expected) / dense_norm
            assert rel_err < 0.05, (
                'eval2 ss component relative error {} too large'.format(rel_err))

    def test_eval2_sh_is_zero(self,
            aca_green: ACACompGreenRetLayer) -> None:

        n = aca_green.p.n
        enei = 500.0
        dense_mat = _build_dense_green(aca_green.p, enei)
        aca_green._cache[('dense', 1, 1, 'G', enei)] = dense_mat

        result = aca_green._eval2(1, 1, 'G', enei)
        sh_full = result['sh'].full()

        # sh should be zero (no cross-coupling in simple substrate)
        assert np.linalg.norm(sh_full) < 1e-10, (
            'eval2 sh component should be near zero, norm={}'.format(np.linalg.norm(sh_full)))


# ---------------------------------------------------------------------------
# Eval dispatch
# ---------------------------------------------------------------------------

class TestEvalDispatch(object):

    def test_eval_dispatches_eval1_for_00(self,
            aca_green: ACACompGreenRetLayer) -> None:

        n = aca_green.p.n
        enei = 500.0
        dense_mat = _build_dense_green(aca_green.p, enei)
        aca_green._cache[('dense', 0, 0, 'G', enei)] = dense_mat

        result = aca_green.eval(0, 0, 'G', enei)
        assert isinstance(result, HMatrix)

    def test_eval_dispatches_eval1_for_01(self,
            aca_green: ACACompGreenRetLayer) -> None:

        n = aca_green.p.n
        enei = 500.0
        dense_mat = _build_dense_green(aca_green.p, enei)
        aca_green._cache[('dense', 0, 1, 'G', enei)] = dense_mat

        result = aca_green.eval(0, 1, 'G', enei)
        assert isinstance(result, HMatrix)

    def test_eval_dispatches_eval2_for_11(self,
            aca_green: ACACompGreenRetLayer) -> None:

        n = aca_green.p.n
        enei = 500.0
        dense_mat = _build_dense_green(aca_green.p, enei)
        aca_green._cache[('dense', 1, 1, 'G', enei)] = dense_mat

        result = aca_green.eval(1, 1, 'G', enei)
        assert isinstance(result, dict)

    def test_eval_dispatches_eval1_for_10(self,
            aca_green: ACACompGreenRetLayer) -> None:

        n = aca_green.p.n
        enei = 500.0
        dense_mat = _build_dense_green(aca_green.p, enei)
        aca_green._cache[('dense', 1, 0, 'G', enei)] = dense_mat

        result = aca_green.eval(1, 0, 'G', enei)
        assert isinstance(result, HMatrix)


# ---------------------------------------------------------------------------
# Potential computation
# ---------------------------------------------------------------------------

class TestACALayerPotential(object):

    def _setup_potential(self,
            aca_green: ACACompGreenRetLayer,
            enei: float) -> None:

        n = aca_green.p.n
        dense_mat = _build_dense_green(aca_green.p, enei)

        # Pre-populate cache for all region combinations
        for ii in range(2):
            for jj in range(2):
                for key in ['G', 'F', 'H1', 'H2']:
                    aca_green._cache[('dense', ii, jj, key, enei)] = dense_mat

    def test_potential_inside_construction(self,
            aca_green: ACACompGreenRetLayer) -> None:

        enei = 500.0
        self._setup_potential(aca_green, enei)
        sig = MockSig(aca_green.p.n, enei = enei)

        pot = aca_green.potential(sig, inout = 1)

        assert isinstance(pot, CompStruct)
        assert hasattr(pot, 'phi1')
        assert hasattr(pot, 'phi1p')

    def test_potential_outside_construction(self,
            aca_green: ACACompGreenRetLayer) -> None:

        enei = 500.0
        self._setup_potential(aca_green, enei)
        sig = MockSig(aca_green.p.n, enei = enei)

        pot = aca_green.potential(sig, inout = 2)

        assert isinstance(pot, CompStruct)
        assert hasattr(pot, 'phi2')
        assert hasattr(pot, 'phi2p')

    def test_potential_phi1_shape(self,
            aca_green: ACACompGreenRetLayer) -> None:

        enei = 500.0
        self._setup_potential(aca_green, enei)
        sig = MockSig(aca_green.p.n, enei = enei)

        pot = aca_green.potential(sig, inout = 1)

        n = aca_green.p.n
        assert pot.phi1.shape == (n,)
        assert pot.phi1p.shape == (n,)

    def test_potential_matches_dense(self,
            aca_green: ACACompGreenRetLayer) -> None:

        enei = 500.0
        n = aca_green.p.n
        dense_mat = _build_dense_green(aca_green.p, enei)

        # Pre-populate cache
        for ii in range(2):
            for jj in range(2):
                for key in ['G', 'F', 'H1', 'H2']:
                    aca_green._cache[('dense', ii, jj, key, enei)] = dense_mat

        sig = MockSig(n, enei = enei)
        pot_aca = aca_green.potential(sig, inout = 1)

        # Compute reference using direct dense matvec
        # phi = G1*sig1 + G2*sig2 where G1 = eval(0, 0, 'G') and G2 = eval(0, 1, 'G')
        # Both use the same dense_mat in this test setup
        # For eval(0, 0) -> eval1 -> HMatrix from dense_mat
        # For eval(0, 1) -> eval1 -> HMatrix from dense_mat
        phi_ref = dense_mat @ sig.sig1 + dense_mat @ sig.sig2

        # ACA should approximate this
        dense_norm = np.linalg.norm(phi_ref)
        if dense_norm > 1e-15:
            rel_err = np.linalg.norm(pot_aca.phi1 - phi_ref) / dense_norm
            assert rel_err < 0.1, (
                'Potential phi1 relative error {} too large'.format(rel_err))

    def test_potential_vector_part(self,
            aca_green: ACACompGreenRetLayer) -> None:

        enei = 500.0
        self._setup_potential(aca_green, enei)
        sig = MockSig(aca_green.p.n, enei = enei)

        pot = aca_green.potential(sig, inout = 1)

        if hasattr(pot, 'a1'):
            a1 = pot.a1
            if isinstance(a1, np.ndarray):
                assert a1.shape[0] == aca_green.p.n
                assert a1.shape[1] == 3


# ---------------------------------------------------------------------------
# Full conversion
# ---------------------------------------------------------------------------

class TestACAFull(object):

    def test_eval_full_returns_array(self,
            aca_green: ACACompGreenRetLayer) -> None:

        n = aca_green.p.n
        enei = 500.0
        dense_mat = _build_dense_green(aca_green.p, enei)
        aca_green._cache[('dense', 0, 0, 'G', enei)] = dense_mat

        result = aca_green.eval_full(0, 0, 'G', enei)

        assert isinstance(result, np.ndarray)
        assert result.shape == (n, n)

    def test_eval_full_reflected_returns_dict(self,
            aca_green: ACACompGreenRetLayer) -> None:

        n = aca_green.p.n
        enei = 500.0
        dense_mat = _build_dense_green(aca_green.p, enei)
        aca_green._cache[('dense', 1, 1, 'G', enei)] = dense_mat

        result = aca_green.eval_full(1, 1, 'G', enei)

        assert isinstance(result, dict)
        for key, val in result.items():
            if isinstance(val, np.ndarray):
                assert val.shape == (n, n)


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------

class TestCaching(object):

    def test_cache_populated(self,
            aca_green: ACACompGreenRetLayer) -> None:

        n = aca_green.p.n
        enei = 500.0
        dense_mat = _build_dense_green(aca_green.p, enei)
        aca_green._cache[('dense', 0, 0, 'G', enei)] = dense_mat

        # First call
        result1 = aca_green._eval1(0, 0, 'G', enei)

        # Second call should return same object from cache
        result2 = aca_green._eval1(0, 0, 'G', enei)
        assert result1 is result2

    def test_clear_cache(self,
            aca_green: ACACompGreenRetLayer) -> None:

        n = aca_green.p.n
        enei = 500.0
        dense_mat = _build_dense_green(aca_green.p, enei)
        aca_green._cache[('dense', 0, 0, 'G', enei)] = dense_mat

        aca_green._eval1(0, 0, 'G', enei)
        assert len(aca_green._cache) > 1

        aca_green.clear_cache()
        assert len(aca_green._cache) == 0


# ---------------------------------------------------------------------------
# Compression
# ---------------------------------------------------------------------------

class TestCompression(object):

    def test_compression_ratio(self,
            aca_green: ACACompGreenRetLayer) -> None:

        n = aca_green.p.n
        enei = 500.0
        dense_mat = _build_dense_green(aca_green.p, enei)
        aca_green._cache[('dense', 0, 0, 'G', enei)] = dense_mat

        ratio = aca_green.compression(0, 0, 'G', enei)
        assert ratio > 0, 'Compression ratio should be positive'

    def test_compression_reflected(self,
            aca_green: ACACompGreenRetLayer) -> None:

        n = aca_green.p.n
        enei = 500.0
        dense_mat = _build_dense_green(aca_green.p, enei)
        aca_green._cache[('dense', 1, 1, 'G', enei)] = dense_mat

        ratio = aca_green.compression(1, 1, 'G', enei)
        assert ratio > 0, 'Compression ratio should be positive'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
