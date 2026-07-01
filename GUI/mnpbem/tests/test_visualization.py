import sys
import os

import numpy as np
import pytest
from typing import Any, Dict, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from mnpbem.misc.bemplot import BemPlot
from mnpbem.misc.vecarray import VecArray, _cone_plot
from mnpbem.misc.valarray import ValArray
from mnpbem.misc.plotting import arrowplot, coneplot, coneplot2, mycolormap


# ============================================================================
# Helper: minimal particle-like object for testing ValArray / plotval
# ============================================================================

class _MockParticle(object):

    def __init__(self, nverts: int = 8,
            nfaces: int = 6) -> None:
        # simple cube-like vertices
        self.nverts = nverts
        self.n = nfaces
        self.verts = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype = float)
        # triangular faces (6 faces from 2 triangles each = 12, but keep simple)
        self.faces = np.array([
            [0, 1, 2, np.nan],
            [0, 2, 3, np.nan],
            [4, 5, 6, np.nan],
            [4, 6, 7, np.nan],
            [0, 1, 5, np.nan],
            [0, 5, 4, np.nan]], dtype = float)
        self.area = np.ones(nfaces) * 0.5


# ============================================================================
# Tests: BemPlot creation and basic properties
# ============================================================================

class TestBemPlotCreation(object):

    def test_default_creation(self) -> None:
        bp = BemPlot()
        assert bp is not None
        assert len(bp.var) == 0
        assert bp.siz is None
        assert bp.opt['ind'] is None
        assert bp.opt['scale'] == 1.0

    def test_creation_with_kwargs(self) -> None:
        bp = BemPlot(scale = 2.0)
        assert bp.opt['scale'] == 2.0

    def test_creation_with_fun(self) -> None:
        bp = BemPlot(fun = np.abs)
        test_val = np.array([-1.0 + 2.0j])
        result = bp.opt['fun'](test_val)
        assert np.isclose(result[0], np.sqrt(5.0))

    def test_creation_with_sfun(self) -> None:
        sfun = lambda x: np.sqrt(x)
        bp = BemPlot(sfun = sfun)
        assert np.isclose(bp.opt['sfun'](4.0), 2.0)

    def test_repr(self) -> None:
        bp = BemPlot()
        s = repr(bp)
        assert 'BemPlot' in s
        assert 'nvars=0' in s


# ============================================================================
# Tests: BemPlot.get static factory
# ============================================================================

class TestBemPlotGet(object):

    def setup_method(self) -> None:
        BemPlot.clear_current()

    def teardown_method(self) -> None:
        BemPlot.clear_current()

    def test_get_creates_new(self) -> None:
        bp = BemPlot.get()
        assert bp is not None
        assert isinstance(bp, BemPlot)

    def test_get_returns_same_instance(self) -> None:
        bp1 = BemPlot.get()
        bp2 = BemPlot.get()
        assert bp1 is bp2

    def test_get_reinitializes_options(self) -> None:
        bp1 = BemPlot.get(scale = 1.0)
        bp2 = BemPlot.get(scale = 3.0)
        assert bp1 is bp2
        assert bp2.opt['scale'] == 3.0

    def test_clear_current(self) -> None:
        bp1 = BemPlot.get()
        BemPlot.clear_current()
        bp2 = BemPlot.get()
        assert bp1 is not bp2


# ============================================================================
# Tests: BemPlot.figname
# ============================================================================

class TestBemPlotFigname(object):

    def test_figname_default(self) -> None:
        bp = BemPlot()
        name = bp.figname()
        assert '(real)' in name

    def test_figname_with_abs(self) -> None:
        bp = BemPlot(fun = np.abs)
        name = bp.figname()
        assert '(abs)' in name

    def test_figname_with_paging(self) -> None:
        bp = BemPlot()
        bp.siz = (3, 4)
        bp.opt['ind'] = 5
        name = bp.figname()
        assert 'Element' in name


# ============================================================================
# Tests: BemPlot.tight_caxis
# ============================================================================

class TestBemPlotTightCaxis(object):

    def test_tight_caxis_empty(self) -> None:
        bp = BemPlot()
        cmin, cmax = bp.tight_caxis()
        assert cmin is None
        assert cmax is None


# ============================================================================
# Tests: BemPlot.set_opt
# ============================================================================

class TestBemPlotSetOpt(object):

    def test_set_scale(self) -> None:
        bp = BemPlot()
        bp.set_opt(scale = 5.0)
        assert bp.opt['scale'] == 5.0

    def test_set_fun(self) -> None:
        bp = BemPlot()
        bp.set_opt(fun = np.imag)
        test_val = np.array([1.0 + 2.0j])
        assert np.isclose(bp.opt['fun'](test_val)[0], 2.0)


# ============================================================================
# Tests: plotval on particle surface
# ============================================================================

class TestPlotVal(object):

    def setup_method(self) -> None:
        plt.close('all')
        BemPlot.clear_current()

    def teardown_method(self) -> None:
        plt.close('all')
        BemPlot.clear_current()

    def test_plotval_creates_figure(self) -> None:
        p = _MockParticle()
        val = np.random.rand(p.nverts)
        bp = BemPlot()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection = '3d')
        bp.plotval(p, val)
        assert len(bp.var) == 1
        assert isinstance(bp.var[0], ValArray)

    def test_plotval_replot_same_particle(self) -> None:
        p = _MockParticle()
        val1 = np.random.rand(p.nverts)
        val2 = np.random.rand(p.nverts)
        bp = BemPlot()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection = '3d')
        bp.plotval(p, val1)
        bp.plotval(p, val2)
        # should still be one var entry (reused)
        assert len(bp.var) == 1

    def test_plottrue_creates_figure(self) -> None:
        p = _MockParticle()
        bp = BemPlot()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection = '3d')
        bp.plottrue(p)
        assert len(bp.var) == 1
        assert bp.var[0].truecolor is True


# ============================================================================
# Tests: plotarrow (arrow visualization)
# ============================================================================

class TestPlotArrow(object):

    def setup_method(self) -> None:
        plt.close('all')
        BemPlot.clear_current()

    def teardown_method(self) -> None:
        plt.close('all')
        BemPlot.clear_current()

    def test_plotarrow_no_error(self) -> None:
        pos = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype = float)
        vec = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype = float)
        bp = BemPlot()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection = '3d')
        bp.plotarrow(pos, vec)
        assert len(bp.var) == 1
        assert isinstance(bp.var[0], VecArray)
        assert bp.var[0].mode == 'arrow'

    def test_plotarrow_with_scale(self) -> None:
        pos = np.array([[0, 0, 0], [1, 1, 1]], dtype = float)
        vec = np.array([[1, 0, 0], [0, 1, 0]], dtype = float)
        bp = BemPlot(scale = 2.0)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection = '3d')
        bp.plotarrow(pos, vec)
        assert bp.opt['scale'] == 2.0

    def test_plotarrow_replot(self) -> None:
        pos = np.array([[0, 0, 0], [1, 0, 0]], dtype = float)
        vec1 = np.array([[1, 0, 0], [0, 1, 0]], dtype = float)
        vec2 = np.array([[0, 0, 1], [1, 1, 0]], dtype = float)
        bp = BemPlot()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection = '3d')
        bp.plotarrow(pos, vec1)
        bp.plotarrow(pos, vec2)
        # same positions => reuse entry
        assert len(bp.var) == 1

    def test_arrowplot_standalone(self) -> None:
        pos = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype = float)
        vec = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype = float)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection = '3d')
        bp = arrowplot(pos, vec)
        assert isinstance(bp, BemPlot)
        assert len(bp.var) == 1


# ============================================================================
# Tests: plotcone (cone visualization)
# ============================================================================

class TestPlotCone(object):

    def setup_method(self) -> None:
        plt.close('all')
        BemPlot.clear_current()

    def teardown_method(self) -> None:
        plt.close('all')
        BemPlot.clear_current()

    def test_plotcone_no_error(self) -> None:
        pos = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype = float)
        vec = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype = float)
        bp = BemPlot()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection = '3d')
        bp.plotcone(pos, vec)
        assert len(bp.var) == 1
        assert isinstance(bp.var[0], VecArray)
        assert bp.var[0].mode == 'cone'

    def test_plotcone_with_scale(self) -> None:
        pos = np.array([[0, 0, 0], [2, 2, 2]], dtype = float)
        vec = np.array([[1, 0, 0], [0, 0, 1]], dtype = float)
        bp = BemPlot(scale = 0.5)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection = '3d')
        bp.plotcone(pos, vec)
        assert bp.opt['scale'] == 0.5

    def test_plotcone_replot(self) -> None:
        pos = np.array([[0, 0, 0], [1, 0, 0]], dtype = float)
        vec1 = np.array([[1, 0, 0], [0, 1, 0]], dtype = float)
        vec2 = np.array([[0, 0, 1], [1, 1, 0]], dtype = float)
        bp = BemPlot()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection = '3d')
        bp.plotcone(pos, vec1)
        bp.plotcone(pos, vec2)
        # same positions => reuse entry
        assert len(bp.var) == 1

    def test_coneplot_standalone(self) -> None:
        pos = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype = float)
        vec = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype = float)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection = '3d')
        bp = coneplot(pos, vec)
        assert isinstance(bp, BemPlot)
        assert len(bp.var) == 1

    def test_coneplot2_standalone(self) -> None:
        pos = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype = float)
        vec = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype = float)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection = '3d')
        h = coneplot2(pos, vec)
        # returns the Poly3DCollection or None
        assert h is None or isinstance(h, Poly3DCollection)


# ============================================================================
# Tests: _cone_plot helper function
# ============================================================================

class TestConePlotHelper(object):

    def setup_method(self) -> None:
        plt.close('all')

    def teardown_method(self) -> None:
        plt.close('all')

    def test_cone_plot_basic(self) -> None:
        pos = np.array([[0, 0, 0], [1, 0, 0]], dtype = float)
        vec = np.array([[0, 0, 1], [1, 0, 0]], dtype = float)
        vec_len = np.sqrt(np.sum(vec ** 2, axis = 1))
        scale = np.ones(pos.shape[0])
        fig = plt.figure()
        ax = fig.add_subplot(111, projection = '3d')
        h = _cone_plot(pos, vec, vec_len, scale)
        assert h is not None
        assert isinstance(h, Poly3DCollection)

    def test_cone_plot_zero_vectors(self) -> None:
        pos = np.array([[0, 0, 0], [1, 0, 0]], dtype = float)
        vec = np.array([[0, 0, 0], [0, 0, 0]], dtype = float)
        vec_len = np.zeros(2)
        scale = np.ones(2)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection = '3d')
        h = _cone_plot(pos, vec, vec_len, scale)
        # all zero vectors -> no polygons -> returns None
        assert h is None

    def test_cone_plot_single_vector(self) -> None:
        pos = np.array([[0, 0, 0]], dtype = float)
        vec = np.array([[0, 0, 1]], dtype = float)
        vec_len = np.array([1.0])
        scale = np.array([1.0])
        fig = plt.figure()
        ax = fig.add_subplot(111, projection = '3d')
        h = _cone_plot(pos, vec, vec_len, scale)
        assert h is not None


# ============================================================================
# Tests: VecArray
# ============================================================================

class TestVecArray(object):

    def test_creation(self) -> None:
        pos = np.array([[0, 0, 0], [1, 0, 0]], dtype = float)
        vec = np.array([[1, 0, 0], [0, 1, 0]], dtype = float)
        va = VecArray(pos, vec, 'arrow')
        assert va.mode == 'arrow'
        assert va.pos.shape == (2, 3)

    def test_isbase_match(self) -> None:
        pos = np.array([[0, 0, 0], [1, 0, 0]], dtype = float)
        vec = np.array([[1, 0, 0], [0, 1, 0]], dtype = float)
        va = VecArray(pos, vec)
        assert va.isbase(pos) == True

    def test_isbase_mismatch(self) -> None:
        pos1 = np.array([[0, 0, 0]], dtype = float)
        pos2 = np.array([[1, 1, 1]], dtype = float)
        vec = np.array([[1, 0, 0]], dtype = float)
        va = VecArray(pos1, vec)
        assert va.isbase(pos2) == False

    def test_ispage_false(self) -> None:
        pos = np.array([[0, 0, 0]], dtype = float)
        vec = np.array([[1, 0, 0]], dtype = float)
        va = VecArray(pos, vec)
        assert va.ispage() is False

    def test_ispage_true(self) -> None:
        pos = np.array([[0, 0, 0]], dtype = float)
        # (1, 3, 2) - multi-dimensional
        vec = np.random.rand(1, 3, 2)
        va = VecArray(pos, vec)
        assert va.ispage() is True

    def test_depends(self) -> None:
        pos = np.array([[0, 0, 0]], dtype = float)
        vec = np.array([[1, 0, 0]], dtype = float)
        va = VecArray(pos, vec)
        assert va.depends('scale') is True
        assert va.depends('fun') is True
        # not paged, so 'ind' should not depend
        assert va.depends('ind') is False

    def test_init2(self) -> None:
        pos = np.array([[0, 0, 0]], dtype = float)
        vec1 = np.array([[1, 0, 0]], dtype = float)
        vec2 = np.array([[0, 1, 0]], dtype = float)
        va = VecArray(pos, vec1)
        va.init2(vec2, 'cone')
        assert np.allclose(va.vec, vec2)
        assert va.mode == 'cone'

    def test_get_vec(self) -> None:
        pos = np.array([[0, 0, 0]], dtype = float)
        vec = np.array([[1.0 + 2.0j, 0, 0]], dtype = complex)
        va = VecArray(pos, vec)
        opt = {'fun': np.real, 'ind': None, 'scale': 1.0, 'sfun': lambda x: x}
        result = va.get_vec(opt)
        assert np.allclose(result, np.array([[1.0, 0, 0]]))


# ============================================================================
# Tests: ValArray
# ============================================================================

class TestValArray(object):

    def test_creation(self) -> None:
        p = _MockParticle()
        val = np.random.rand(p.nverts)
        va = ValArray(p, val)
        assert va.truecolor is False
        assert va.val.shape[0] == p.nverts

    def test_creation_no_val(self) -> None:
        p = _MockParticle()
        va = ValArray(p)
        # default golden color => truecolor True
        assert va.truecolor is True
        assert va.val.shape == (p.nverts, 3)

    def test_isbase(self) -> None:
        p = _MockParticle()
        val = np.random.rand(p.nverts)
        va = ValArray(p, val)
        assert va.isbase(p) == True

    def test_ispage_false(self) -> None:
        p = _MockParticle()
        val = np.random.rand(p.nverts)
        va = ValArray(p, val)
        assert va.ispage() is False

    def test_depends(self) -> None:
        p = _MockParticle()
        val = np.random.rand(p.nverts)
        va = ValArray(p, val)
        assert va.depends('fun') is True
        assert va.depends('ind') is False  # not paged

    def test_min_max(self) -> None:
        p = _MockParticle()
        val = np.arange(p.nverts, dtype = float)
        va = ValArray(p, val)
        opt = {'fun': np.real, 'ind': None}
        assert va.min_val(opt) == 0.0
        assert va.max_val(opt) == float(p.nverts - 1)


# ============================================================================
# Tests: mycolormap
# ============================================================================

class TestMyColormap(object):

    def test_std1(self) -> None:
        cmap = mycolormap('std:1', n = 50)
        assert cmap.shape == (50, 3)
        assert np.all(cmap >= 0.0)
        assert np.all(cmap <= 1.0)

    def test_cen1(self) -> None:
        cmap = mycolormap('cen:1', n = 100)
        assert cmap.shape == (100, 3)

    def test_invalid_key(self) -> None:
        with pytest.raises(ValueError):
            mycolormap('invalid_key')


# ============================================================================
# Tests: mixed arrow and cone on same BemPlot
# ============================================================================

class TestMixedPlots(object):

    def setup_method(self) -> None:
        plt.close('all')
        BemPlot.clear_current()

    def teardown_method(self) -> None:
        plt.close('all')
        BemPlot.clear_current()

    def test_arrow_and_cone_together(self) -> None:
        pos1 = np.array([[0, 0, 0], [1, 0, 0]], dtype = float)
        vec1 = np.array([[1, 0, 0], [0, 1, 0]], dtype = float)
        pos2 = np.array([[2, 2, 2], [3, 3, 3]], dtype = float)
        vec2 = np.array([[0, 0, 1], [1, 1, 1]], dtype = float)

        bp = BemPlot()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection = '3d')
        bp.plotarrow(pos1, vec1)
        bp.plotcone(pos2, vec2)
        assert len(bp.var) == 2
        assert bp.var[0].mode == 'arrow'
        assert bp.var[1].mode == 'cone'

    def test_val_and_arrow_together(self) -> None:
        p = _MockParticle()
        val = np.random.rand(p.nverts)
        pos = np.array([[0, 0, 0], [1, 0, 0]], dtype = float)
        vec = np.array([[1, 0, 0], [0, 1, 0]], dtype = float)

        bp = BemPlot()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection = '3d')
        bp.plotval(p, val)
        bp.plotarrow(pos, vec)
        assert len(bp.var) == 2
        assert isinstance(bp.var[0], ValArray)
        assert isinstance(bp.var[1], VecArray)
