"""
BEM plotting class for MNPBEM.

MATLAB: @bemplot/
"""

import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from .valarray import ValArray
from .vecarray import VecArray
from .options import getbemoptions


class BemPlot(object):
    """
    Plotting value arrays and vector functions within MNPBEM.

    MATLAB: @bemplot

    Parameters
    ----------
    fun : callable, optional
        Plot function (default: np.real)
    scale : float, optional
        Scale factor for vector array (default: 1)
    sfun : callable, optional
        Scale function for vector array (default: identity)

    Methods
    -------
    plotval(p, val, **kwargs) -> None
    plotarrow(pos, vec, **kwargs) -> None
    plotcone(pos, vec, **kwargs) -> None
    plottrue(p, val=None, **kwargs) -> None
    refresh(*keys) -> None
    get(**kwargs) -> BemPlot (static)
    figname() -> str
    tight_caxis() -> None
    """

    # class-level storage for the current BemPlot instance
    # (replaces MATLAB figure UserData mechanism)
    _current = None

    def __init__(self, **kwargs: Any) -> None:
        self.var = []
        self.siz = None
        self.opt = {
            'ind': None,
            'fun': lambda x: np.real(x),
            'scale': 1.0,
            'sfun': lambda x: x}

        op = getbemoptions(kwargs)
        if 'fun' in op:
            self.opt['fun'] = op['fun']
        if 'scale' in op:
            self.opt['scale'] = op['scale']
        if 'sfun' in op:
            self.opt['sfun'] = op['sfun']

    @staticmethod
    def get(**kwargs: Any) -> 'BemPlot':
        """
        MATLAB: @bemplot/get.m

        Open new figure or get existing BemPlot instance.
        Creates a new BemPlot if none is current, otherwise
        reinitializes the existing one with the given options.
        """
        if BemPlot._current is None:
            obj = BemPlot(**kwargs)
            BemPlot._current = obj
        else:
            obj = BemPlot._current
            op = getbemoptions(kwargs)
            if 'fun' in op:
                obj.opt['fun'] = op['fun']
            if 'scale' in op:
                obj.opt['scale'] = op['scale']
            if 'sfun' in op:
                obj.opt['sfun'] = op['sfun']
        return obj

    @staticmethod
    def clear_current() -> None:
        """
        Clear the current BemPlot instance.
        Useful for resetting state between plots.
        """
        BemPlot._current = None

    def plotval(self, p: object, val: np.ndarray,
            **kwargs: Any) -> None:
        """
        MATLAB: @bemplot/plotval.m

        Plot value array on surface.
        """
        # initialization functions
        def inifun(p_arg: object) -> ValArray:
            return ValArray(p_arg, val)

        def inifun2(var: ValArray) -> ValArray:
            var.init2(val)
            return var

        self._plot(p, inifun, inifun2, **kwargs)

    def plottrue(self, p: object,
            val: Optional[np.ndarray] = None,
            **kwargs: Any) -> None:
        """
        MATLAB: @bemplot/plottrue.m

        Plot with true colors on surface.
        """
        def inifun(p_arg: object) -> ValArray:
            return ValArray(p_arg, val, truecolor = True)

        def inifun2(var: ValArray) -> ValArray:
            var.init2(val, truecolor = True)
            return var

        self._plot(p, inifun, inifun2, **kwargs)

    def plotarrow(self, pos: np.ndarray, vec: np.ndarray,
            **kwargs: Any) -> None:
        """
        MATLAB: @bemplot/plotarrow.m

        Plot vector array with arrows.
        """
        def inifun(pos_arg: np.ndarray) -> VecArray:
            return VecArray(pos_arg, vec, 'arrow')

        def inifun2(var: VecArray) -> VecArray:
            var.init2(vec, 'arrow')
            return var

        self._plot(pos, inifun, inifun2, **kwargs)

    def plotcone(self, pos: np.ndarray, vec: np.ndarray,
            **kwargs: Any) -> None:
        """
        MATLAB: @bemplot/plotcone.m

        Plot vector array with cones.
        """
        def inifun(pos_arg: np.ndarray) -> VecArray:
            return VecArray(pos_arg, vec, 'cone')

        def inifun2(var: VecArray) -> VecArray:
            var.init2(vec, 'cone')
            return var

        self._plot(pos, inifun, inifun2, **kwargs)

    def _plot(self, p: object,
            inifun: Callable, inifun2: Callable,
            **kwargs: Any) -> None:
        """
        MATLAB: @bemplot/plot.m

        Core plot function.
        """
        # initialize value array
        var = inifun(p)

        # handle size argument
        if hasattr(var, 'ispage') and var.ispage() and self.siz is not None:
            assert var.pagesize() == self.siz

        # has object been plotted before?
        ind = None
        for i, v in enumerate(self.var):
            if v.isbase(p):
                ind = i
                break

        if ind is None:
            ind = len(self.var)
            self.var.append(var)
        else:
            self.var[ind] = inifun2(self.var[ind])

        # handle paging
        if hasattr(self.var[ind], 'ispage') and self.var[ind].ispage() and self.siz is None:
            self.siz = self.var[ind].pagesize()
            self.opt['ind'] = 0

        # plot
        self.var[ind].plot(self.opt, **kwargs)

        # store as current instance
        BemPlot._current = self

    def refresh(self, *keys: str) -> None:
        """
        MATLAB: @bemplot/refresh.m

        Refresh value and vector plots.
        """
        for i, var in enumerate(self.var):
            if var.depends(*keys):
                var.plot(self.opt)
                self.var[i] = var

    def set_opt(self, **kwargs: Any) -> None:
        """
        MATLAB: @bemplot/set.m

        Set plot options and refresh.
        """
        keys = []
        if 'ind' in kwargs:
            ind = kwargs['ind']
            if isinstance(ind, (list, tuple)) and self.siz is not None:
                ind = np.ravel_multi_index(ind, self.siz)
            self.opt['ind'] = ind
            keys.append('ind')

        if 'fun' in kwargs:
            self.opt['fun'] = kwargs['fun']
            keys.append('fun')
        if 'scale' in kwargs:
            self.opt['scale'] = kwargs['scale']
            keys.append('scale')
        if 'sfun' in kwargs:
            self.opt['sfun'] = kwargs['sfun']
            keys.append('sfun')

        if keys:
            self.refresh(*keys)

    def figname(self) -> str:
        """
        MATLAB: @bemplot/private/figname.m

        Generate figure title string based on current plot options.
        """
        fun = self.opt['fun']

        # determine function name
        # try to detect common functions by testing behavior
        try:
            test_val = np.array([1.0 + 1.0j])
            result = fun(test_val)
            if np.isclose(result[0], 1.0):
                name = '(real)'
            elif np.isclose(result[0], 1.0j) or np.isclose(result[0], 1.0):
                name = '(imag)'
            elif np.isclose(np.abs(result[0]), np.sqrt(2.0)):
                name = '(abs)'
            else:
                name = '(fun)'
        except Exception:
            name = '(fun)'

        if self.siz is not None and self.opt['ind'] is not None:
            ind_val = self.opt['ind']
            ind_tuple = np.unravel_index(ind_val, self.siz)
            # MATLAB uses 1-based indexing, Python uses 0-based
            ind_str = str(list(ind_tuple))
            siz_str = str(list(self.siz))
            name = 'Element {} of {}  {}'.format(ind_str, siz_str, name)

        return name

    def tight_caxis(self) -> Tuple[Optional[float], Optional[float]]:
        """
        MATLAB: @bemplot/private/contextmenu.m/caxfun

        Compute tight color axis limits from all value/vector arrays.

        Returns
        -------
        clim : tuple of (cmin, cmax) or (None, None)
        """
        cmin_vals = []
        cmax_vals = []
        for var in self.var:
            mn = var.min_val(self.opt)
            mx = var.max_val(self.opt)
            if mn is not None:
                cmin_vals.append(mn)
            if mx is not None:
                cmax_vals.append(mx)

        if len(cmin_vals) == 0:
            return (None, None)

        return (min(cmin_vals), max(cmax_vals))

    def __repr__(self) -> str:
        return 'BemPlot(nvars={}, siz={})'.format(
            len(self.var), self.siz)
