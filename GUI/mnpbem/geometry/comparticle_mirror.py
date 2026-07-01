from __future__ import annotations

import numpy as np
from typing import Optional, List, Tuple, Any, Callable, Union
from .particle import Particle
from .comparticle import ComParticle


class CompStructMirror(object):
    """Structure for compound of points or particles with mirror symmetry.

    MATLAB: @compstructmirror
    """

    def __init__(self,
            p: 'ComParticleMirror',
            enei: float,
            fun: Optional[Callable] = None) -> None:
        self.p = p
        self.enei = enei
        self.fun = fun
        self.val = []  # type: List[Any]

    def full(self) -> Any:
        if self.fun is not None:
            return self.fun(self)
        return self

    def expand(self) -> Tuple:
        """Expand structure to full particle size using mirror symmetry.

        MATLAB: @compstructmirror/expand.m
        """
        from ..greenfun import CompStruct

        if len(self.val) == 0:
            return ()

        names = [k for k in self.val[0].val.keys()
                 if k != 'symval']

        n_out = len(self.val)
        results = []
        for i in range(n_out):
            results.append(CompStruct(self.p.full(), self.enei))

        for i in range(n_out):
            for name in names:
                val1 = getattr(self.val[i], name, None)
                if val1 is None:
                    continue

                symval = self.val[i].symval
                val2 = val1.copy() if isinstance(val1, np.ndarray) else val1

                # scalar fields
                scalar_names = {'phi', 'phip', 'phi1', 'phi2', 'phi1p', 'phi2p',
                                'sig', 'sig1', 'sig2'}
                # vector fields
                vector_names = {'a1', 'a1p', 'a2', 'a2p', 'e', 'h', 'h1', 'h2'}

                if name in scalar_names:
                    n_sym = symval.shape[1]
                    n_base = val1.shape[0]
                    total_len = n_base * n_sym
                    if val1.ndim == 1:
                        expanded = np.empty(total_len, dtype = val1.dtype)
                        expanded[:n_base] = val1
                        for k in range(1, n_sym):
                            expanded[k * n_base:(k + 1) * n_base] = symval[-1, k] * val1
                    else:
                        expanded = np.empty((total_len,) + val1.shape[1:], dtype = val1.dtype)
                        expanded[:n_base] = val1
                        for k in range(1, n_sym):
                            expanded[k * n_base:(k + 1) * n_base] = symval[-1, k] * val1
                    val2 = expanded

                elif name in vector_names:
                    n_sym = symval.shape[1]
                    n_base = val1.shape[0]
                    total_len = n_base * n_sym
                    if val1.ndim == 2:
                        expanded = np.empty((total_len, 3), dtype = val1.dtype)
                        expanded[:n_base] = val1
                        for k in range(1, n_sym):
                            for l_idx in range(3):
                                expanded[k * n_base:(k + 1) * n_base, l_idx] = symval[l_idx, k] * val1[:, l_idx]
                    elif val1.ndim == 3:
                        expanded = np.empty((total_len, 3) + val1.shape[2:], dtype = val1.dtype)
                        expanded[:n_base] = val1
                        for k in range(1, n_sym):
                            for l_idx in range(3):
                                expanded[k * n_base:(k + 1) * n_base, l_idx] = symval[l_idx, k] * val1[:, l_idx]
                    else:
                        expanded = val1
                    val2 = expanded

                setattr(results[i], name, val2)

        return tuple(results)

    def __repr__(self) -> str:
        return 'CompStructMirror(p={}, enei={}, nval={})'.format(
            self.p, self.enei, len(self.val))


class ComParticleMirror(object):
    """Compound of particles with mirror symmetry in a dielectric environment.

    MATLAB: @comparticlemirror

    Mirror symmetry reduces computation by exploiting geometric symmetry planes.
    sigma=+1 (symmetric), sigma=-1 (antisymmetric).

    Parameters
    ----------
    eps : list
        Cell array of dielectric functions
    particles : list of Particle
        Cell array of n particles
    inout : ndarray
        Index to medium EPS
    sym : str
        Symmetry key: 'x', 'y', or 'xy'
    closed_args : tuple, optional
        Arguments passed to closed method
    """

    name = 'bemparticle'
    needs = ['sym']

    def __init__(self,
            eps: List,
            particles: List[Particle],
            inout: Any,
            sym: Any = 'x',
            closed_args: Optional[Tuple] = None,
            **kwargs: Any) -> None:

        # Handle MATLAB-style calling: closed_args as 4th positional arg
        if not isinstance(sym, str):
            if closed_args is not None:
                raise ValueError('[error] Cannot pass both numeric sym and closed_args')
            closed_args = tuple(sym) if hasattr(sym, '__iter__') else (sym,)
            sym = kwargs.pop('sym', 'x')

        if sym not in ('x', 'y', 'xy'):
            raise ValueError(
                "ComParticleMirror: 'sym' must be one of 'x', 'y', 'xy'; got {!r}.".format(sym))

        self.eps = eps
        self.p = list(particles)
        self.inout = np.atleast_2d(inout)
        self.sym = sym
        self._mask = list(range(len(self.p)))

        # symmetry table
        self._init_symtable()

        # build full particle via mirror symmetry
        self._build_full(**kwargs)

        # closed surfaces
        if closed_args is not None and len(closed_args) > 0:
            self.set_closed(*closed_args)
        else:
            self.pfull.closed = [None] * len(self.pfull.p)

    def _init_symtable(self) -> None:
        if self.sym in ('x', 'y'):
            self.symtable = np.array([[1, 1],
                                      [1, -1]], dtype = np.float64)
        elif self.sym == 'xy':
            self.symtable = np.array([[1,  1,  1,  1],
                                      [1,  1, -1, -1],
                                      [1, -1,  1, -1],
                                      [1, -1, -1,  1]], dtype = np.float64)

    def _build_full(self, **kwargs: Any) -> None:
        """Build full particle using mirror symmetry operations.

        MATLAB: @comparticlemirror/init.m lines 49-64
        """
        mirror_map = {'x': 0, 'y': 1, 'xy': None}

        p_list = list(self.p)
        inout_list = self.inout.tolist()

        # mirror in x direction
        if self.sym in ('x', 'xy'):
            orig_len = len(p_list)
            for i in range(orig_len):
                p_list.append(p_list[i].flip(0))  # flip x (0-indexed)
                inout_list.append(inout_list[i])

        # mirror in y direction
        if self.sym in ('y', 'xy'):
            orig_len = len(p_list)
            for i in range(orig_len):
                p_list.append(p_list[i].flip(1))  # flip y (0-indexed)
                inout_list.append(inout_list[i])

        inout_arr = np.array(inout_list)
        self.pfull = ComParticle(self.eps, p_list, inout_arr, **kwargs)

    def set_closed(self, *args: Any) -> None:
        """Indicate closed surfaces of particles for Green function evaluation.

        MATLAB: @comparticlemirror/closed.m
        """
        n_sym_cols = self.symtable.shape[1]
        n_full = len(self.pfull.p)
        # MATLAB reshape is column-major, so use Fortran order here so that
        # particle k's mirror copies occupy row k-1 of `ind`.
        ind = np.arange(1, n_full + 1).reshape(-1, n_sym_cols, order='F')

        if self.pfull.closed is None:
            self.pfull.closed = [None] * n_full

        for arg in args:
            if not isinstance(arg, (list, tuple)) or not isinstance(arg[0], (list, Particle)):
                # simple index list
                if np.isscalar(arg):
                    indices = [arg]
                else:
                    indices = list(arg)

                sign_arr = np.sign(indices)
                abs_arr = np.abs(indices)
                tab_rows = []
                for idx_val, s_val in zip(abs_arr, sign_arr):
                    row = ind[idx_val - 1, :]
                    tab_rows.append(s_val * row)

                tab = np.array(tab_rows).ravel()
                for j in tab:
                    self.pfull.closed[abs(j) - 1] = tab.tolist()
            else:
                idx = arg[0]
                tab = ind[idx - 1, :].ravel()
                particles_to_concat = []
                for t in tab:
                    particles_to_concat.append(self.pfull.p[t - 1])
                for extra_p in arg[1:]:
                    particles_to_concat.append(extra_p)
                combined = particles_to_concat[0]
                for pp in particles_to_concat[1:]:
                    combined = combined + pp
                for j in range(len(tab)):
                    self.pfull.closed[j] = combined

    def full(self) -> ComParticle:
        """Return full particle produced with mirror symmetry.

        MATLAB: full(obj) -> obj.pfull
        """
        return self.pfull

    def closedparticle(self, ind: int) -> Tuple:
        """Return particle with closed surface for particle ind.

        MATLAB: closedparticle(obj, ind)

        Note: Always returns loc=None for mirror particles to force the
        temporary Green function path in _handle_closed_surfaces.
        The loc indices from pfull.closedparticle() reference the full
        expanded particle and cannot be used to directly index the
        mirror-reduced Green function matrix F.
        """
        full, dir_val, _loc = self.pfull.closedparticle(ind)
        return full, dir_val, None

    def symindex(self, tab: np.ndarray) -> int:
        """Index of symmetry values within symmetry table.

        MATLAB: symindex(obj, tab)

        Parameters
        ----------
        tab : ndarray
            Two or four symmetry values

        Returns
        -------
        ind : int
            Index to symmetry table (0-indexed)
        """
        tab = np.atleast_1d(tab)
        for i in range(self.symtable.shape[0]):
            if np.allclose(self.symtable[i, :], tab):
                return i
        raise ValueError('[error] Symmetry values {} not found in table'.format(tab))

    def symvalue(self, key: Union[str, List[str]]) -> np.ndarray:
        """Symmetry values for given key.

        MATLAB: symvalue(obj, key)

        Parameters
        ----------
        key : str or list of str
            '+', '-' for sym = {'x', 'y'}, and
            '++', '+-', '-+', '--' for sym = 'xy'

        Returns
        -------
        val : ndarray
            Value array, shape (n_keys, n_sym_cols)
        """
        if isinstance(key, (list, tuple)):
            rows = []
            for k in key:
                rows.append(self._symvalue_single(k))
            return np.array(rows, dtype = np.float64)
        else:
            return self._symvalue_single(key).reshape(1, -1)

    def _symvalue_single(self, key: str) -> np.ndarray:
        sym_map = {
            '+':  np.array([1,  1], dtype = np.float64),
            '-':  np.array([1, -1], dtype = np.float64),
            '++': np.array([1,  1,  1,  1], dtype = np.float64),
            '+-': np.array([1,  1, -1, -1], dtype = np.float64),
            '-+': np.array([1, -1,  1, -1], dtype = np.float64),
            '--': np.array([1, -1, -1,  1], dtype = np.float64),
        }
        if key not in sym_map:
            raise ValueError('[error] Unknown symmetry key: {}'.format(key))
        return sym_map[key]

    def set_mask(self, ind: Any) -> 'ComParticleMirror':
        """Mask out particles indicated by ind.

        MATLAB: mask(obj, ind)
        """
        if np.isscalar(ind):
            ind = [ind]
        self._mask = [i - 1 for i in ind]

        # also mask the full particle
        ip = np.arange(1, len(self.pfull.p) + 1).reshape(len(self.p), -1)
        full_ind = ip[np.array([i - 1 for i in ind]), :].ravel()
        self.pfull.set_mask(full_ind.tolist())
        return self

    # delegate properties to underlying comparticle
    @property
    def nvec(self) -> np.ndarray:
        return self.pfull.pc.nvec[:self._half_nfaces]

    @property
    def pos(self) -> np.ndarray:
        return self.pfull.pc.pos[:self._half_nfaces]

    @property
    def area(self) -> np.ndarray:
        return self.pfull.pc.area[:self._half_nfaces]

    @property
    def _half_nfaces(self) -> int:
        n_sym = self.symtable.shape[1]
        return self.pfull.nfaces // n_sym

    @property
    def nfaces(self) -> int:
        return self._half_nfaces

    @property
    def n(self) -> int:
        return self.nfaces

    @property
    def np(self) -> int:
        return self.pfull.np

    @property
    def mask(self) -> np.ndarray:
        mask_arr = np.zeros(len(self.p), dtype = bool)
        for i in self._mask:
            mask_arr[i] = True
        return mask_arr

    def eps1(self, enei: float) -> np.ndarray:
        full_eps = self.pfull.eps1(enei)
        return full_eps[:self._half_nfaces]

    def eps2(self, enei: float) -> np.ndarray:
        full_eps = self.pfull.eps2(enei)
        return full_eps[:self._half_nfaces]

    @property
    def index(self) -> np.ndarray:
        return self.pfull.index[:self._half_nfaces]

    def index_func(self, particle_indices: Any) -> np.ndarray:
        return self.pfull.index_func(particle_indices)

    @property
    def verts(self) -> np.ndarray:
        return self.pfull.verts

    def __repr__(self) -> str:
        return 'ComParticleMirror(sym={}, nparticles={}, nfaces_half={})'.format(
            self.sym, len(self.p), self.nfaces)
