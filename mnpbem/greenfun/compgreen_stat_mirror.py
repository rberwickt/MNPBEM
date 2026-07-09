import numpy as np
from typing import Optional, List, Tuple, Any, Union

from .compgreen_stat import CompGreenStat, CompStruct
from ..geometry.comparticle_mirror import CompStructMirror


class CompGreenStatMirror(object):
    """Quasistatic Green function for composite particles with mirror symmetry.

    MATLAB: @compgreenstatmirror

    Parameters
    ----------
    p : ComParticleMirror
        Particle with mirror symmetry
    """

    name = 'greenfunction'
    needs = {'sim': 'stat', 'sym': True}

    def __init__(self,
            p: Any,
            _dummy: Any = None,
            **options: Any) -> None:
        self.p = p
        # Green function between half particle and full particle.
        # Closed surface correction is handled inside CompGreenStat._init
        # via the loc=None path (temporary Green function approach).
        self.g = CompGreenStat(p, p.full(), **options)

    @property
    def deriv(self) -> str:
        return self.g.deriv

    @property
    def con(self) -> Any:
        return self.g.con

    def eval(self,
            key: str) -> List:
        """Evaluate quasistatic Green function with mirror symmetry.

        MATLAB: @compgreenstatmirror/eval.m

        Parameters
        ----------
        key : str
            G, F, H1, H2, Gp, H1p, H2p

        Returns
        -------
        g : list
            List of contracted Green function matrices for each symmetry value
        """
        mat = getattr(self.g, key)
        tab = self.p.symtable

        n_sym = tab.shape[0]
        g = [0.0] * n_sym

        if isinstance(mat, np.ndarray):
            if mat.ndim == 2:
                # G, F, H1, H2: (n, n*n_sym_cols)
                n = mat.shape[0]
                n_blocks = mat.shape[1] // n
                sub_mats = []
                for b in range(n_blocks):
                    sub_mats.append(mat[:, b * n:(b + 1) * n])

                for i_sym in range(n_sym):
                    g[i_sym] = np.zeros_like(sub_mats[0])
                    for j_block in range(tab.shape[1]):
                        g[i_sym] = g[i_sym] + tab[i_sym, j_block] * sub_mats[j_block]

            elif mat.ndim == 3:
                # Gp, H1p, H2p: (n, 3, n*n_sym_cols)
                n = mat.shape[0]
                n_blocks = mat.shape[2] // n
                sub_mats = []
                for b in range(n_blocks):
                    sub_mats.append(mat[:, :, b * n:(b + 1) * n])

                for i_sym in range(n_sym):
                    g[i_sym] = np.zeros_like(sub_mats[0])
                    for j_block in range(tab.shape[1]):
                        g[i_sym] = g[i_sym] + tab[i_sym, j_block] * sub_mats[j_block]

        return g

    @property
    def G(self) -> List:
        return self.eval('G')

    @property
    def F(self) -> List:
        return self.eval('F')

    @property
    def H1(self) -> List:
        return self.eval('H1')

    @property
    def H2(self) -> List:
        return self.eval('H2')

    @property
    def Gp(self) -> List:
        return self.eval('Gp')

    @property
    def H1p(self) -> List:
        return self.eval('H1p')

    @property
    def H2p(self) -> List:
        return self.eval('H2p')

    def field(self,
            sig: CompStructMirror,
            inout: int = 1) -> CompStructMirror:
        """Electric field inside/outside of particle surface.

        MATLAB: @compgreenstatmirror/field.m

        Parameters
        ----------
        sig : CompStructMirror
            Surface charges
        inout : int
            Fields inside (1) or outside (2)

        Returns
        -------
        field : CompStructMirror
            Electric field
        """
        assert self.g.deriv == 'cart'

        field_out = CompStructMirror(sig.p, sig.enei, sig.fun)

        # derivative of Green function
        Gp = self.Gp

        # divergent part for diagonal Green function elements
        n = self.p.nfaces
        nvec = self.p.nvec
        sign = 1.0 if inout == 1 else -1.0
        div = sign * 2 * np.pi * _outer_nvec_eye(nvec, n)

        # add divergent part
        H = []
        for gp_i in Gp:
            if isinstance(gp_i, np.ndarray):
                H.append(gp_i + div)
            else:
                H.append(div.copy())

        for i in range(len(sig.val)):
            isig = sig.val[i]
            ind = self.p.symindex(isig.symval[-1, :])

            # electric field
            e = -_matmul_3d(H[ind], isig.sig)

            val = CompStruct(isig.p, isig.enei, e = e)
            val.symval = isig.symval
            field_out.val.append(val)

        return field_out

    def potential(self,
            sig: CompStructMirror,
            inout: int = 1) -> CompStructMirror:
        """Determine potentials and surface derivatives inside/outside of particle.

        MATLAB: @compgreenstatmirror/potential.m

        Parameters
        ----------
        sig : CompStructMirror
            Surface charges
        inout : int
            Potentials inside (1) or outside (2)

        Returns
        -------
        pot : CompStructMirror
            Potentials and surface derivatives
        """
        pot = CompStructMirror(sig.p, sig.enei, sig.fun)

        G = self.G
        H = self.H1 if inout == 1 else self.H2

        for i in range(len(sig.val)):
            isig = sig.val[i]
            ind = self.p.symindex(isig.symval[-1, :])

            phi = _matmul(G[ind], isig.sig)
            phip = _matmul(H[ind], isig.sig)

            if inout == 1:
                val = CompStruct(sig.p, sig.enei, phi1 = phi, phi1p = phip)
            else:
                val = CompStruct(sig.p, sig.enei, phi2 = phi, phi2p = phip)

            val.symval = isig.symval
            pot.val.append(val)

        return pot

    def __repr__(self) -> str:
        return 'CompGreenStatMirror(p={})'.format(self.p)


def _outer_nvec_eye(nvec: np.ndarray, n: int) -> np.ndarray:
    """Compute outer(nvec, eye(n)) for divergent part.

    Returns (n, 3, n) tensor where result[i, :, j] = nvec[i, :] * delta_{ij}
    """
    result = np.zeros((n, 3, n), dtype = nvec.dtype)
    for i in range(n):
        result[i, :, i] = nvec[i, :]
    return result


def _matmul(a: Any, x: Any) -> Any:
    """Generalized matrix multiplication."""
    if isinstance(a, (int, float)) and a == 0:
        return 0
    if isinstance(x, (int, float)) and x == 0:
        return 0
    if isinstance(a, np.ndarray) and isinstance(x, np.ndarray):
        return a @ x
    return a * x


def _matmul_3d(a: Any, x: Any) -> Any:
    """3D tensor * vector multiplication for Gp-type matrices.

    a is (n, 3, n), x is (n,) or (n, npol) -> result is (n, 3) or (n, 3, npol)
    """
    if isinstance(a, (int, float)) and a == 0:
        return 0
    if isinstance(x, (int, float)) and x == 0:
        return 0
    if isinstance(a, np.ndarray) and a.ndim == 3 and isinstance(x, np.ndarray):
        if x.ndim == 1:
            result = np.zeros((a.shape[0], 3), dtype = a.dtype)
            for j in range(3):
                result[:, j] = a[:, j, :] @ x
            return result
        else:
            npol = x.shape[1]
            result = np.zeros((a.shape[0], 3, npol), dtype = a.dtype)
            for j in range(3):
                result[:, j, :] = a[:, j, :] @ x
            return result
    if isinstance(a, np.ndarray) and a.ndim == 2 and isinstance(x, np.ndarray):
        return a @ x
    return a * x
