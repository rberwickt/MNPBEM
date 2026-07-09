import numpy as np
from typing import Optional, List, Tuple, Any, Union

from .compgreen_ret import CompGreenRet
from .compgreen_stat import CompStruct
from ..geometry.comparticle_mirror import CompStructMirror


class CompGreenRetMirror(object):
    """Green function for composite particles with mirror symmetry (retarded).

    MATLAB: @compgreenretmirror

    Parameters
    ----------
    p : ComParticleMirror
        Particle with mirror symmetry
    """

    name = 'greenfunction'
    needs = {'sim': 'ret', 'sym': True}

    def __init__(self,
            p: Any,
            _dummy: Any = None,
            **options: Any) -> None:
        self.p = p
        # Green function between half particle and full particle
        self.g = CompGreenRet(p, p.full(), **options)

    @property
    def con(self) -> Any:
        return self.g.con

    @property
    def deriv(self) -> str:
        return self.g.deriv

    def eval(self,
            i: int,
            j: int,
            key: str,
            enei: float,
            **kwargs: Any) -> List:
        """Evaluate retarded Green function with mirror symmetry.

        MATLAB: @compgreenretmirror/eval.m

        Parameters
        ----------
        i : int
            Row index (inout)
        j : int
            Column index (inout)
        key : str
            G, F, H1, H2, Gp, H1p, H2p
        enei : float
            Light wavelength in vacuum

        Returns
        -------
        g : list
            List of contracted Green function matrices for each symmetry value
        """
        # evaluate full Green function
        mat = self.g.eval(i, j, key, enei, **kwargs)
        tab = self.p.symtable

        n_sym = tab.shape[0]
        g = [0.0] * n_sym

        if isinstance(mat, (int, float)) and mat == 0:
            return g

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

    def field(self,
            sig: CompStructMirror,
            inout: int = 1) -> CompStructMirror:
        """Electric and magnetic field inside/outside of particle surface.

        MATLAB: @compgreenretmirror/field.m

        Parameters
        ----------
        sig : CompStructMirror
            Surface charges and currents
        inout : int
            Fields inside (1) or outside (2)

        Returns
        -------
        field : CompStructMirror
            Electric and magnetic fields
        """
        assert self.g.deriv == 'cart'

        enei = sig.enei
        k = 2 * np.pi / enei

        field_out = CompStructMirror(sig.p, sig.enei, sig.fun)

        # Green functions
        G1 = self.eval(inout - 1, 0, 'G', enei)
        G2 = self.eval(inout - 1, 1, 'G', enei)

        if inout == 1:
            H1p = self.eval(inout - 1, 0, 'H1p', enei)
            H2p = self.eval(inout - 1, 1, 'H1p', enei)
        else:
            H1p = self.eval(inout - 1, 0, 'H2p', enei)
            H2p = self.eval(inout - 1, 1, 'H2p', enei)

        for i in range(len(sig.val)):
            isig = sig.val[i]
            symval = isig.symval

            x_idx = self.p.symindex(symval[0, :])
            y_idx = self.p.symindex(symval[1, :])
            z_idx = self.p.symindex(symval[2, :])
            ind = [x_idx, y_idx, z_idx]

            # electric field E = i*k*A - grad(V)
            e = (1j * k * _indmul(G1, isig.h1, ind)
                 - _matmul_3d(H1p[z_idx], isig.sig1)
                 + 1j * k * _indmul(G2, isig.h2, ind)
                 - _matmul_3d(H2p[z_idx], isig.sig2))

            # magnetic field
            h = _indcross(H1p, isig.h1, ind) + _indcross(H2p, isig.h2, ind)

            val = CompStruct(sig.p, sig.enei, e = e, h = h)
            val.symval = isig.symval
            field_out.val.append(val)

        return field_out

    def potential(self,
            sig: CompStructMirror,
            inout: int = 1) -> CompStructMirror:
        """Determine potentials and surface derivatives inside/outside of particle.

        MATLAB: @compgreenretmirror/potential.m

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
        enei = sig.enei
        pot = CompStructMirror(sig.p, sig.enei, sig.fun)

        H_key = 'H1' if inout == 1 else 'H2'

        G1 = self.eval(inout - 1, 0, 'G', enei)
        G2 = self.eval(inout - 1, 1, 'G', enei)
        H1 = self.eval(inout - 1, 0, H_key, enei)
        H2 = self.eval(inout - 1, 1, H_key, enei)

        for i in range(len(sig.val)):
            isig = sig.val[i]
            symval = isig.symval

            x_idx = self.p.symindex(symval[0, :])
            y_idx = self.p.symindex(symval[1, :])
            z_idx = self.p.symindex(symval[2, :])
            ind = [x_idx, y_idx, z_idx]

            # scalar potential
            phi = _matmul(G1[z_idx], isig.sig1) + _matmul(G2[z_idx], isig.sig2)
            phip = _matmul(H1[z_idx], isig.sig1) + _matmul(H2[z_idx], isig.sig2)
            # vector potential
            a = _indmul(G1, isig.h1, ind) + _indmul(G2, isig.h2, ind)
            ap = _indmul(H1, isig.h1, ind) + _indmul(H2, isig.h2, ind)

            if inout == 1:
                val = CompStruct(sig.p, enei,
                                 phi1 = phi, phi1p = phip, a1 = a, a1p = ap)
            else:
                val = CompStruct(sig.p, enei,
                                 phi2 = phi, phi2p = phip, a2 = a, a2p = ap)
            val.symval = isig.symval
            pot.val.append(val)

        return pot

    def __repr__(self) -> str:
        return 'CompGreenRetMirror(p={})'.format(self.p)


def _matmul(a: Any, x: Any) -> Any:
    """Generalized matrix multiplication."""
    if isinstance(a, (int, float)) and a == 0:
        return 0
    if isinstance(x, (int, float)) and x == 0:
        return 0
    if isinstance(a, np.ndarray) and isinstance(x, np.ndarray):
        if a.ndim == 2 and x.ndim == 1:
            return a @ x
        elif a.ndim == 2 and x.ndim == 2:
            return a @ x
        else:
            return a @ x
    return a * x


def _matmul_3d(a: Any, x: Any) -> Any:
    """Matrix multiply for 3D tensor (Gp, H1p, H2p) with scalar."""
    if isinstance(a, (int, float)) and a == 0:
        return 0
    if isinstance(x, (int, float)) and x == 0:
        return 0
    if isinstance(a, np.ndarray) and a.ndim == 3 and isinstance(x, np.ndarray):
        # a is (n, 3, n), x is (n,) or (n, npol)
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
    # fallback: treat as outer product nvec * (H @ sig)
    if isinstance(a, np.ndarray) and a.ndim == 2 and isinstance(x, np.ndarray):
        if x.ndim == 1:
            # a is nvec-like or just a matrix
            return a @ x
        return a @ x
    return a * x


def _indmul(mat_list: List,
        v: np.ndarray,
        ind: List[int]) -> Any:
    """Indexed matrix multiplication.

    MATLAB: indmul in compgreenretmirror/field.m and potential.m
    """
    if isinstance(mat_list[0], (int, float)) and mat_list[0] == 0:
        return 0

    if v.ndim == 2:
        # v is (n, 3) or (n, 3)
        n = v.shape[0]
        result = np.zeros_like(v)
        for j in range(3):
            vj = v[:, j]
            result[:, j] = _matmul(mat_list[ind[j]], vj)
        return result
    elif v.ndim == 3:
        # v is (n, 3, npol)
        n = v.shape[0]
        npol = v.shape[2]
        result = np.zeros_like(v)
        for j in range(3):
            vj = v[:, j, :]  # (n, npol)
            result[:, j, :] = _matmul(mat_list[ind[j]], vj)
        return result
    return 0


def _indcross(mat_list: List,
        v: np.ndarray,
        ind: List[int]) -> Any:
    """Indexed cross product.

    MATLAB: indcross in compgreenretmirror/field.m
    """
    if isinstance(mat_list[0], (int, float)) and mat_list[0] == 0:
        return 0

    def imat(k: int, i: int) -> np.ndarray:
        m = mat_list[ind[k]]
        if m.ndim == 3:
            return m[:, i, :]  # (n, n)
        return m

    def ivec(i: int) -> np.ndarray:
        if v.ndim == 2:
            return v[:, i]
        elif v.ndim == 3:
            return v[:, i, :]
        return v

    if v.ndim == 2:
        n = v.shape[0]
        result = np.zeros((n, 3), dtype = v.dtype)
    elif v.ndim == 3:
        n = v.shape[0]
        npol = v.shape[2]
        result = np.zeros((n, 3, npol), dtype = v.dtype)
    else:
        return 0

    # cross product components
    result_0 = _matmul(imat(2, 1), ivec(2)) - _matmul(imat(1, 2), ivec(1))
    result_1 = _matmul(imat(0, 2), ivec(0)) - _matmul(imat(2, 0), ivec(2))
    result_2 = _matmul(imat(1, 0), ivec(1)) - _matmul(imat(0, 1), ivec(0))

    if v.ndim == 2:
        result[:, 0] = result_0
        result[:, 1] = result_1
        result[:, 2] = result_2
    elif v.ndim == 3:
        result[:, 0, :] = result_0
        result[:, 1, :] = result_1
        result[:, 2, :] = result_2

    return result
