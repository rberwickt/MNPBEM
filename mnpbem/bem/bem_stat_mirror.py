import numpy as np
from typing import Optional, List, Tuple, Any, Union

from ..greenfun import CompStruct
from ..greenfun.compgreen_stat_mirror import CompGreenStatMirror
from ..geometry.comparticle_mirror import CompStructMirror
from ..utils.gpu import lu_factor_dispatch, lu_solve_dispatch, to_host, is_cupy_array


def _mirror_stat_eval_host(g: Any, key: str) -> List:
    """Mirror-symmetry-contracted quasistatic Green block list as host arrays.

    Same audit fix as ``bem_ret_mirror._mirror_eval_host`` for the
    quasistatic Green function: the underlying base attribute (G, F, H1...)
    may be a cupy ndarray under MNPBEM_GPU=1 and the upstream mirror
    contraction skips silently for non-numpy inputs.  Bring it to host
    here so BEMStatMirror always receives populated numpy blocks.
    """
    tab = g.p.symtable
    n_sym = tab.shape[0]
    out: List = [0.0] * n_sym

    mat = getattr(g.g, key)
    if isinstance(mat, (int, float)) and mat == 0:
        return out
    if is_cupy_array(mat):
        mat = to_host(mat)
    if not isinstance(mat, np.ndarray):
        return out

    if mat.ndim == 2:
        n = mat.shape[0]
        n_blocks = mat.shape[1] // n
        sub_mats = [mat[:, b * n:(b + 1) * n] for b in range(n_blocks)]
        for i_sym in range(n_sym):
            out[i_sym] = np.zeros_like(sub_mats[0])
            for j_block in range(tab.shape[1]):
                out[i_sym] = out[i_sym] + tab[i_sym, j_block] * sub_mats[j_block]
    elif mat.ndim == 3:
        n = mat.shape[0]
        n_blocks = mat.shape[2] // n
        sub_mats = [mat[:, :, b * n:(b + 1) * n] for b in range(n_blocks)]
        for i_sym in range(n_sym):
            out[i_sym] = np.zeros_like(sub_mats[0])
            for j_block in range(tab.shape[1]):
                out[i_sym] = out[i_sym] + tab[i_sym, j_block] * sub_mats[j_block]
    return out


class BEMStatMirror(object):
    """BEM solver for quasistatic approximation with mirror symmetry.

    Given an external excitation, BEMStatMirror computes the surface
    charges such that the boundary conditions of Maxwell's equations
    in the quasistatic approximation are fulfilled.

    MATLAB: @bemstatmirror

    Parameters
    ----------
    p : ComParticleMirror
        Composite particle with mirror symmetry
    enei : float, optional
        Light wavelength in vacuum for pre-initialization
    """

    name = 'bemsolver'
    needs = {'sim': 'stat', 'sym': True}

    def __init__(self,
            p: Any,
            enei: Optional[float] = None,
            **options: Any) -> None:
        self.p = p
        self.enei = None  # type: Optional[float]

        # Green function
        self.g = CompGreenStatMirror(p, p, **options)

        # surface derivative of Green function (list, one per symmetry value).
        # Use the host-promoting wrapper so MNPBEM_GPU=1 (cupy assembly) does
        # not produce a zero list -- see _mirror_stat_eval_host.
        self.F = _mirror_stat_eval_host(self.g, 'F')

        # resolvent matrices
        self.mat_lu = None  # type: Optional[List]

        if enei is not None:
            self._init_matrices(enei)

    def _init_matrices(self, enei: float) -> 'BEMStatMirror':
        """Initialize matrices for BEM solver.

        MATLAB: @bemstatmirror/subsref.m case '()'
        """
        if self.enei is not None and np.isclose(self.enei, enei):
            return self

        # inside and outside dielectric function
        eps1 = self.p.eps1(enei)
        eps2 = self.p.eps2(enei)

        # Lambda [Garcia de Abajo, Eq. (23)]
        lambda_diag = 2 * np.pi * (eps1 + eps2) / (eps1 - eps2)

        self.mat_lu = []
        for i in range(len(self.F)):
            # BEM resolvent matrix
            self.mat_lu.append(lu_factor_dispatch(-(np.diag(lambda_diag) + self.F[i])))

        self.enei = enei
        return self

    def solve(self, exc: CompStructMirror) -> Tuple[CompStructMirror, 'BEMStatMirror']:
        """Surface charge for given excitation.

        MATLAB: @bemstatmirror/mldivide.m

        Parameters
        ----------
        exc : CompStructMirror
            External excitation with field 'phip'

        Returns
        -------
        sig : CompStructMirror
            Surface charge
        obj : BEMStatMirror
            Updated solver
        """
        self._init_matrices(exc.enei)

        sig = CompStructMirror(self.p, exc.enei, getattr(exc, 'fun', None))

        for i in range(len(exc.val)):
            ind = self.p.symindex(exc.val[i].symval[-1, :])

            sig_val = _lu_solve_multi(self.mat_lu[ind], exc.val[i].phip)

            # v1.7 Phase 1.4: host-materialize for user-facing access.
            if is_cupy_array(sig_val):
                sig_val = to_host(sig_val)

            val = CompStruct(self.p, exc.enei, sig = sig_val)
            val.symval = exc.val[i].symval
            sig.val.append(val)

        return sig, self

    def __truediv__(self, exc: CompStructMirror) -> Tuple[CompStructMirror, 'BEMStatMirror']:
        return self.solve(exc)

    def __mul__(self, sig: CompStructMirror) -> CompStructMirror:
        """Induced potential for given surface charge.

        MATLAB: @bemstatmirror/mtimes.m
        """
        pot1 = self.potential(sig, 1)
        pot2 = self.potential(sig, 2)

        result = CompStructMirror(self.p, sig.enei, sig.fun)
        for i in range(len(sig.val)):
            combined = CompStruct(self.p, sig.enei)
            for attr in ('phi1', 'phi1p'):
                v = getattr(pot1.val[i], attr, None)
                if v is not None:
                    setattr(combined, attr, v)
            for attr in ('phi2', 'phi2p'):
                v = getattr(pot2.val[i], attr, None)
                if v is not None:
                    setattr(combined, attr, v)
            combined.symval = sig.val[i].symval
            result.val.append(combined)

        return result

    def potential(self,
            sig: CompStructMirror,
            inout: int = 2) -> CompStructMirror:
        """Potentials and surface derivatives inside/outside of particle.

        MATLAB: @bemstatmirror/potential.m
        """
        return self.g.potential(sig, inout)

    def field(self,
            sig: CompStructMirror,
            inout: int = 2) -> CompStructMirror:
        """Electric field inside/outside of particle surface.

        MATLAB: @bemstatmirror/field.m
        """
        return self.g.field(sig, inout)

    def __call__(self, enei: float) -> 'BEMStatMirror':
        return self._init_matrices(enei)

    def __repr__(self) -> str:
        status = 'enei={}'.format(self.enei) if self.enei is not None else 'not initialized'
        return 'BEMStatMirror(p={}, {})'.format(self.p, status)


def _lu_solve_multi(lu_piv: Tuple, b: Any) -> Any:
    if isinstance(b, np.ndarray):
        if b.ndim == 1:
            return lu_solve_dispatch(lu_piv, b)
        else:
            return lu_solve_dispatch(lu_piv, b.reshape(b.shape[0], -1)).reshape(b.shape)
    return lu_solve_dispatch(lu_piv, np.asarray(b))


def _matmul(a: Any, x: Any) -> Any:
    if isinstance(a, (int, float)):
        if a == 0:
            return 0
        return a * x
    if isinstance(x, (int, float)):
        if x == 0:
            return 0
        return a * x
    if np.isscalar(a):
        return a * x
    if isinstance(a, np.ndarray) and isinstance(x, np.ndarray):
        if x.ndim == 1:
            return a @ x
        elif x.ndim == 2:
            return a @ x
    return a @ x
