import os
import sys

from typing import List, Dict, Tuple, Optional, Union, Any, Callable

import numpy as np
from scipy.linalg import lu_factor, lu_solve

from ..greenfun import CompGreenStatLayer, CompStruct
from ..utils.gpu import lu_factor_dispatch, lu_solve_dispatch, to_host, is_cupy_array


class BEMStatLayer(object):

    name = 'bemsolver'
    needs = {'sim': 'stat'}

    def __init__(self,
            p: Any,
            layer: Any,
            enei: Optional[float] = None,
            **options: Any) -> None:

        self.p = p
        self.layer = layer

        self.enei = None
        self.mat_lu = None
        self._A_lu = None
        self._rhs_scale = None

        # Green function with layer
        # MATLAB: obj.g = compgreenstatlayer(p, p, layer, varargin{:})
        self.g = CompGreenStatLayer(p, p, layer, **options)

        # Surface derivative of Green function
        self.F = self.g.F

        if enei is not None:
            self(enei)

    def _init_matrices(self,
            enei: float) -> 'BEMStatLayer':

        if self.enei is not None and np.isclose(self.enei, enei):
            return self

        # MATLAB @bemstatlayer/subsref.m "()" branch:
        #   [H1, H2] = eval(obj.g, enei, 'H1', 'H2')
        #   mat = -inv(eps1 * H1 - eps2 * H2) * (eps1 - eps2)
        # The eps1/eps2 are inside/outside dielectric functions of the
        # particle (per-face). They are scalars for homogeneous setups.
        H1 = self.g.eval(enei, 'H1')
        H2 = self.g.eval(enei, 'H2')

        eps1 = np.atleast_1d(self.p.eps1(enei)).astype(complex)
        eps2 = np.atleast_1d(self.p.eps2(enei)).astype(complex)
        n = H1.shape[0]
        if eps1.size == 1:
            eps1 = np.full(n, eps1[0], dtype = complex)
        if eps2.size == 1:
            eps2 = np.full(n, eps2[0], dtype = complex)

        # Use diagonal multiplication to avoid forming dense diag matrices.
        A = eps1[:, np.newaxis] * H1 - eps2[:, np.newaxis] * H2
        rhs_scale = eps1 - eps2  # per-face

        self._A_lu = lu_factor_dispatch(A)
        self._rhs_scale = rhs_scale
        self.enei = enei

        return self

    def solve(self,
            exc: CompStruct) -> Tuple[CompStruct, 'BEMStatLayer']:

        return self.__truediv__(exc)

    def __truediv__(self,
            exc: CompStruct) -> Tuple[CompStruct, 'BEMStatLayer']:

        self._init_matrices(exc.enei)

        phip = exc.phip
        orig_shape = phip.shape
        if phip.ndim == 1:
            phip_2d = phip.reshape(-1, 1)
        elif phip.ndim > 2:
            phip_2d = phip.reshape(phip.shape[0], -1)
        else:
            phip_2d = phip

        # MATLAB mat * phip = -inv(A) * diag(eps1 - eps2) * phip
        rhs = self._rhs_scale[:, np.newaxis] * phip_2d
        sig_result = -lu_solve_dispatch(self._A_lu, rhs)

        if sig_result.shape != orig_shape:
            sig_result = sig_result.reshape(orig_shape)

        # v1.7 Phase 1.4: host-materialize before returning to user.
        if is_cupy_array(sig_result):
            sig_result = to_host(sig_result)

        sig = CompStruct(self.p, exc.enei, sig = sig_result)

        return sig, self

    def __mul__(self,
            sig: CompStruct) -> CompStruct:

        pot1 = self.potential(sig, 1)
        pot2 = self.potential(sig, 2)

        phi = CompStruct(self.p, sig.enei,
            phi1 = pot1.phi1, phi1p = pot1.phi1p,
            phi2 = pot2.phi2, phi2p = pot2.phi2p)
        return phi

    def field(self,
            sig: CompStruct,
            inout: int = 2) -> CompStruct:

        return self.g.field(sig, inout)

    def potential(self,
            sig: CompStruct,
            inout: int = 2) -> CompStruct:

        return self.g.potential(sig, inout)

    def clear(self) -> 'BEMStatLayer':

        # v1.7 A3 fix: drop the real LU factor / rhs scale held in
        # _A_lu and _rhs_scale.  Previous versions only reset the
        # unused mat_lu attribute, leaving GPU LU memory pinned until
        # the next wavelength rebuild.
        self.mat_lu = None
        self._A_lu = None
        self._rhs_scale = None
        self.enei = None
        return self

    def __call__(self,
            enei: float) -> 'BEMStatLayer':

        return self._init_matrices(enei)

    def __repr__(self) -> str:
        status = 'enei={:.1f}nm'.format(self.enei) if self.enei is not None else 'not initialized'
        n = self.p.n if hasattr(self.p, 'n') else self.p.nfaces if hasattr(self.p, 'nfaces') else '?'
        return 'BEMStatLayer(p: {} faces, {})'.format(n, status)
