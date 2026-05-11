"""
BEM solver for quasistatic approximation.

MATLAB: BEM/@bemstat/
100% identical to MATLAB MNPBEM implementation.

Given an external excitation, BEMStat computes the surface charges
such that the boundary conditions of Maxwell's equations in the
quasistatic approximation are fulfilled.

Reference:
    Garcia de Abajo and Howie, PRB 65, 115418 (2002)
    Hohenester et al., PRL 103, 106801 (2009)
"""

import os

import numpy as np
from scipy.linalg import lu_factor, lu_solve
from ..greenfun import CompGreenStat, CompStruct
from ..utils.matlab_compat import msqrt
from ..utils.gpu import lu_factor_dispatch, lu_solve_dispatch, to_host, is_cupy_array


def _vram_share_lu_kwargs() -> dict:
    if os.environ.get('MNPBEM_VRAM_SHARE', '0') != '1':
        return {}
    n_gpus = int(os.environ.get('MNPBEM_VRAM_SHARE_GPUS', '1'))
    if n_gpus <= 1:
        return {}
    backend = os.environ.get('MNPBEM_VRAM_SHARE_BACKEND', 'cusolvermg')
    return {'n_gpus': n_gpus, 'backend': backend}


class BEMStat(object):
    """
    BEM solver for quasistatic approximation.

    MATLAB: @bemstat

    Properties
    ----------
    name : str
        'bemsolver' (constant)
    needs : dict
        {'sim': 'stat'} (constant)
    p : ComParticle
        Composite particle (see comparticle)
    F : ndarray
        Surface derivative of Green function
    enei : float or None
        Light wavelength in vacuum
    g : CompGreenStat (private)
        Green function (needed in bemstat/field)
    mat : ndarray (private)
        -inv(Lambda + F)

    Methods
    -------
    __init__(p, enei=None, **options)
        Initialize quasistatic BEM solver
    solve(exc)
        Solve BEM equations for given excitation
    __truediv__(exc)
        Surface charge for given excitation (operator \)
    __mul__(sig)
        Induced potential for given surface charge (operator *)
    field(sig, inout=2)
        Electric field inside/outside of particle surface
    potential(sig, inout=2)
        Potentials and surface derivatives inside/outside of particle
    clear()
        Clear auxiliary matrices
    __call__(enei)
        Computes resolvent matrix for later use in __truediv__

    Examples
    --------
    >>> from mnpbem import EpsConst, EpsTable, trisphere, ComParticle
    >>> from mnpbem.bem import BEMStat
    >>>
    >>> # Create gold sphere
    >>> eps_tab = [EpsConst(1.0), EpsTable('gold.dat')]
    >>> sphere = trisphere(144, 10.0)
    >>> p = ComParticle(eps_tab, [sphere], [[2, 1]])
    >>>
    >>> # Create BEM solver
    >>> bem = BEMStat(p)
    >>>
    >>> # Solve for excitation
    >>> sig = bem \ exc  # or sig = bem.solve(exc)
    >>>
    >>> # Get induced potential
    >>> phi = bem * sig
    """

    # Class constants
    name = 'bemsolver'
    needs = {'sim': 'stat'}

    def __init__(self, p, enei=None, **options):
        """
        Initialize quasistatic BEM solver.

        MATLAB: bemstat.m, private/init.m

        Parameters
        ----------
        p : ComParticle
            Compound of particles (see comparticle)
        enei : float, optional
            Light wavelength in vacuum
        **options : dict
            Additional options passed to CompGreenStat. Special keys:

            schur : bool or 'auto', optional
                Activate Schur-complement elimination of EpsNonlocal
                cover-layer faces. ``True`` or ``'auto'`` enables the
                reduction whenever a cover layer is detected via
                ``detect_shell_core_partition``; ``False`` (default) keeps
                the full BEM matrix.

        Examples
        --------
        >>> bem = BEMStat(p)
        >>> bem = BEMStat(p, enei=600.0)
        """
        # Validate particle
        if p is None:
            raise ValueError(
                "BEMStat: 'p' must be a ComParticle (or compatible particle "
                "object), got None.")
        if not (hasattr(p, 'pos') and hasattr(p, 'nvec') and hasattr(p, 'eps')):
            raise TypeError(
                "BEMStat: 'p' must expose ComParticle-like attributes "
                "(pos, nvec, eps); got {!r}.".format(type(p).__name__))

        # Save particle
        self.p = p

        # Schur option (extract before forwarding to CompGreenStat).
        self._schur_opt = options.pop('schur', False)
        self._schur_active = False
        self._shell_idx = None
        self._core_idx = None
        self._schur_reduce_rhs = None
        self._schur_recover = None

        # Initialize properties
        self.enei = None
        self.mat_lu = None

        # Green function
        # MATLAB: obj.g = compgreenstat(p, p, varargin{:})
        self.g = CompGreenStat(p, p, **options)

        # Surface derivative of Green function
        # MATLAB: obj.F = subsref(obj.g, substruct('.', 'F'))
        F_obj = self.g.F
        # If hmatrix=True swapped self.g for an ACACompGreenStat, F is an
        # HMatrix; convert to dense so the standard LU solver works.
        if hasattr(F_obj, 'full') and not isinstance(F_obj, np.ndarray):
            F_obj = F_obj.full()
        self.F = F_obj

        # Initialize for given wavelength
        # MATLAB: if exist('enei', 'var') && ~isempty(enei)
        if enei is not None:
            self(enei)

    def _init_matrices(self, enei):
        """
        Initialize matrices for BEM solver.

        MATLAB: bemstat/subsref.m case '()'

        Parameters
        ----------
        enei : float
            Light wavelength in vacuum
        """
        # Use previously computed matrices?
        # MATLAB: if isempty(obj.enei) || obj.enei ~= enei
        if self.enei is None or self.enei != enei:
            # Inside and outside dielectric function
            # MATLAB: eps1 = obj.p.eps1(enei); eps2 = obj.p.eps2(enei);
            eps1 = self.p.eps1(enei)
            eps2 = self.p.eps2(enei)

            # Lambda [Garcia de Abajo, Eq. (23)]
            # MATLAB: lambda = 2 * pi * (eps1 + eps2) ./ (eps1 - eps2)
            lambda_diag = 2 * np.pi * (eps1 + eps2) / (eps1 - eps2)

            # BEM resolvent matrix
            # MATLAB: obj.mat = -inv(diag(lambda) + obj.F)
            Lambda = np.diag(lambda_diag)
            M_full = -(Lambda + self.F)

            # Optional Schur-complement reduction over EpsNonlocal cover-
            # layer faces. The reduced matrix has size (M, M) where M is the
            # number of core (non-shell) faces. Mathematically equivalent to
            # the full block solve. VRAM-share kwargs propagate into mat_lu.
            _lu_opts = _vram_share_lu_kwargs()
            self._schur_active = False
            if self._schur_opt:
                from .schur_helpers import (
                    schur_eliminate, detect_shell_core_partition,
                )
                partition = detect_shell_core_partition(self.p)
                if partition is not None:
                    shell_idx, core_idx = partition
                    M_eff, reduce_rhs, recover = schur_eliminate(
                            np.asarray(M_full), shell_idx, core_idx)
                    self._shell_idx = shell_idx
                    self._core_idx = core_idx
                    self._schur_reduce_rhs = reduce_rhs
                    self._schur_recover = recover
                    self._schur_active = True
                    self.mat_lu = lu_factor_dispatch(M_eff, **_lu_opts)
                else:
                    self.mat_lu = lu_factor_dispatch(M_full, **_lu_opts)
            else:
                self.mat_lu = lu_factor_dispatch(M_full, **_lu_opts)

            # Save energy
            # MATLAB: obj.enei = enei
            self.enei = enei

        return self

    def solve(self, exc):
        """
        Solve BEM equations for given excitation.

        MATLAB: bemstat/solve.m

        Parameters
        ----------
        exc : CompStruct
            compstruct with fields for external excitation

        Returns
        -------
        sig : CompStruct
            compstruct with fields for surface charge
        obj : BEMStat
            Updated BEM solver object

        Examples
        --------
        >>> sig, bem = bem.solve(exc)
        """
        # MATLAB: [sig, obj] = mldivide(obj, exc)
        return self.__truediv__(exc)

    def __truediv__(self, exc):
        """
        Surface charge for given excitation.

        MATLAB: bemstat/mldivide.m

        Usage
        -----
        sig = obj \ exc

        Parameters
        ----------
        exc : CompStruct
            compstruct with field 'phip' for external excitation

        Returns
        -------
        sig : CompStruct
            compstruct with field for surface charge
        obj : BEMStat
            Updated BEM solver object

        Examples
        --------
        >>> sig, bem = bem \ exc
        """
        # Initialize BEM solver (if needed)
        # MATLAB: obj = subsref(obj, substruct('()', {exc.enei}))
        self._init_matrices(exc.enei)

        # Solve: σ = mat · φₚ
        # MATLAB: sig = compstruct(obj.p, exc.enei, 'sig', matmul(obj.mat, exc.phip))
        if self._schur_active:
            sig_result = self._schur_solve(exc.phip)
        else:
            sig_result = self._lu_solve(self.mat_lu, exc.phip)
        # v1.7 Phase 1.4 fix: host-materialize so np.asarray(sig.sig) works.
        if is_cupy_array(sig_result):
            sig_result = to_host(sig_result)
        sig = CompStruct(self.p, exc.enei, sig=sig_result)

        return sig, self

    def _schur_solve(self, phip):
        # Reduced RHS lives only on core faces. Solve (M, M) reduced system
        # then recover the full sigma vector via the cached
        # _schur_recover callable.
        b_full = np.asarray(phip)
        b_eff = self._schur_reduce_rhs(b_full)
        sig_core = self._lu_solve(self.mat_lu, b_eff)
        sig_full = self._schur_recover(sig_core, b_full)
        return sig_full

    def __mul__(self, sig):
        """
        Induced potential for given surface charge.

        MATLAB: bemstat/mtimes.m

        Usage
        -----
        phi = obj * sig

        Parameters
        ----------
        sig : CompStruct
            compstruct with fields for surface charge

        Returns
        -------
        phi : CompStruct
            compstruct with fields for induced potential

        Examples
        --------
        >>> phi = bem * sig
        """
        # MATLAB: phi = potential(obj, sig, 1) + potential(obj, sig, 2)
        pot1 = self.potential(sig, 1)
        pot2 = self.potential(sig, 2)

        # Combine potentials
        # pot1 has phi1, phi1p; pot2 has phi2, phi2p
        # Return combined result
        phi = CompStruct(self.p, sig.enei,
                        phi1=pot1.phi1, phi1p=pot1.phi1p,
                        phi2=pot2.phi2, phi2p=pot2.phi2p)
        return phi

    def field(self, sig, inout=2):
        """
        Electric field inside/outside of particle surface.

        MATLAB: bemstat/field.m

        Parameters
        ----------
        sig : CompStruct
            COMPSTRUCT object with surface charges
        inout : int, optional
            Electric field inside (inout=1) or outside (inout=2, default) of particle

        Returns
        -------
        field : CompStruct
            COMPSTRUCT object with electric field

        Examples
        --------
        >>> field = bem.field(sig, inout=2)
        """
        # Compute field from derivative of Green function or from potential interpolation
        # MATLAB: switch obj.g.deriv
        if self.g.deriv == 'cart':
            # MATLAB: field = obj.g.field(sig, inout)
            return self.g.field(sig, inout)

        elif self.g.deriv == 'norm':
            # Electric field in normal direction
            # MATLAB: switch inout
            #           case 1: e = -outer(obj.p.nvec, matmul(obj.g.H1, sig.sig))
            #           case 2: e = -outer(obj.p.nvec, matmul(obj.g.H2, sig.sig))
            if inout == 1:
                H = self.g.H1
            else:
                H = self.g.H2

            # e = -outer(nvec, H @ sig.sig)
            # MATLAB outer(nvec, scalar) creates (n, 3) matrix
            H_sig = self._matmul(H, sig.sig)
            e = -self._outer(self.p.nvec, H_sig)

            # Tangential directions computed by interpolation and derivative
            # MATLAB: phi = interp(obj.p, matmul(obj.g.G, sig.sig))
            #         [phi1, phi2, t1, t2] = deriv(obj.p, phi)
            G_sig = self._matmul(self.g.G, sig.sig)
            phi = self.p.interp(G_sig)
            phi1, phi2, t1, t2 = self.p.deriv(phi)

            # Normal vector
            # MATLAB: nvec = cross(t1, t2)
            #         h = sqrt(dot(nvec, nvec, 2)); nvec = bsxfun(@rdivide, nvec, h)
            nvec = np.cross(t1, t2)
            h = msqrt(np.sum(nvec * nvec, axis=1, keepdims=True))
            nvec = nvec / h

            # Tangential derivative of PHI
            # MATLAB: phip = outer(bsxfun(@rdivide, cross(t2, nvec, 2), h), phi1) -
            #                outer(bsxfun(@rdivide, cross(t1, nvec, 2), h), phi2)
            tvec1 = np.cross(t2, nvec) / h
            tvec2 = np.cross(t1, nvec) / h
            phip = self._outer(tvec1, phi1) - self._outer(tvec2, phi2)

            # Add electric field in tangential direction
            # MATLAB: e = e - phip
            e = e - phip

            # Set output
            # MATLAB: field = compstruct(obj.p, sig.enei, 'e', e)
            field = CompStruct(self.p, sig.enei, e=e)
            return field

    def potential(self, sig, inout=2):
        """
        Determine potentials and surface derivatives inside/outside of particle.

        MATLAB: bemstat/potential.m

        Parameters
        ----------
        sig : CompStruct
            compstruct with surface charges
        inout : int, optional
            Potential inside (inout=1) or outside (inout=2, default) of particle

        Returns
        -------
        pot : CompStruct
            compstruct object with potentials

        Examples
        --------
        >>> pot = bem.potential(sig, inout=2)
        """
        # MATLAB: pot = obj.g.potential(sig, inout)
        return self.g.potential(sig, inout)

    def clear(self):
        """
        Clear auxiliary matrices.

        MATLAB: bemstat/clear.m

        Returns
        -------
        self : BEMStat
            Returns self for chaining

        Examples
        --------
        >>> bem = bem.clear()
        """
        # MATLAB: obj.mat = []
        # v1.7 A3 fix: also reset enei so the cache gate in _init_matrices
        # does not skip rebuild when the user re-solves at the same
        # wavelength after clear().  Stale enei + mat_lu=None previously
        # crashed __truediv__ with a NoneType unpack error.  Schur
        # auxiliaries are likewise dropped so a subsequent solve does not
        # accidentally reuse the recover callable bound to a freed factor.
        self.mat_lu = None
        self.enei = None
        self._schur_active = False
        self._schur_reduce_rhs = None
        self._schur_recover = None
        return self

    def __call__(self, enei):
        """
        Computes resolvent matrix for later use in mldivide.

        MATLAB: bemstat/subsref.m case '()'

        Parameters
        ----------
        enei : float
            Light wavelength in vacuum

        Returns
        -------
        self : BEMStat
            Returns self for chaining

        Examples
        --------
        >>> bem = bem(600.0)
        """
        return self._init_matrices(enei)

    @staticmethod
    def _lu_solve(lu_piv, b):
        if isinstance(lu_piv, tuple) and len(lu_piv) == 3 and lu_piv[0] in ("cpu", "gpu", "mgpu"):
            if b.ndim == 1:
                return lu_solve_dispatch(lu_piv, b)
            return lu_solve_dispatch(lu_piv, b.reshape(b.shape[0], -1)).reshape(b.shape)
        if b.ndim == 1:
            return lu_solve(lu_piv, b, check_finite=False)
        else:
            return lu_solve(lu_piv, b.reshape(b.shape[0], -1), check_finite=False).reshape(b.shape)

    def _matmul(self, a, x):
        """
        Generalized matrix multiplication for tensors.

        MATLAB: Misc/matmul.m
        """
        if np.isscalar(a) or (isinstance(a, np.ndarray) and a.size == 1):
            if a == 0:
                return 0
            else:
                return a * x
        elif np.isscalar(x) or (isinstance(x, np.ndarray) and x.size == 1):
            if x == 0:
                return 0
            else:
                return a * x
        else:
            # A is matrix/tensor
            siza = a.shape
            sizx = x.shape if hasattr(x, 'shape') else (len(x),)

            # Check if we need special handling for 3D arrays
            if len(siza) == 3:
                # a is (n1, 3, n2), x is (n2,) or (n2, ...)
                n1, _, n2 = siza

                if len(sizx) == 1:
                    # x is 1D
                    y = np.tensordot(a, x, axes=([2], [0]))
                else:
                    # x is multi-dimensional
                    a_flat = a.reshape(n1 * 3, n2)
                    x_flat = x.reshape(n2, -1)
                    y_flat = a_flat @ x_flat

                    new_shape = (n1, 3) + sizx[1:]
                    y = y_flat.reshape(new_shape)

                return y
            else:
                # Standard 2D matrix multiplication
                if len(sizx) == 1:
                    return a @ x
                else:
                    return a @ x.reshape(sizx[0], -1).reshape((sizx[0],) + sizx[1:])

    def _outer(self, nvec, scalar):
        """
        Outer product: nvec * scalar.

        MATLAB: outer(nvec, scalar)

        Parameters
        ----------
        nvec : ndarray, shape (n, 3)
            Normal vectors
        scalar : ndarray, shape (n,)
            Scalar values

        Returns
        -------
        result : ndarray, shape (n, 3)
            nvec * scalar[:, None]
        """
        if scalar.ndim == 1:
            return nvec * scalar[:, np.newaxis]
        else:
            # Handle higher dimensions
            return nvec[:, :, np.newaxis] * scalar[:, np.newaxis, :]

    def __repr__(self):
        """String representation."""
        status = "λ={:.1f}nm".format(self.enei) if self.enei is not None else "not initialized"
        return "BEMStat(p: {} faces, {})".format(
            self.p.n if hasattr(self.p, 'n') else '?', status)

    def __str__(self):
        """Detailed string representation."""
        return (
            "bemstat:\n"
            "  p: {}\n"
            "  F: {}\n"
            "  enei: {}\n"
            "  mat: {}".format(
                self.p,
                self.F.shape if hasattr(self, 'F') else 'not computed',
                self.enei,
                self.mat_lu[0].shape if self.mat_lu is not None else 'not computed')
        )
