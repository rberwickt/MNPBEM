"""
Plane wave excitation within quasistatic approximation.

This module provides plane wave excitation for quasistatic BEM simulations.

Reference:
    MATLAB MNPBEM Simulation/static/@planewavestat

Matches MATLAB MNPBEM implementation exactly.
"""

import numpy as np
from ..greenfun import CompStruct


class PlaneWaveStat(object):
    """
    Plane wave excitation within quasistatic approximation.

    Provides electric field and potential excitation for quasistatic BEM solvers.
    Computes absorption, scattering, and extinction cross sections.

    Parameters
    ----------
    pol : ndarray
        Light polarization vector(s), shape (3,) or (npol, 3)
    medium : int, optional
        Medium index for excitation (default: 1)

    Attributes
    ----------
    pol : ndarray
        Light polarization vector(s)
    medium : int
        Medium index through which particle is excited

    Notes
    -----
    MATLAB: Simulation/static/@planewavestat/planewavestat.m

    The quasistatic approximation assumes wavelength >> particle size,
    so only the electric field (no magnetic field) is considered.

    Cross sections are computed using dipole approximation:
    - Absorption: σ_abs = 4πk·Im(pol·dip)
    - Scattering: σ_sca = (8π/3)k⁴·|dip|²
    - Extinction: σ_ext = σ_abs + σ_sca

    Examples
    --------
    >>> import numpy as np
    >>> from mnpbem.simulation import PlaneWaveStat
    >>>
    >>> # X-polarized plane wave
    >>> pol = np.array([1.0, 0.0, 0.0])
    >>> exc = PlaneWaveStat(pol)
    >>>
    >>> # Get excitation potential at wavelength 600nm
    >>> from mnpbem import trisphere, ComParticle, EpsConst
    >>> p = ComParticle([EpsConst(1.0), EpsConst(2.0)],
    ...                  [trisphere(144, 10.0)], [[2, 1]])
    >>> pot = exc(p, 600.0)
    """

    # Class constants
    # MATLAB: @planewavestat line 5-8
    name = 'planewave'
    needs = {'sim': 'stat'}

    def __init__(self, pol, medium=1, **options):
        """
        Initialize plane wave excitation.

        MATLAB: planewavestat.m + init.m

        Parameters
        ----------
        pol : array_like
            Light polarization vector(s), shape (3,) or (npol, 3)
        medium : int, optional
            Medium index for computing spectra (default: 1)
        **options : dict
            Additional options (for compatibility)
        """
        # MATLAB: init.m line 16
        if pol is None:
            raise ValueError("PlaneWaveStat: 'pol' must be a polarization vector, got None.")
        self.pol = np.asarray(pol)
        if self.pol.dtype == object or self.pol.size == 0:
            raise ValueError(
                "PlaneWaveStat: 'pol' must be a numeric array of shape (3,) or (npol, 3).")
        if self.pol.ndim == 1:
            if self.pol.size != 3:
                raise ValueError(
                    "PlaneWaveStat: 1D 'pol' must have length 3, got length {}."
                    .format(self.pol.size))
            self.pol = self.pol.reshape(1, -1)
        elif self.pol.ndim == 2:
            if self.pol.shape[1] != 3:
                raise ValueError(
                    "PlaneWaveStat: 2D 'pol' must have shape (npol, 3), got {}."
                    .format(self.pol.shape))
        else:
            raise ValueError(
                "PlaneWaveStat: 'pol' must be 1D or 2D, got {}D.".format(self.pol.ndim))

        # MATLAB: init.m line 32
        self.medium = options.get('medium', medium)

    def field(self, p, enei):
        """
        Electric field for plane wave excitation.

        MATLAB: field.m

        Parameters
        ----------
        p : ComParticle
            Particle or points where field is computed
        enei : float
            Light wavelength in vacuum (nm)

        Returns
        -------
        exc : CompStruct
            CompStruct object containing electric field 'e'

        Notes
        -----
        MATLAB: field.m line 12-13

        The field is constant everywhere (quasistatic approximation):
        E = pol (same at all points)
        """
        # MATLAB: field.m line 12-13
        # exc = compstruct(p, enei, 'e', repmat(reshape(obj.pol, [1, size(obj.pol')]), [p.n, 1, 1]))

        n = p.n  # number of points
        npol = self.pol.shape[0]  # number of polarizations

        # Create field array: (n, 3, npol)
        e = np.zeros((n, 3, npol), dtype=complex)
        for i in range(npol):
            e[:, :, i] = self.pol[i, :]  # Broadcast polarization to all points

        # Squeeze if only one polarization
        if npol == 1:
            e = e[:, :, 0]

        return CompStruct(p, enei, e=e)

    def potential(self, p, enei):
        """
        Potential of plane wave excitation for use in BEMStat.

        MATLAB: potential.m / planewavestatmirror/potential.m

        Parameters
        ----------
        p : ComParticle or ComParticleMirror
            Particle surface where potential is computed
        enei : float
            Light wavelength in vacuum (nm)

        Returns
        -------
        exc : CompStruct or CompStructMirror
            CompStruct with surface derivative 'phip' of scalar potential.
            Returns CompStructMirror when p is a ComParticleMirror.
        """
        from ..geometry.comparticle_mirror import ComParticleMirror, CompStructMirror

        if isinstance(p, ComParticleMirror):
            return self._potential_mirror(p, enei)

        # MATLAB: potential.m line 13
        # exc = compstruct(p, enei, 'phip', -p.nvec * transpose(obj.pol))

        # p.nvec is (nfaces, 3), obj.pol is (npol, 3)
        # Result should be (nfaces, npol) for multiple polarizations
        phip = -p.nvec @ self.pol.T  # (nfaces, 3) @ (3, npol) = (nfaces, npol)

        # Squeeze if only one polarization
        if self.pol.shape[0] == 1:
            phip = phip[:, 0]

        return CompStruct(p, enei, phip=phip)

    def _potential_mirror(self, p, enei):
        """
        Potential for mirror-symmetric particle.

        MATLAB: planewavestatmirror/potential.m

        Decomposes excitation into symmetry components.
        """
        from ..geometry.comparticle_mirror import CompStructMirror

        exc = CompStructMirror(p, enei)

        def make_component(sym_keys, pol):
            """Create a CompStruct with symmetry values and phip."""
            symval = p.symvalue(sym_keys)
            phip = -p.nvec @ np.array(pol, dtype=float).reshape(1, -1).T
            phip = phip[:, 0]
            val = CompStruct(p, enei, phip=phip)
            val.symval = symval
            return val

        if p.sym == 'x':
            exc.val.append(make_component(['+', '-', '-'], [1, 0, 0]))
            exc.val.append(make_component(['-', '+', '+'], [0, 1, 0]))
            exc.val.append(make_component(['-', '+', '+'], [0, 0, 1]))
        elif p.sym == 'y':
            exc.val.append(make_component(['+', '-', '+'], [1, 0, 0]))
            exc.val.append(make_component(['-', '+', '-'], [0, 1, 0]))
            exc.val.append(make_component(['+', '-', '+'], [0, 0, 1]))
        elif p.sym == 'xy':
            exc.val.append(make_component(['++', '--', '-+'], [1, 0, 0]))
            exc.val.append(make_component(['--', '++', '+-'], [0, 1, 0]))
            exc.val.append(make_component(['-+', '+-', '++'], [0, 0, 1]))

        return exc

    def absorption(self, sig):
        """
        Absorption cross section for plane wave excitation.

        MATLAB: absorption.m / planewavestatmirror/absorption.m

        Parameters
        ----------
        sig : CompStruct or CompStructMirror
            CompStruct object containing surface charge

        Returns
        -------
        abs : ndarray
            Absorption cross section for each polarization
        """
        from ..geometry.comparticle_mirror import CompStructMirror
        if isinstance(sig, CompStructMirror):
            expanded = sig.expand()
            return sum(self.absorption(s) for s in expanded)

        # Induced dipole moment.
        # A5 fix: materialize cupy sig on host so numpy matmul does not raise.
        _sig_raw = sig.sig
        sig_arr = _sig_raw.get() if (hasattr(_sig_raw, 'get')
            and not isinstance(_sig_raw, np.ndarray)) else np.asarray(_sig_raw)
        # area: (nfaces,), pos: (nfaces, 3), sig: (nfaces,) or (nfaces, npol)
        area_pos = sig.p.area[:, np.newaxis] * sig.p.pos  # (nfaces, 3)

        if sig_arr.ndim == 1:
            # Single polarization
            dip = area_pos.T @ sig_arr  # (3, nfaces) @ (nfaces,) = (3,)
            dip = dip.reshape(3, 1)
        else:
            # Multiple polarizations
            dip = area_pos.T @ sig_arr  # (3, nfaces) @ (nfaces, npol) = (3, npol)

        # MATLAB: absorption.m lines 15-17
        # Dielectric function and wavenumber
        eps_func = sig.p.eps[self.medium - 1]  # Python 0-indexed
        eps_val, k = eps_func(sig.enei)

        # MATLAB: absorption.m line 20
        # abs = 4 * pi * k .* imag(dot(transpose(obj.pol), dip, 1))

        # pol: (npol, 3), dip: (3, npol)
        # pol.T · dip: element-wise for each polarization
        pol_dot_dip = np.sum(self.pol * dip.T, axis=1)  # (npol,)
        abs_cs = 4 * np.pi * k * np.imag(pol_dot_dip)

        if abs_cs.size == 1:
            return abs_cs[0]
        return abs_cs

    def scattering(self, sig):
        """
        Scattering cross section for plane wave excitation.

        MATLAB: scattering.m / planewavestatmirror/scattering.m

        Parameters
        ----------
        sig : CompStruct or CompStructMirror
            CompStruct object containing surface charge

        Returns
        -------
        sca : ndarray
            Scattering cross section for each polarization
        """
        from ..geometry.comparticle_mirror import CompStructMirror
        if isinstance(sig, CompStructMirror):
            expanded = sig.expand()
            return sum(self.scattering(s) for s in expanded)

        # Induced dipole moment.
        # A5 fix: materialize cupy sig on host so numpy matmul does not raise.
        _sig_raw = sig.sig
        sig_arr = _sig_raw.get() if (hasattr(_sig_raw, 'get')
            and not isinstance(_sig_raw, np.ndarray)) else np.asarray(_sig_raw)
        area_pos = sig.p.area[:, np.newaxis] * sig.p.pos  # (nfaces, 3)

        if sig_arr.ndim == 1:
            dip = area_pos.T @ sig_arr  # (3,)
            dip = dip.reshape(3, 1)
        else:
            dip = area_pos.T @ sig_arr  # (3, npol)

        # MATLAB: scattering.m lines 14-15
        eps_func = sig.p.eps[self.medium - 1]
        eps_val, k = eps_func(sig.enei)

        # Ensure k is real for embedding medium (should be lossless)
        k = np.real(k)

        # MATLAB: scattering.m line 17
        # sca = 8 * pi / 3 * k .^ 4 .* sum(abs(dip) .^ 2, 1)
        sca = 8 * np.pi / 3 * k**4 * np.sum(np.abs(dip)**2, axis=0)

        # Ensure real result
        sca = np.real(sca)

        if sca.size == 1:
            return float(sca[0])
        return sca

    def extinction(self, sig):
        """
        Extinction cross section for plane wave excitation.

        MATLAB: extinction.m

        Parameters
        ----------
        sig : CompStruct
            CompStruct object containing surface charge

        Returns
        -------
        ext : ndarray
            Extinction cross section for each polarization

        Notes
        -----
        MATLAB: extinction.m line 11

        Extinction is the sum of absorption and scattering:
        ext = sca + abs
        """
        # MATLAB: extinction.m line 11
        return self.scattering(sig) + self.absorption(sig)

    def __call__(self, p, enei):
        """
        External potential for use in BEMStat.

        MATLAB: subsref.m case '()'

        Parameters
        ----------
        p : ComParticle
            Particle surface
        enei : float
            Light wavelength in vacuum (nm)

        Returns
        -------
        exc : CompStruct
            CompStruct with potential information

        Examples
        --------
        >>> exc = planew(p, 600.0)
        """
        # MATLAB: subsref.m line 30
        return self.potential(p, enei)

    def __repr__(self):
        return "PlaneWaveStat(pol={}, medium={})".format(self.pol.tolist(), self.medium)

    def __str__(self):
        return (
            "Plane Wave Excitation (Quasistatic):\n"
            "  Polarization: {}\n"
            "  Medium: {}".format(self.pol, self.medium)
        )
