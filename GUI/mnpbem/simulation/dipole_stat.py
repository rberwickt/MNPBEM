"""
Dipole excitation in quasistatic approximation.

Excitation of an oscillating dipole in quasistatic approximation.
Given a dipole oscillating with some frequency, DipoleStat computes
the external potentials needed for BEM simulations and determines
the total and radiative scattering rates for the dipole.

Reference:
    MATLAB MNPBEM Simulation/static/@dipolestat

Matches MATLAB MNPBEM implementation exactly.
"""

import numpy as np
from ..greenfun import CompStruct
from ..utils.matlab_compat import msqrt


class DipoleStat(object):
    """
    Excitation of an oscillating dipole in quasistatic approximation.

    Provides electric field and potential excitation for quasistatic BEM solvers.
    Computes total and radiative decay rates for oscillating dipoles.

    Parameters
    ----------
    pt : ComPoint or Particle
        Compound of points (or compoint) for dipole positions
    dip : ndarray, optional
        Directions of dipole moments, shape (3,), (3, 3), or (npt, 3, ndip)
        Default: eye(3) - three orthogonal dipoles
    full : bool, optional
        If True, dipole moments are given at each position
        If False, same dipole moments are used for all positions (default)
    **options : dict
        Additional arguments to be passed to CompGreenStat

    Attributes
    ----------
    pt : ComPoint
        Dipole positions
    dip : ndarray
        Dipole moments, shape (npt, 3, ndip)
    varargin : tuple
        Additional arguments for Green function

    Notes
    -----
    MATLAB: Simulation/static/@dipolestat/dipolestat.m

    The electric field of a static dipole at position r is:
        E = (3(p·r̂)r̂ - p) / (4πε₀ε r³)

    For quasistatic regime (wavelength >> size), this gives the induced fields.

    Examples
    --------
    >>> import numpy as np
    >>> from mnpbem.simulation import DipoleStat
    >>>
    >>> # Single dipole at origin, z-polarized
    >>> from mnpbem import ComPoint
    >>> pt = ComPoint([np.array([[0, 0, 0]])], [1])
    >>> dip = np.array([0, 0, 1])
    >>> exc = DipoleStat(pt, dip)
    """

    # Class constants
    # MATLAB: @dipolestat line 8-11
    name = 'dipole'
    needs = {'sim': 'stat'}

    def __init__(self, pt, dip=None, full=False, **options):
        """
        Initialize dipole excitation.

        MATLAB: dipolestat.m + init.m

        Parameters
        ----------
        pt : ComPoint
            Compound of points for dipole positions
        dip : array_like, optional
            Directions of dipole moments
            - None or not provided: eye(3) - three orthogonal dipoles
            - (3,): single dipole direction, same for all positions
            - (3, 3): three dipole directions, same for all positions
            - (npt, 3, ndip): full dipole moments at each position (requires full=True)
        full : bool, optional
            Indicate that dipole moments are given at each position
        **options : dict
            Additional arguments to be passed to CompGreenStat
        """
        # MATLAB: dipolestat.m line 33
        self.pt = pt
        # MATLAB: dipolestat.m line 34
        self._init(dip, full, **options)

    def _init(self, dip=None, full=False, **options):
        """
        Initialize dipole moments.

        MATLAB: init.m

        Parameters
        ----------
        dip : array_like or None
            Dipole moments
        full : bool
            Whether dipole moments are given at all positions
        **options : dict
            Additional arguments
        """
        # MATLAB: init.m lines 12-22
        if dip is None:
            # Default values for dipole orientations
            dip = np.eye(3)
            full = False

        # Convert to numpy array
        dip = np.asarray(dip, dtype=float)

        # MATLAB: init.m line 25
        # Save options for Green function
        self.varargin = options

        # MATLAB: init.m lines 28-40
        # Dipole moments given at all positions
        if full:
            # MATLAB: init.m lines 29-32
            if dip.ndim == 2:
                dip = dip.reshape(dip.shape + (1,))
            self.dip = dip
        else:
            # Same dipole moments for all positions
            # MATLAB: init.m lines 34-39
            if dip.ndim == 1:
                dip = dip.reshape(1, -1)

            # Reshape: (ndip, 3) -> (1, 3, ndip) -> (npt, 3, ndip)
            # MATLAB: reshape(dip.', [1, fliplr(size(dip))])
            dip_reshaped = dip.T.reshape(1, dip.shape[1], dip.shape[0])
            # Replicate for all positions
            # MATLAB: repmat(..., [obj.pt.n, 1, 1])
            self.dip = np.tile(dip_reshaped, (self.pt.n, 1, 1))

    def field(self, p, enei):
        """
        Electric field for dipole excitation.

        MATLAB: field.m

        Parameters
        ----------
        p : ComParticle or Particle
            Points or particle surface where field is computed
        enei : float
            Light wavelength in vacuum (nm)

        Returns
        -------
        exc : CompStruct
            CompStruct object containing electric field 'e'
            Shape: (n, 3, npt, ndip) where last two dimensions correspond
            to dipole positions and dipole moments

        Notes
        -----
        MATLAB: field.m

        The electric field is computed using Jackson Eq. (4.13):
            E = [3(p·r̂)r̂ - p] / (4πε₀ε r³)

        screened by the dielectric function of the embedding medium.
        """
        # MATLAB: field.m line 15
        pt = self.pt

        # MATLAB: field.m line 17
        # Compute electric field for unit dipole
        e = self._efield(p.pos, pt.pos, self.dip, pt.eps1(enei))

        # MATLAB: field.m line 47
        # Save in compstruct
        return CompStruct(p, enei, e=e)

    def _efield(self, pos1, pos2, dip, eps):
        """
        Electric field at POS1 for dipole positions POS2 and dipole moments DIP.

        MATLAB: field.m lines 50-80 (efield function)

        Parameters
        ----------
        pos1 : ndarray
            Observation positions, shape (n1, 3)
        pos2 : ndarray
            Dipole positions, shape (n2, 3)
        dip : ndarray
            Dipole moments, shape (n2, 3, ndip)
        eps : ndarray
            Dielectric function at dipole positions, shape (n2,)

        Returns
        -------
        e : ndarray
            Electric field, shape (n1, 3, n2, ndip)
        """
        # MATLAB: field.m line 55
        n1 = pos1.shape[0]
        n2 = pos2.shape[0]
        ndip = dip.shape[2]

        # Allocate output array
        e = np.zeros((n1, 3, n2, ndip), dtype=complex)

        # MATLAB: field.m lines 58-64
        # Distance vector
        x = pos1[:, 0:1] - pos2[:, 0].T  # (n1, n2)
        y = pos1[:, 1:2] - pos2[:, 1].T
        z = pos1[:, 2:3] - pos2[:, 2].T

        # Distance
        r = msqrt(x**2 + y**2 + z**2)

        # Normalize distance vector
        x = x / r
        y = y / r
        z = z / r

        # MATLAB: field.m lines 66-79
        for i in range(ndip):
            # Dipole moment
            dx = np.tile(dip[:, 0, i], (n1, 1))  # (n1, n2)
            dy = np.tile(dip[:, 1, i], (n1, 1))
            dz = np.tile(dip[:, 2, i], (n1, 1))

            # Inner product [x, y, z] · dip
            inner = x * dx + y * dy + z * dz

            # Electric field [Jackson Eq. (4.13)]
            # Screen electric field of unit dipole by dielectric function
            # MATLAB: field.m lines 76-78
            e[:, 0, :, i] = (3 * x * inner - dx) / (r**3 * eps)
            e[:, 1, :, i] = (3 * y * inner - dy) / (r**3 * eps)
            e[:, 2, :, i] = (3 * z * inner - dz) / (r**3 * eps)

        return e

    def potential(self, p, enei):
        """
        Potential of dipole excitation for use in BEMStat.

        MATLAB: potential.m

        Parameters
        ----------
        p : ComParticle or Particle
            Particle surface where potential is computed
        enei : float
            Light wavelength in vacuum (nm)

        Returns
        -------
        exc : CompStruct
            CompStruct with surface derivative 'phip' of scalar potential
            Shape: (nfaces, npt, ndip) where last two dimensions correspond
            to dipole positions and dipole moments

        Notes
        -----
        MATLAB: potential.m

        The surface derivative is computed from the electric field:
            φ' = -nvec · E
        """
        # MATLAB: potential.m line 15
        # Electric field
        exc = self.field(p, enei)

        # MATLAB: potential.m line 17
        # Surface derivative of scalar potential: -nvec · E
        # nvec: (nfaces, 3), exc.e: (nfaces, 3, npt, ndip)
        # Result: (nfaces, npt, ndip)
        phip = -np.einsum('ij,ij...->i...', p.nvec, exc.e)

        return CompStruct(p, enei, phip=phip)

    def decayrate(self, sig):
        """
        Total and radiative decay rate for oscillating dipole.

        Returns decay rates in units of the free-space decay rate.

        MATLAB: decayrate.m

        Parameters
        ----------
        sig : CompStruct
            CompStruct object containing surface charge

        Returns
        -------
        tot : ndarray
            Total decay rate, shape (npt, ndip)
        rad : ndarray
            Radiative decay rate, shape (npt, ndip)
        rad0 : ndarray
            Free-space decay rate, shape (npt, ndip)

        Notes
        -----
        MATLAB: decayrate.m

        The decay rates are computed using Wigner-Weisskopf theory:
            Γ₀ = (4/3) k₀³ (free-space)
            Γ_tot = Γ₀ [1 + Im(E_ind · p) / (0.5 n_b Γ₀)]
            Γ_rad = |n_b² dip_ind + dip|²
        """
        # MATLAB: decayrate.m line 16
        p, enei = sig.p, sig.enei

        # MATLAB: decayrate.m lines 18-22
        # Green function (persistent in MATLAB, we create new each time for simplicity)
        from ..greenfun import CompGreenStat
        g = CompGreenStat(self.pt, sig.p, **self.varargin)

        # MATLAB: decayrate.m line 25
        # Induced electric field.
        # A5 fix: materialize cupy sig.sig on host before invoking the Green
        # function so numpy matmul does not raise on a cupy operand.
        _sig_raw = sig.sig
        if hasattr(_sig_raw, 'get') and not isinstance(_sig_raw, np.ndarray):
            sig.sig = _sig_raw.get()
        field_struct = g.field(sig)
        _e_raw = field_struct.e
        e = (_e_raw.get() if (hasattr(_e_raw, 'get')
            and not isinstance(_e_raw, np.ndarray)) else np.asarray(_e_raw))

        # MATLAB: decayrate.m line 27
        # Wigner-Weisskopf decay rate in free space
        gamma = 4 / 3 * (2 * np.pi / sig.enei) ** 3

        # MATLAB: decayrate.m lines 30-31
        # Induced dipole moment
        # sig.p.pos: (nfaces, 3), sig.p.area: (nfaces,), sig.sig: (nfaces,) or (nfaces, npt, ndip)
        area_pos = sig.p.pos * sig.p.area[:, np.newaxis]  # (nfaces, 3)

        npt = self.pt.n
        ndip = self.dip.shape[2]

        # Reshape e from (npt, 3, npol) to (npt, 3, npt, ndip) for decayrate indexing
        # (matches DipoleRet.decayrate, DipoleStatLayer.decayrate).
        if e.ndim == 3 and e.shape[2] == npt * ndip:
            e = e.reshape(npt, 3, npt, ndip)
        elif e.ndim == 3 and e.shape[2] == ndip and npt == 1:
            e = e.reshape(npt, 3, npt, ndip)
        # A5 fix: materialize cupy sig on host so numpy matmul does not raise.
        _sig_raw2 = sig.sig
        sig_arr = (_sig_raw2.get() if (hasattr(_sig_raw2, 'get')
            and not isinstance(_sig_raw2, np.ndarray)) else np.asarray(_sig_raw2))
        # Reshape sig to (nfaces, npt*ndip) for matrix multiply
        if sig_arr.ndim == 1:
            sig_flat = sig_arr.reshape(-1, 1)
        else:
            sig_flat = sig_arr.reshape(sig_arr.shape[0], -1)
        indip = area_pos.T @ sig_flat  # (3, npt*ndip)
        indip = indip.reshape(3, npt, ndip)

        # MATLAB: decayrate.m lines 33-35
        # Decay rates for oscillating dipole
        npt = self.pt.n
        ndip = self.dip.shape[2]
        tot = np.zeros((npt, ndip))
        rad = np.zeros((npt, ndip))
        rad0 = np.zeros((npt, ndip))

        # MATLAB: decayrate.m lines 37-60
        for ipos in range(npt):
            for idip in range(ndip):
                # MATLAB: decayrate.m line 41
                # Refractive index
                nb = np.sqrt(self.pt.eps1(sig.enei)[ipos])
                if np.imag(nb) != 0:
                    import warnings
                    warnings.warn('Dipole embedded in medium with complex dielectric function')

                # MATLAB: decayrate.m line 48
                # Dipole moment of oscillator
                dip = self.dip[ipos, :, idip]

                # MATLAB: decayrate.m lines 50-52
                # Radiative decay rate
                # DIP is the transition dipole moment for the dipole in vacuum,
                # which is screened by the dielectric function
                if indip.ndim == 3:
                    indip_i = indip[:, ipos, idip]
                else:
                    indip_i = indip[:, 0]
                rad[ipos, idip] = np.linalg.norm(nb**2 * indip_i + dip) ** 2

                # MATLAB: decayrate.m lines 55-56
                # Total decay rate
                e_i = e[ipos, :, ipos, idip]
                tot[ipos, idip] = 1 + np.imag(e_i @ dip) / (0.5 * nb * gamma)

                # MATLAB: decayrate.m line 58
                # Free-space decay rate
                rad0[ipos, idip] = nb * gamma

        return tot, rad, rad0

    def farfield(self, spec, enei):
        """
        Electromagnetic fields of dipoles in the far-field limit.

        MATLAB: farfield.m

        Parameters
        ----------
        spec : SpectrumStat
            SPECTRUMSTAT object
        enei : float
            Wavelength of light in vacuum (nm)

        Returns
        -------
        field : CompStruct
            CompStruct object that holds far-fields

        Notes
        -----
        MATLAB: farfield.m

        Far-field amplitude:
            E_far = (k²/ε) exp(-ik·r) dir × (dir × dip)
            H_far = (k²/n_b) exp(-ik·r) dir × dip
        """
        # MATLAB: farfield.m line 13
        # Normal vectors of unit sphere at infinity
        dir = spec.pinfty.nvec

        # MATLAB: farfield.m lines 14-19
        # Table of dielectric functions
        epstab = self.pt.eps
        # Wavenumber of light in medium
        eps_val, k = epstab[spec.medium - 1](enei)
        # Refractive index
        nb = np.sqrt(eps_val)

        # MATLAB: farfield.m lines 22-26
        pt = self.pt
        dip = self.dip
        # Dielectric screening of dipoles
        # MATLAB: dip = matmul(diag(eps ./ pt.eps1(enei)), dip)
        screening = eps_val / pt.eps1(enei)
        dip = screening[:, np.newaxis, np.newaxis] * dip

        # MATLAB: farfield.m lines 29-31
        # Make compstruct object
        from ..particle import ComParticle
        field = CompStruct(
            ComParticle(epstab, [spec.pinfty], spec.medium), enei
        )

        # MATLAB: farfield.m lines 33-35
        n1 = dir.shape[0]
        n2 = dip.shape[0]
        n3 = dip.shape[2]

        # MATLAB: farfield.m lines 38-41
        # Far fields
        e = np.zeros((n1, 3, n2, n3), dtype=complex)
        h = np.zeros((n1, 3, n2, n3), dtype=complex)

        # Find dipoles that are connected through medium
        ind = pt.index[pt.inout == spec.medium]

        # MATLAB: farfield.m lines 43-56
        if len(ind) > 0:
            # Green function for k r -> ∞
            # MATLAB: g = exp(-1i * k * matmul(dir, permute(pt.pos, [2, 1, 3])))
            g = np.exp(-1j * k * (dir @ pt.pos.T))  # (n1, n2)
            g = g[:, np.newaxis, :, np.newaxis]  # (n1, 1, n2, 1)
            g = np.tile(g, (1, 3, 1, n3))  # (n1, 3, n2, n3)

            # Reshape direction and dipole orientation
            dir_rep = dir[:, :, np.newaxis, np.newaxis]  # (n1, 3, 1, 1)
            dir_rep = np.tile(dir_rep, (1, 1, n2, n3))  # (n1, 3, n2, n3)

            dip_perm = dip.transpose(1, 0, 2)  # (3, n2, ndip)
            dip_rep = dip_perm[np.newaxis, :, :, :]  # (1, 3, n2, ndip)
            dip_rep = np.tile(dip_rep, (n1, 1, 1, 1))  # (n1, 3, n2, ndip)

            # Far-field amplitude
            # MATLAB: h = cross(dir, dip, 2) .* g
            h = np.cross(dir_rep, dip_rep, axis=1) * g
            # MATLAB: e = cross(h, dir, 2)
            e = np.cross(h, dir_rep, axis=1)

            # MATLAB: field.m lines 54-55
            e = k**2 * e / eps_val
            h = k**2 * h / nb

        field.e = e
        field.h = h

        return field

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
        """
        # MATLAB: subsref.m line 25
        return self.potential(p, enei)

    def __repr__(self):
        return "DipoleStat(npt={}, ndip={})".format(self.pt.n, self.dip.shape[2])

    def __str__(self):
        return (
            "Dipole Excitation (Quasistatic):\n"
            "  Positions: {}\n"
            "  Dipole orientations: {}\n"
            "  Dipole shape: {}".format(
                self.pt.n, self.dip.shape[2], self.dip.shape)
        )
