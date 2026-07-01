"""
Plane wave excitation for solution of full Maxwell equations.

This module provides plane wave excitation for retarded BEM simulations.

Reference:
    MATLAB MNPBEM Simulation/retarded/@planewaveret

Matches MATLAB MNPBEM implementation exactly.
"""

import numpy as np
from ..greenfun import CompStruct
from ..utils.gpu import _CUPY_OK, USE_GPU, _cp


class PlaneWaveRet(object):
    """
    Plane wave excitation for solution of full Maxwell equations.

    Provides electric and magnetic field excitation for retarded BEM solvers.
    Computes absorption, scattering, and extinction cross sections using
    full electromagnetic theory.

    Parameters
    ----------
    pol : ndarray
        Light polarization vector(s), shape (3,) or (npol, 3)
    dir : ndarray
        Light propagation direction vector(s), shape (3,) or (npol, 3)
    medium : int, optional
        Medium index for excitation (default: 1)
    pinfty : object, optional
        Unit sphere at infinity for spectrum calculations

    Attributes
    ----------
    pol : ndarray
        Light polarization vector(s)
    dir : ndarray
        Light propagation direction vector(s)
    medium : int
        Medium index through which particle is excited
    spec : object
        Spectrum object for cross section calculations

    Notes
    -----
    MATLAB: Simulation/retarded/@planewaveret/planewaveret.m

    The plane wave is given by:
        E = E₀ exp(ik·r)
        H = (n_b/Z₀) k̂ × E

    where n_b is the refractive index of the medium.

    Cross sections are computed using:
    - Extinction: Optical theorem (forward scattering)
    - Scattering: Radiated power integration
    - Absorption: ext - sca

    Examples
    --------
    >>> import numpy as np
    >>> from mnpbem.simulation import PlaneWaveRet
    >>>
    >>> # X-polarized plane wave propagating in +z direction
    >>> pol = np.array([1.0, 0.0, 0.0])
    >>> dir = np.array([0.0, 0.0, 1.0])
    >>> exc = PlaneWaveRet(pol, dir)
    >>>
    >>> # Get excitation potential at wavelength 600nm
    >>> pot = exc(p, 600.0)
    """

    # Class constants
    # MATLAB: @planewaveret line 5-8
    name = 'planewave'
    needs = {'sim': 'ret'}

    def __init__(self, pol, dir, medium=1, **options):
        """
        Initialize plane wave excitation.

        MATLAB: planewaveret.m + init.m

        Parameters
        ----------
        pol : array_like
            Light polarization vector(s), shape (3,) or (npol, 3)
        dir : array_like
            Light propagation direction vector(s), shape (3,) or (npol, 3)
        medium : int, optional
            Medium index for excitation (default: 1)
        **options : dict
            Additional options:
                - pinfty : unit sphere at infinity
                - medium : can also be specified in options
        """
        # MATLAB: init.m line 18
        if pol is None:
            raise ValueError("PlaneWaveRet: 'pol' must be a polarization vector, got None.")
        if dir is None:
            raise ValueError("PlaneWaveRet: 'dir' must be a propagation direction vector, got None.")
        self.pol = np.asarray(pol)
        self.dir = np.asarray(dir)

        if self.pol.dtype == object or self.pol.size == 0:
            raise ValueError(
                "PlaneWaveRet: 'pol' must be a numeric array of shape (3,) or (npol, 3).")
        if self.dir.dtype == object or self.dir.size == 0:
            raise ValueError(
                "PlaneWaveRet: 'dir' must be a numeric array of shape (3,) or (npol, 3).")

        # Ensure 2D arrays
        if self.pol.ndim == 1:
            if self.pol.size != 3:
                raise ValueError(
                    "PlaneWaveRet: 1D 'pol' must have length 3, got length {}."
                    .format(self.pol.size))
            self.pol = self.pol.reshape(1, -1)
        elif self.pol.ndim == 2:
            if self.pol.shape[1] != 3:
                raise ValueError(
                    "PlaneWaveRet: 2D 'pol' must have shape (npol, 3), got {}.".format(self.pol.shape))
        else:
            raise ValueError("PlaneWaveRet: 'pol' must be 1D or 2D, got {}D.".format(self.pol.ndim))

        if self.dir.ndim == 1:
            if self.dir.size != 3:
                raise ValueError(
                    "PlaneWaveRet: 1D 'dir' must have length 3, got length {}."
                    .format(self.dir.size))
            self.dir = self.dir.reshape(1, -1)
        elif self.dir.ndim == 2:
            if self.dir.shape[1] != 3:
                raise ValueError(
                    "PlaneWaveRet: 2D 'dir' must have shape (npol, 3), got {}.".format(self.dir.shape))
        else:
            raise ValueError("PlaneWaveRet: 'dir' must be 1D or 2D, got {}D.".format(self.dir.ndim))

        # MATLAB: init.m line 24
        self.medium = options.get('medium', medium)

        # MATLAB: init.m lines 26-30
        # Initialize spectrum (for scattering calculations)
        pinfty_arg = options.get('pinfty', None)
        if pinfty_arg is not None:
            # User provided pinfty
            from ..spectrum import SpectrumRet
            self.spec = SpectrumRet(pinfty_arg, medium=self.medium)
        else:
            # MATLAB: obj.spec = spectrumret(trisphere(256, 2), 'medium', obj.medium)
            from ..spectrum import SpectrumRet
            self.spec = SpectrumRet(medium=self.medium)  # Uses MATLAB pinfty256.bin

    def field(self, p, enei, inout=1):
        """
        Electric and magnetic field for plane wave excitation.

        MATLAB: field.m

        Parameters
        ----------
        p : ComParticle
            Particle or points where fields are computed
        enei : float
            Light wavelength in vacuum (nm)
        inout : int, optional
            Compute fields at inner (inout=1, default) or outer (inout=2) surface

        Returns
        -------
        exc : CompStruct
            CompStruct object containing electric field 'e' and magnetic field 'h'

        Notes
        -----
        MATLAB: field.m lines 15-49

        For a plane wave:
            E = E₀ exp(i k·r)
            H = (n_b/Z₀) dir × E

        where n_b = √ε is the refractive index.
        """
        # MATLAB: field.m line 18
        # Refractive index
        eps_func = p.eps[self.medium - 1]  # Python 0-indexed
        eps_val, _ = eps_func(enei)
        nb = np.sqrt(eps_val)

        # MATLAB: field.m lines 20-21
        # Wavenumbers
        k0 = 2 * np.pi / enei
        k = k0 * nb

        # MATLAB: field.m lines 23-26
        pol = self.pol
        dir = self.dir

        # Assert orthogonality (MATLAB: field.m line 26)
        dot_prod = np.sum(pol * dir, axis=1)
        assert np.allclose(dot_prod, 0), "Polarization and propagation direction must be orthogonal"

        # MATLAB: field.m lines 31-34
        # Index to excited faces
        ind = np.where(p.inout[:, inout - 1] == self.medium)[0]
        # Get all face indices for excited particles
        face_indices = []
        for i in ind:
            face_indices.extend(p.index_func(i + 1))  # 1-indexed in MATLAB
        ind = np.array(face_indices, dtype = int) if face_indices else np.array([], dtype = int)

        # MATLAB: field.m lines 37-38
        npol = pol.shape[0]
        e = np.zeros((p.n, 3, npol), dtype=complex)
        h = np.zeros((p.n, 3, npol), dtype=complex)

        # MATLAB: field.m lines 40-46
        if len(ind) > 0:
            use_gpu_path = _CUPY_OK and USE_GPU
            if use_gpu_path:
                pos_g = _cp.asarray(p.pos[ind, :])
                dir_g = _cp.asarray(dir)
                pol_g = _cp.asarray(pol)
                for i in range(npol):
                    phase_g = _cp.exp(1j * k * (pos_g @ dir_g[i, :])) / (1j * k0)
                    e_block = 1j * k0 * phase_g[:, None] * pol_g[i, :]
                    e[ind, :, i] = _cp.asnumpy(e_block)
                    e_full = e[:, :, i]
                    dir_rep = np.tile(dir[i, :], (p.n, 1))
                    h[:, :, i] = nb * np.cross(dir_rep, e_full)
            else:
                for i in range(npol):
                    # Phase factor: exp(i k r·dir)
                    phase = np.exp(1j * k * p.pos[ind, :] @ dir[i, :]) / (1j * k0)

                    # MATLAB: field.m line 43
                    # Electric field: E = i k₀ phase * pol
                    e[ind, :, i] = 1j * k0 * phase[:, np.newaxis] * pol[i, :]

                    # MATLAB: field.m lines 44-45
                    # Magnetic field: H = n_b dir × E
                    e_full = e[:, :, i]
                    dir_rep = np.tile(dir[i, :], (p.n, 1))
                    h[:, :, i] = nb * np.cross(dir_rep, e_full)

        # Squeeze if only one polarization
        if npol == 1:
            e = e[:, :, 0]
            h = h[:, :, 0]

        # MATLAB: field.m line 49
        return CompStruct(p, enei, e=e, h=h)

    def potential(self, p, enei):
        """
        Potential of plane wave excitation for use in BEMRet.

        MATLAB: potential.m

        Parameters
        ----------
        p : ComParticle
            Particle surface where potential is computed
        enei : float
            Light wavelength in vacuum (nm)

        Returns
        -------
        exc : CompStruct
            CompStruct with scalar and vector potentials and their derivatives

        Notes
        -----
        MATLAB: potential.m lines 1-59

        For a plane wave in Lorenz gauge:
            A = (E₀/ik₀) exp(ik·r)
            A' = (dir·nvec) A

        where A is the vector potential.
        """
        # MATLAB: potential.m line 13
        exc = CompStruct(p, enei)

        # MATLAB: potential.m lines 17-20
        pol = self.pol
        dir = self.dir

        # Assert orthogonality
        dot_prod = np.sum(pol * dir, axis=1)
        assert np.allclose(dot_prod, 0), "Polarization and propagation direction must be orthogonal"

        # MATLAB: potential.m lines 23-27
        eps_func = p.eps[self.medium - 1]
        eps_val, _ = eps_func(enei)
        nb = np.sqrt(eps_val)

        k0 = 2 * np.pi / enei
        k = k0 * nb

        # MATLAB: potential.m line 30
        # Loop over inside and outside of particle surfaces
        for inout in range(1, 3):  # 1, 2
            # MATLAB: potential.m lines 33-34
            npol = pol.shape[0]
            a = np.zeros((p.nfaces, 3, npol), dtype=complex)
            ap = np.zeros((p.nfaces, 3, npol), dtype=complex)

            # MATLAB: potential.m lines 37-38
            # Index to excited faces
            ind = np.where(p.inout[:, inout - 1] == self.medium)[0]
            face_indices = []
            for i in ind:
                face_indices.extend(p.index_func(i + 1))  # 1-indexed in MATLAB
            ind = np.array(face_indices) if face_indices else np.array([], dtype=int)

            # MATLAB: potential.m line 40-48
            if len(ind) > 0:
                use_gpu_path = _CUPY_OK and USE_GPU
                if use_gpu_path:
                    pos_g = _cp.asarray(p.pos[ind, :])
                    nvec_g = _cp.asarray(p.nvec[ind, :])
                    dir_g = _cp.asarray(dir)
                    pol_g = _cp.asarray(pol)
                    for i in range(npol):
                        phase_g = _cp.exp(1j * k * (pos_g @ dir_g[i, :])) / (1j * k0)
                        nvec_dot_dir_g = nvec_g @ dir_g[i, :]
                        a_block = phase_g[:, None] * pol_g[i, :]
                        ap_block = (1j * k * nvec_dot_dir_g)[:, None] * phase_g[:, None] * pol_g[i, :]
                        a[ind, :, i] = _cp.asnumpy(a_block)
                        ap[ind, :, i] = _cp.asnumpy(ap_block)
                else:
                    for i in range(npol):
                        # MATLAB: potential.m line 42
                        # Phase factor
                        phase = np.exp(1j * k * p.pos[ind, :] @ dir[i, :]) / (1j * k0)

                        # MATLAB: potential.m line 44
                        # Vector potential: A = phase * pol
                        a[ind, :, i] = phase[:, np.newaxis] * pol[i, :]

                        # MATLAB: potential.m lines 45-46
                        # Surface derivative: A' = (i k nvec·dir) * phase * pol
                        nvec_dot_dir = p.nvec[ind, :] @ dir[i, :]
                        ap[ind, :, i] = (1j * k * nvec_dot_dir)[:, np.newaxis] * phase[:, np.newaxis] * pol[i, :]

            # Squeeze if single polarization
            if npol == 1:
                a = a[:, :, 0]
                ap = ap[:, :, 0]

            # MATLAB: potential.m lines 51-56
            if inout == 1:
                exc = exc.set(a1=a, a1p=ap)
            else:
                exc = exc.set(a2=a, a2p=ap)

        return exc

    def absorption(self, sig):
        """
        Absorption cross section for plane wave excitation.

        MATLAB: absorption.m

        Parameters
        ----------
        sig : CompStruct
            CompStruct object containing surface currents

        Returns
        -------
        abs : ndarray
            Absorption cross section

        Notes
        -----
        MATLAB: absorption.m line 11

        Absorption is computed from:
        abs = ext - sca
        """
        # MATLAB: absorption.m line 11
        ext = self.extinction(sig)
        sca, _ = self.scattering(sig)  # scattering returns (sca, dsca) tuple
        return ext - sca

    def scattering(self, sig):
        """
        Scattering cross section for plane wave excitation.

        MATLAB: scattering.m

        Parameters
        ----------
        sig : CompStruct
            CompStruct object containing surface currents

        Returns
        -------
        sca : ndarray
            Scattering cross section
        dsca : CompStruct
            Differential scattering cross section

        Notes
        -----
        MATLAB: scattering.m lines 1-20

        Uses spectrum object to compute radiated power, normalized
        to incoming power (0.5 * n_b).
        """
        # MATLAB: scattering.m line 13
        # Total and differential radiated power
        sca, dsca = self.spec.scattering(sig)

        # MATLAB: scattering.m line 16
        # Refractive index of embedding medium
        # MATLAB: nb = sqrt(sig.p.eps{1}(sig.enei))
        eps_func = sig.p.eps[self.medium - 1]
        eps_val, _ = eps_func(sig.enei)
        nb = np.real(np.sqrt(eps_val))

        # MATLAB: scattering.m line 19
        # The scattering cross section is the radiated power normalized to
        # the incoming power, which is proportional to 0.5 * nb
        sca = sca / (0.5 * nb)
        if dsca is not None and hasattr(dsca, 'dsca'):
            from ..greenfun import CompStruct
            dsca = CompStruct(dsca.p, dsca.enei, dsca = dsca.dsca / (0.5 * nb))

        return sca, dsca

    def extinction(self, sig):
        """
        Extinction cross section for plane wave excitation.

        MATLAB: extinction.m

        Parameters
        ----------
        sig : CompStruct
            CompStruct object containing surface currents

        Returns
        -------
        ext : ndarray or float
            Extinction cross section

        Notes
        -----
        MATLAB: extinction.m lines 1-17

        Uses optical theorem:
            ext = (4*pi/k) * Im(conj(pol) . E_forward)

        where E_forward is the far-field scattering amplitude evaluated
        in the forward propagation direction(s).

        MATLAB INNER() calls DOT which conjugates the first argument,
        hence conj(pol) . e_forward.
        """
        # MATLAB: extinction.m line 12
        # Far-field amplitude in the forward direction(s)
        # [field, k] = farfield(obj.spec, sig, obj.dir)
        field = self.spec.farfield(sig, self.dir)

        # Get wavenumber
        _, k = sig.p.eps[self.medium - 1](sig.enei)

        # MATLAB: extinction.m line 16
        # ext = 4*pi/k * diag(imag(inner(obj.pol, field.e))).'
        # MATLAB inner() conjugates the first argument: inner(a, b) = conj(a) . b
        #
        # field.e shape: (npol, 3) when direction has npol rows (one per polarization)
        # or (npol, 3, npol) if multiple polarizations
        # self.pol shape: (npol, 3)
        e_forward = field.e  # (npol, 3) or (npol, 3, npol_sig)

        npol = self.pol.shape[0]

        if e_forward.ndim == 2:
            # e_forward: (npol_dir, 3)
            # Each direction row i corresponds to polarization i
            # Compute conj(pol[i]) . e_forward[i] for each i
            pol_dot_e = np.sum(np.conj(self.pol) * e_forward[:npol, :], axis = 1)
        else:
            # e_forward: (npol_dir, 3, npol_sig)
            # Compute diagonal: for each polarization i, take
            #   conj(pol[i]) . e_forward[i, :, i]
            pol_dot_e = np.zeros(npol, dtype = complex)
            for i in range(npol):
                pol_dot_e[i] = np.sum(np.conj(self.pol[i]) * e_forward[i, :, i])

        ext = 4 * np.pi / k * np.imag(pol_dot_e)
        ext = np.real(ext)

        # Return scalar for single polarization
        if npol == 1:
            return float(ext[0])
        return ext

    def __call__(self, p, enei):
        """
        External potential for use in BEMRet.

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
        # MATLAB: subsref.m line 31
        return self.potential(p, enei)

    def __repr__(self):
        return "PlaneWaveRet(pol={}, dir={}, medium={})".format(
            self.pol.tolist(), self.dir.tolist(), self.medium)

    def __str__(self):
        return (
            "Plane Wave Excitation (Retarded):\n"
            "  Polarization: {}\n"
            "  Direction: {}\n"
            "  Medium: {}".format(self.pol, self.dir, self.medium)
        )
