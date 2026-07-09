"""
Retarded EELS excitation for full Maxwell equations.

Excitation of an electron beam with high kinetic energy. Given an electron
beam, EELSRet computes the external potentials needed for retarded BEM
simulations and determines the energy loss probability for the electrons.

Reference:
    Garcia de Abajo et al., PRB 65, 115418 (2002), RMP 82, 209 (2010).
    MATLAB MNPBEM Simulation/retarded/@eelsret

Matches MATLAB MNPBEM implementation exactly.
"""

import numpy as np
from scipy.special import kv as besselk
from typing import Optional, Tuple

from ..greenfun import CompStruct
from ..misc import EV2NM, BOHR, HARTREE, FINE, outer
from .eels_base import EELSBase


class EELSRet(EELSBase):
    """
    Electron energy loss spectroscopy with full Maxwell equations (retarded).

    Given an electron beam, computes the external potentials needed for
    retarded BEM simulations and determines the EELS loss probability,
    including surface loss, bulk (Cherenkov) loss, and radiative (photon)
    loss probabilities.

    Parameters
    ----------
    p : ComParticle
        Particle object for EELS simulation
    impact : ndarray, shape (nimp, 2)
        Impact parameters (x, y) of electron beams
    width : float
        Width of electron beam for potential smearing
    vel : float
        Electron velocity in units of speed of light
    cutoff : float, optional
        Distance for integration refinement (default: 10 * width)
    phiout : float, optional
        Half aperture collection angle of spectrometer (default: 1e-2)
    pinfty : object, optional
        Sphere at infinity for photon loss probability
    medium : int, optional
        Index of embedding medium (default: 1)

    Attributes
    ----------
    p : ComParticle
        Particle object
    impact : ndarray
        Impact parameters
    width : float
        Beam width
    vel : float
        Electron velocity / c
    phiout : float
        Half aperture collection angle
    spec : SpectrumRet
        Spectrum object for radiative loss calculations

    Notes
    -----
    MATLAB: Simulation/retarded/@eelsret/

    The surface loss is computed from boundary charges and currents:
        p_surf = (alpha^2 / (a0 * Eh * pi * v)) * Im(area' * (phi * (sig - v*h_z)))

    Bulk loss includes Cherenkov radiation inside particles.
    Radiative loss is obtained from far-field scattering on a unit sphere.
    """

    # Class constants
    # MATLAB: @eelsret line 9-11
    name = 'eels'
    needs = {'sim': 'ret'}

    def __init__(self,
            p: object,
            impact: np.ndarray,
            width: float,
            vel: float,
            cutoff: Optional[float] = None,
            phiout: float = 1e-2,
            pinfty: Optional[object] = None,
            medium: int = 1,
            **options) -> None:
        """
        Initialize retarded EELS excitation.

        MATLAB: eelsret.m constructor + init.m

        Parameters
        ----------
        p : ComParticle
            Particle object
        impact : ndarray, shape (nimp, 2)
            Impact parameters
        width : float
            Beam width
        vel : float
            Electron velocity / c
        cutoff : float, optional
            Distance for integration refinement
        phiout : float, optional
            Spectrometer half aperture angle
        pinfty : object, optional
            Sphere at infinity for photon loss probability
        medium : int, optional
            Index of embedding medium
        """
        # MATLAB: eelsret.m line 36
        super(EELSRet, self).__init__(
            p, impact, width, vel,
            cutoff = cutoff, phiout = phiout, **options)

        # MATLAB: eelsret.m line 38 -> init.m
        self._init_spectrum(pinfty = pinfty, medium = medium, **options)

    def _init_spectrum(self,
            pinfty: Optional[object] = None,
            medium: int = 1,
            **options) -> None:
        """
        Initialize sphere at infinity for photon loss probability.

        MATLAB: @eelsret/init.m

        Parameters
        ----------
        pinfty : object, optional
            Sphere at infinity
        medium : int, optional
            Embedding medium index
        """
        # MATLAB: init.m lines 5-12
        from ..spectrum import SpectrumRet

        if pinfty is not None:
            self.spec = SpectrumRet(pinfty, medium = medium)
        else:
            self.spec = SpectrumRet(medium = medium)

    def potential(self,
            p: object,
            enei: float) -> CompStruct:
        """
        Potential of electron beam excitation for use in BEMRet.

        MATLAB: @eelsret/potential.m

        Parameters
        ----------
        p : ComParticle
            Particle surface where potential is computed
        enei : float
            Light wavelength in vacuum (nm)

        Returns
        -------
        exc : CompStruct
            CompStruct with scalar and vector potentials and derivatives:
            phi1, phi1p, phi2, phi2p, a1, a1p, a2, a2p
        """
        # MATLAB: potential.m lines 16-22
        p_obj = self.p
        eps_vals = np.array([eps_func(enei) for eps_func in p_obj.eps], dtype = object)
        eps_arr = np.array([e[0] if hasattr(e, '__len__') else e for e in eps_vals], dtype = complex)

        gamma = 1.0 / np.sqrt(1 - eps_arr * self.vel ** 2)
        q = 2 * np.pi / (enei * self.vel)

        # ---- allocate compstruct for excitation ----
        # MATLAB: potential.m lines 24-34
        exc = CompStruct(p_obj, enei)
        n_faces = p_obj.n
        n_imp = self.impact.shape[0]

        exc.phi1 = np.zeros((n_faces, n_imp), dtype = complex)
        exc.phi1p = np.zeros((n_faces, n_imp), dtype = complex)
        exc.phi2 = np.zeros((n_faces, n_imp), dtype = complex)
        exc.phi2p = np.zeros((n_faces, n_imp), dtype = complex)
        exc.a1 = np.zeros((n_faces, 3, n_imp), dtype = complex)
        exc.a1p = np.zeros((n_faces, 3, n_imp), dtype = complex)
        exc.a2 = np.zeros((n_faces, 3, n_imp), dtype = complex)
        exc.a2p = np.zeros((n_faces, 3, n_imp), dtype = complex)

        # ---- excitation of embedding medium ----
        # MATLAB: potential.m lines 37-43
        # MATLAB: ind = find(any(p.inout == 1, 2)') — particle indices (1-indexed)
        ind = np.where(np.any(p_obj.inout == 1, axis = 1))[0] + 1  # 1-indexed

        phi, phip = self.potinfty(q, gamma[0], ind)

        exc = self._add_potential(exc, p_obj, phi, phip, 1, eps_arr[0], self.vel)

        # ---- excitation for other media ----
        # MATLAB: potential.m lines 46-56
        unique_mats = np.unique(p_obj.inout.ravel())
        for mat in unique_mats:
            if mat == 1:
                continue

            # MATLAB 1-indexed material
            mat_int = int(mat)
            # MATLAB: ind = find(any(p.inout == mat, 2)') — particle indices (1-indexed)
            ind = np.where(np.any(p_obj.inout == mat_int, axis = 1))[0] + 1

            phi, phip = self.potinfty(q, gamma[mat_int - 1], ind, mat_int)

            # MATLAB: `if ~all(phi == 0)` evaluates per-column then requires
            # all entries of the vector to be true. Skip addpotential if ANY
            # column is all-zero (e.g., far impact with no crossing).
            if np.all(np.any(phi != 0, axis = 0)):
                exc = self._add_potential(exc, p_obj, phi, phip, mat_int, eps_arr[mat_int - 1], self.vel)

        return exc

    def _add_potential(self,
            exc: CompStruct,
            p: object,
            phi: np.ndarray,
            phip: np.ndarray,
            mat: int,
            eps: complex,
            vel: float) -> CompStruct:
        """
        Add potential to external excitation.

        MATLAB: @eelsret/potential.m -> addpotential subfunction

        Parameters
        ----------
        exc : CompStruct
            CompStruct object for external potential
        p : ComParticle
            Particle object
        phi : ndarray
            Scalar potential
        phip : ndarray
            Surface derivative of scalar potential
        mat : int
            Index to medium (1-indexed)
        eps : complex
            Dielectric function
        vel : float
            Electron beam velocity / c

        Returns
        -------
        exc : CompStruct
            Updated CompStruct
        """
        # MATLAB: potential.m lines 72-73
        # Vector potential: a = vel * outer([0,0,1], phi)
        zhat = np.tile(np.array([[0, 0, 1]]), (p.n, 1))
        a = vel * outer(zhat, phi)
        ap = vel * outer(zhat, phip)

        # Scalar potential
        phi_scaled = phi / eps
        phip_scaled = phip / eps

        # MATLAB: potential.m lines 78-90
        # Index to inner and outer surface elements
        ind1 = []
        ind2 = []
        for ip in range(p.np):
            face_indices = getattr(p, "index_func", p.index)(ip + 1)
            if p.inout[ip, 0] == mat:
                ind1.extend(face_indices)
            if p.inout[ip, 1] == mat:
                ind2.extend(face_indices)
        ind1 = np.array(ind1, dtype = int) if len(ind1) > 0 else np.array([], dtype = int)
        ind2 = np.array(ind2, dtype = int) if len(ind2) > 0 else np.array([], dtype = int)

        if len(ind1) > 0:
            exc.phi1[ind1, :] = exc.phi1[ind1, :] + phi_scaled[ind1, :]
            exc.phi1p[ind1, :] = exc.phi1p[ind1, :] + phip_scaled[ind1, :]
            exc.a1[ind1, :, :] = exc.a1[ind1, :, :] + a[ind1, :, :]
            exc.a1p[ind1, :, :] = exc.a1p[ind1, :, :] + ap[ind1, :, :]

        if len(ind2) > 0:
            exc.phi2[ind2, :] = exc.phi2[ind2, :] + phi_scaled[ind2, :]
            exc.phi2p[ind2, :] = exc.phi2p[ind2, :] + phip_scaled[ind2, :]
            exc.a2[ind2, :, :] = exc.a2[ind2, :, :] + a[ind2, :, :]
            exc.a2p[ind2, :, :] = exc.a2p[ind2, :, :] + ap[ind2, :, :]

        return exc

    def loss(self,
            sig: object) -> Tuple[np.ndarray, np.ndarray]:
        """
        EELS loss probability.

        MATLAB: @eelsret/loss.m

        Parameters
        ----------
        sig : CompStruct
            Surface charge from BEMRet (contains sig1, sig2, h1, h2)

        Returns
        -------
        psurf : ndarray, shape (n_impact,)
            EELS loss probability from surface plasmons
        pbulk : ndarray, shape (n_impact,)
            Loss probability from bulk material
        """
        # MATLAB: loss.m lines 15-24
        p = self.p
        eps_vals = np.array([eps_func(sig.enei) for eps_func in p.eps], dtype = object)
        eps_arr = np.array([e[0] if hasattr(e, '__len__') else e for e in eps_vals], dtype = complex)
        # MATLAB: k = 2*pi/sig.enei * sqrt(eps)
        k_arr = 2 * np.pi / sig.enei * np.sqrt(eps_arr)

        gamma = 1.0 / np.sqrt(1 - eps_arr * self.vel ** 2)
        q = 2 * np.pi / (sig.enei * self.vel)

        # MATLAB: loss.m lines 26-27
        n_imp = self.impact.shape[0]
        psurf = np.zeros(n_imp)

        # ---- ensure sig arrays are 2D/3D for uniform indexing ----
        # BEMRet.solve() squeezes single-polarization: sig1=(n,), h1=(n,3)
        # MATLAB always has sig1=(n,npol), h1=(n,3,npol).
        # A5 fix: materialize cupy sig members on host so numpy matmul does
        # not raise on a cupy operand.
        def _h(x):
            return (x.get() if (hasattr(x, 'get')
                and not isinstance(x, np.ndarray)) else np.asarray(x))
        sig1 = _h(sig.sig1)
        sig2 = _h(sig.sig2)
        h1 = _h(sig.h1)
        h2 = _h(sig.h2)
        if sig1.ndim == 1:
            sig1 = sig1[:, np.newaxis]
        if sig2.ndim == 1:
            sig2 = sig2[:, np.newaxis]
        if h1.ndim == 2:
            h1 = h1[:, :, np.newaxis]
        if h2.ndim == 2:
            h2 = h2[:, :, np.newaxis]

        # MATLAB: loss.m lines 29-30
        # Auxiliary functions for energy loss
        # MATLAB: fun1 = @(ind)(sig.sig1(ind,:) - vel * squeeze(sig.h1(ind,3,:)))
        def fun1(ind: np.ndarray) -> np.ndarray:
            return sig1[ind, :] - self.vel * np.squeeze(h1[np.ix_(ind, [2], np.arange(h1.shape[2]))], axis = 1)

        def fun2(ind: np.ndarray) -> np.ndarray:
            return sig2[ind, :] - self.vel * np.squeeze(h2[np.ix_(ind, [2], np.arange(h2.shape[2]))], axis = 1)

        # MATLAB: loss.m lines 33-34
        # Potential for beam in embedding medium
        phi_infty, _ = self.potinfty(q, gamma[0])
        phi_inside, _ = self.potinside(-q, k_arr[0])
        phi = self.vel * (np.conj(phi_infty) - self.full(phi_inside))

        # MATLAB: loss.m lines 36-44
        # Faces with embedding medium at inside or outside
        # MATLAB: ind1 = p.index(find(p.inout(:,1)==1)')
        ind1 = []
        ind2 = []
        for ip in range(p.np):
            face_indices = getattr(p, "index_func", p.index)(ip + 1)
            if p.inout[ip, 0] == 1:
                ind1.extend(face_indices)
            if p.inout[ip, 1] == 1:
                ind2.extend(face_indices)
        ind1 = np.array(ind1, dtype = int) if len(ind1) > 0 else np.array([], dtype = int)
        ind2 = np.array(ind2, dtype = int) if len(ind2) > 0 else np.array([], dtype = int)

        if len(ind1) > 0:
            psurf -= np.imag(p.area[ind1] @ (phi[ind1, :] * fun1(ind1)))
        if len(ind2) > 0:
            psurf -= np.imag(p.area[ind2] @ (phi[ind2, :] * fun2(ind2)))

        # MATLAB: loss.m lines 47-61
        # Loop over materials
        unique_mats = np.unique(p.inout.ravel())
        for mat in unique_mats:
            if mat == 1:
                continue
            mat_int = int(mat)

            # MATLAB: ind1 = p.index(find(p.inout(:,1)==mat)')
            ind1_mat = []
            ind2_mat = []
            for ip in range(p.np):
                face_indices = getattr(p, "index_func", p.index)(ip + 1)
                if p.inout[ip, 0] == mat_int:
                    ind1_mat.extend(face_indices)
                if p.inout[ip, 1] == mat_int:
                    ind2_mat.extend(face_indices)
            ind1_mat = np.array(ind1_mat, dtype = int) if len(ind1_mat) > 0 else np.array([], dtype = int)
            ind2_mat = np.array(ind2_mat, dtype = int) if len(ind2_mat) > 0 else np.array([], dtype = int)

            # MATLAB: phi = vel*full(obj, potinside(obj,-q,k(mat),find(any(p.inout==mat,2))',mat))
            # Pass particle indices (1-indexed) as mask, not face indices
            mask_particles = np.where(np.any(p.inout == mat_int, axis = 1))[0] + 1  # 1-indexed

            phi_inside_mat, _ = self.potinside(
                -q, k_arr[mat_int - 1],
                mask_particles,
                mat_int)
            phi_mat = self.vel * self.full(phi_inside_mat)

            if len(ind1_mat) > 0:
                psurf -= np.imag(p.area[ind1_mat] @ (phi_mat[ind1_mat, :] * fun1(ind1_mat)))
            if len(ind2_mat) > 0:
                psurf -= np.imag(p.area[ind2_mat] @ (phi_mat[ind2_mat, :] * fun2(ind2_mat)))

        # MATLAB: loss.m lines 64-66
        psurf = FINE ** 2 / (BOHR * HARTREE * np.pi * self.vel) * psurf
        pbulk = self.bulkloss(sig.enei)

        return psurf, pbulk

    def bulkloss(self,
            enei: float) -> np.ndarray:
        """
        EELS bulk loss probability in (1/eV).

        See Garcia de Abajo, RMP 82, 209 (2010), Eq. (18).

        MATLAB: @eelsret/bulkloss.m

        Parameters
        ----------
        enei : float
            Wavelength of light in vacuum (nm)

        Returns
        -------
        pbulk : ndarray, shape (n_impact,)
            Loss probability from bulk material
        """
        # MATLAB: bulkloss.m lines 14-27
        ene = EV2NM / enei
        mass = 0.51e6  # rest mass of electron in eV

        eps_vals = np.array([eps_func(enei) for eps_func in self.p.eps], dtype = object)
        eps_arr = np.array([e[0] if hasattr(e, '__len__') else e for e in eps_vals], dtype = complex)

        q = 2 * np.pi / (enei * self.vel)
        qc = q * np.sqrt((mass / ene) ** 2 * self.vel ** 2 * self.phiout ** 2 + 1)
        k_arr = 2 * np.pi / enei * np.sqrt(eps_arr)

        # MATLAB: bulkloss.m lines 30-33
        # Bulk losses [Eq. (18)]
        # MATLAB's log of negative real returns +iπ; numpy gives -iπ for
        # values carrying a -0j imaginary part (e.g. positive/negative
        # division). Force MATLAB's branch by flipping -0j to +0j imag.
        ratio = (qc ** 2 - k_arr ** 2) / (q ** 2 - k_arr ** 2)
        ratio = np.where(ratio.imag == 0, ratio.real + 0.0j, ratio)
        pbulk = (FINE ** 2 / (BOHR * HARTREE * np.pi * self.vel ** 2)
                 * np.imag((self.vel ** 2 - 1.0 / eps_arr) * np.log(ratio))
                 @ self.path())

        return pbulk

    def rad(self,
            sig: object) -> Tuple[np.ndarray, object]:
        """
        Photon loss probability.

        MATLAB: @eelsret/rad.m

        Parameters
        ----------
        sig : CompStruct
            Surface charge from BEMRet

        Returns
        -------
        prad : ndarray
            Photon loss probability (RMP Sec. IV.2.B)
        dprad : CompStruct
            Differential cross section
        """
        # MATLAB: rad.m lines 15-16
        field = self.spec.farfield(sig)
        sca, dsca = self.spec.scattering(sig)

        # MATLAB: rad.m lines 20-22
        _, k = sig.p.eps[self.spec.medium - 1](sig.enei)

        # MATLAB: rad.m lines 25-26
        prad = FINE ** 2 / (2 * np.pi ** 2 * HARTREE * BOHR * k) * sca
        dprad_val = FINE ** 2 / (2 * np.pi ** 2 * HARTREE * BOHR * k)

        if hasattr(dsca, 'dsca'):
            dprad = CompStruct(self.spec.pinfty, field.enei if hasattr(field, 'enei') else sig.enei,
                               dprad = dprad_val * dsca.dsca)
        else:
            dprad = CompStruct(self.spec.pinfty, sig.enei, dprad = dprad_val * dsca)

        return prad, dprad

    def field(self,
            p: object,
            enei: float,
            inout: int = 1) -> CompStruct:
        """
        Electromagnetic fields for EELS excitation (infinite beam).

        MATLAB: @eelsret/field.m

        Parameters
        ----------
        p : ComParticle or Particle
            Points or particle surface where field is computed
        enei : float
            Light wavelength in vacuum (nm)
        inout : int, optional
            Compute field at inside (1) or outside (2) of p (default: 1)

        Returns
        -------
        exc : CompStruct
            CompStruct object containing electromagnetic fields 'e' and 'h'
        """
        # MATLAB: field.m lines 18-28
        n_imp = self.impact.shape[0]

        if len(self._indimp) > 0:
            ind = np.zeros((n_imp, len(self.p.eps)), dtype = int)
            for k_idx in range(len(self._indimp)):
                ind[self._indimp[k_idx], self._indmat[k_idx] - 1] = 1
        else:
            ind = np.zeros((n_imp, len(self.p.eps)), dtype = int)
        # All beams propagate through embedding medium
        ind[:, 0] = 1

        # MATLAB: field.m lines 30-33
        exc = CompStruct(p, enei)
        exc.e = np.zeros((p.n, 3, n_imp), dtype = complex)
        exc.h = np.zeros((p.n, 3, n_imp), dtype = complex)

        # MATLAB: field.m lines 36-50
        eps_vals = []
        k_vals = []
        for eps_func in p.eps:
            eps, k = eps_func(enei)
            eps_vals.append(eps)
            k_vals.append(k)

        for ip in range(p.np):
            imat = p.inout[ip, inout - 1] - 1  # 0-indexed
            ind1 = np.array(getattr(p, "index_func", p.index)(ip + 1))
            ind2 = np.where(ind[:, imat] != 0)[0]

            if len(ind1) > 0 and len(ind2) > 0:
                e_part, h_part = self._fieldinfty(
                    p.pos[ind1, :], self.impact[ind2, :],
                    k_vals[imat], eps_vals[imat], self.vel, self.width)
                exc.e[np.ix_(ind1, [0, 1, 2], ind2)] = e_part
                exc.h[np.ix_(ind1, [0, 1, 2], ind2)] = h_part

        return exc

    @staticmethod
    def _fieldinfty(pos: np.ndarray,
            b: np.ndarray,
            k: complex,
            eps: complex,
            vel: float,
            width: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fields for infinite electron beam (retarded).

        See Garcia de Abajo, RMP 82, 209 (2010), Eqs. (5, 6).

        MATLAB: @eelsret/field.m -> fieldinfty subfunction

        Parameters
        ----------
        pos : ndarray, shape (n1, 3)
            Observation positions
        b : ndarray, shape (n2, 2)
            Impact parameters
        k : complex
            Wavenumber of light
        eps : complex
            Dielectric function
        vel : float
            Electron velocity / c
        width : float
            Beam width

        Returns
        -------
        e : ndarray, shape (n1, 3, n2)
            Electric field
        h : ndarray, shape (n1, 3, n2)
            Magnetic field
        """
        # MATLAB: field.m lines 57-89
        n1 = pos.shape[0]
        n2 = b.shape[0]

        x = pos[:, 0:1] - b[:, 0:1].T
        y = pos[:, 1:2] - b[:, 1:2].T
        z = np.tile(pos[:, 2:3], (1, n2))

        r = np.sqrt(x ** 2 + y ** 2 + width ** 2)
        x_hat = x / r
        y_hat = y / r

        e = np.zeros((n1, 3, n2), dtype = complex)
        h = np.zeros((n1, 3, n2), dtype = complex)

        # Wavenumber of electron
        q = k / (vel * np.sqrt(eps))
        # Lorentz contraction factor
        gamma = 1.0 / np.sqrt(1 - eps * vel ** 2)

        K0 = besselk(0, q * r / gamma)
        K1 = besselk(1, q * r / gamma)
        fac = 2 * q / (vel * gamma) * np.exp(1j * q * z)

        # Electric field
        e[:, 0, :] = -fac / eps * K1 * x_hat
        e[:, 1, :] = -fac / eps * K1 * y_hat
        e[:, 2, :] = fac / eps * K0 * 1j / gamma

        # Magnetic field
        h[:, 0, :] = vel * fac * K1 * y_hat
        h[:, 1, :] = -vel * fac * K1 * x_hat

        return e, h

    def __call__(self,
            p: object,
            enei: float) -> CompStruct:
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
        """
        return self.potential(p, enei)

    def __repr__(self) -> str:
        return 'EELSRet(n_impact={}, vel={:.4f})'.format(
            self.impact.shape[0], self.vel)

    def __str__(self) -> str:
        return ('EELS Excitation (Retarded):\n'
                '  Impact parameters: {}\n'
                '  Velocity: {:.4f} c\n'
                '  Width: {}').format(
            self.impact.shape[0], self.vel, self.width)
