"""
Static EELS excitation in quasistatic approximation.

Excitation of an electron beam with high kinetic energy. Given an electron
beam, EELSStat computes the external potentials needed for BEM simulations
in the quasistatic limit and determines the energy loss probability.

Reference:
    Garcia de Abajo et al., PRB 65, 115418 (2002), RMP 82, 209 (2010).
    MATLAB MNPBEM Simulation/static/@eelsstat

Matches MATLAB MNPBEM implementation exactly.
"""

import numpy as np
from scipy.special import kv as besselk
from typing import Optional, Tuple

from ..greenfun import CompStruct
from ..misc import EV2NM, BOHR, HARTREE, FINE
from ..utils.matlab_compat import msqrt, mlog
from .eels_base import EELSBase


class EELSStat(EELSBase):
    """
    Electron energy loss spectroscopy in quasistatic approximation.

    Given an electron beam, computes the external potentials needed for
    quasistatic BEM simulations and determines the EELS loss probability.

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

    Notes
    -----
    MATLAB: Simulation/static/@eelsstat/

    The surface loss probability is computed from:
        p_surf = -(alpha^2 / (a0 * Eh * pi)) * Im(area' * (conj(phi) * sig))

    where phi is the potential of the infinite electron beam.
    """

    # Class constants
    # MATLAB: @eelsstat line 9-11
    name = 'eels'
    needs = {'sim': 'stat'}

    def __init__(self,
            p: object,
            impact: np.ndarray,
            width: float,
            vel: float,
            cutoff: Optional[float] = None,
            phiout: float = 1e-2,
            **options) -> None:
        """
        Initialize static EELS excitation.

        MATLAB: eelsstat.m constructor

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
        """
        super(EELSStat, self).__init__(
            p, impact, width, vel,
            cutoff = cutoff, phiout = phiout, **options)

    def potential(self,
            p: object,
            enei: float) -> CompStruct:
        """
        Potential of electron beam excitation for use in BEMStat.

        MATLAB: @eelsstat/potential.m

        Parameters
        ----------
        p : ComParticle
            Particle surface where potential is computed
        enei : float
            Light wavelength in vacuum (nm)

        Returns
        -------
        exc : CompStruct
            CompStruct object containing scalar potentials phi, phip
        """
        # MATLAB: potential.m line 15
        q = 2 * np.pi / (enei * self.vel)

        # MATLAB: potential.m lines 18-20
        phi, phip = self.potinfty(q, 1.0)
        pin, pinp = self.potinside(q, 0.0)

        # MATLAB: potential.m lines 23-25
        eps_vals = np.array([eps_func(enei)[0] if callable(eps_func) else eps_func(enei)
                             for eps_func in self.p.eps])
        if hasattr(eps_vals[0], '__len__'):
            eps_vals = np.array([e[0] if hasattr(e, '__len__') else e for e in eps_vals])

        # Difference of inverse dielectric functions
        # MATLAB: ideps = 1./eps - 1/eps(1)
        ideps = 1.0 / eps_vals - 1.0 / eps_vals[0]

        # MATLAB: potential.m lines 28-29
        # Add potential from beam inside of particle
        if pin.shape[1] > 0 and len(self._indmat) > 0:
            ideps_mat = ideps[self._indmat - 1]
            phi = phi / eps_vals[0] + self.full(pin * ideps_mat[np.newaxis, :])
            phip = phip / eps_vals[0] + self.full(pinp * ideps_mat[np.newaxis, :])
        else:
            phi = phi / eps_vals[0]
            phip = phip / eps_vals[0]

        # MATLAB: potential.m line 32
        return CompStruct(self.p, enei, phi = phi, phip = phip)

    def loss(self,
            sig: object) -> Tuple[np.ndarray, np.ndarray]:
        """
        EELS loss probability in (1/eV).

        MATLAB: @eelsstat/loss.m

        Parameters
        ----------
        sig : CompStruct
            Surface charge from BEMStat

        Returns
        -------
        psurf : ndarray, shape (n_impact,)
            EELS loss probability from surface plasmons
        pbulk : ndarray, shape (n_impact,)
            Loss probability from bulk material
        """
        # MATLAB: loss.m lines 15-17
        q = 2 * np.pi / (sig.enei * self.vel)
        phi, _ = self.potinfty(q, 1.0)

        # MATLAB: loss.m lines 20-23
        # Surface plasmon loss [Eq. (18)].
        # A5 fix: materialize cupy sig on host so numpy matmul does not raise.
        _sig_raw = sig.sig
        sig_arr = (_sig_raw.get() if (hasattr(_sig_raw, 'get')
            and not isinstance(_sig_raw, np.ndarray)) else np.asarray(_sig_raw))
        psurf = (-FINE ** 2 / (BOHR * HARTREE * np.pi)
                 * np.imag(self.p.area @ (np.conj(phi) * sig_arr)))

        # MATLAB: loss.m line 25
        pbulk = self.bulkloss(sig.enei)

        return psurf, pbulk

    def bulkloss(self,
            enei: float) -> np.ndarray:
        """
        EELS bulk loss probability in (1/eV).

        See Garcia de Abajo, RMP 82, 209 (2010), Eq. (19).

        MATLAB: @eelsstat/bulkloss.m

        Parameters
        ----------
        enei : float
            Wavelength of light in vacuum (nm)

        Returns
        -------
        pbulk : ndarray, shape (n_impact,)
            Loss probability from bulk material
        """
        # MATLAB: bulkloss.m lines 14-18
        ene = EV2NM / enei
        mass = 0.51e6  # rest mass of electron in eV

        # MATLAB: bulkloss.m lines 20-25
        eps_vals = np.array([eps_func(enei)[0] if callable(eps_func) else eps_func(enei)
                             for eps_func in self.p.eps])
        if hasattr(eps_vals[0], '__len__'):
            eps_vals = np.array([e[0] if hasattr(e, '__len__') else e for e in eps_vals])

        # Bulk losses [Eq. (17)]
        pbulk = (2 * FINE ** 2 / (BOHR * HARTREE * np.pi * self.vel ** 2)
                 * np.imag(-1.0 / eps_vals) @ self.path()
                 * mlog(msqrt((mass / ene) ** 2 * self.vel ** 2 * self.phiout ** 2 + 1)))

        return pbulk

    def field(self,
            p: object,
            enei: float,
            inout: int = 1) -> CompStruct:
        """
        Electromagnetic fields for EELS excitation.

        MATLAB: @eelsstat/field.m

        Parameters
        ----------
        p : ComParticle or Particle
            Points or particle surface where field is computed
        enei : float
            Light wavelength in vacuum (nm)
        inout : int, optional
            Not used in quasistatic limit

        Returns
        -------
        exc : CompStruct
            CompStruct object containing electric field 'e'
        """
        # MATLAB: field.m lines 14-19
        eps_vals = []
        k_vals = []
        for eps_func in p.eps:
            eps, k = eps_func(enei)
            eps_vals.append(eps)
            k_vals.append(k)
        eps_vals = np.array(eps_vals)
        k_vals = np.array(k_vals)

        ideps = 1.0 / eps_vals - 1.0 / eps_vals[0]
        q = k_vals[0] / (self.vel * np.sqrt(eps_vals[0]))

        # External excitation
        exc = CompStruct(p, enei)
        exc.e = (self._fieldinfty(p.pos, q, k_vals[0], eps_vals[0])
                 + self._fieldinside(p.pos, q, ideps))

        return exc

    def _fieldinfty(self,
            pos: np.ndarray,
            q: float,
            k: float,
            eps: complex) -> np.ndarray:
        """
        Fields for infinite electron beam (quasistatic).

        MATLAB: @eelsstat/field.m -> fieldinfty subfunction

        Parameters
        ----------
        pos : ndarray, shape (n, 3)
            Observation positions
        q : float
            Electron wavenumber
        k : float
            Light wavenumber
        eps : complex
            Dielectric function

        Returns
        -------
        e : ndarray, shape (n, 3, nimp)
            Electric field
        """
        b = self.impact
        vel = self.vel
        n_pos = pos.shape[0]
        n_imp = b.shape[0]

        # MATLAB: field.m lines 37-43
        x = pos[:, 0:1] - b[:, 0:1].T  # (n_pos, n_imp)
        y = pos[:, 1:2] - b[:, 1:2].T
        z = np.tile(pos[:, 2:3], (1, n_imp))

        rr = msqrt(x ** 2 + y ** 2 + self.width ** 2)
        x_hat = x / rr
        y_hat = y / rr

        # MATLAB: field.m lines 48-51
        K0 = besselk(0, q * rr)
        K1 = besselk(1, q * rr)
        fac = 2 * q / vel * np.exp(1j * q * z)

        # MATLAB: field.m lines 53-58
        e = np.zeros((n_pos, 3, n_imp), dtype = complex)
        e[:, 0, :] = -fac / eps * K1 * x_hat
        e[:, 1, :] = -fac / eps * K1 * y_hat
        e[:, 2, :] = 1j * fac / eps * K0

        return e

    def _fieldinside(self,
            pos: np.ndarray,
            q: float,
            ideps: np.ndarray) -> np.ndarray:
        """
        Compute field of electron trajectories inside of particle.

        MATLAB: @eelsstat/field.m -> fieldinside subfunction

        Parameters
        ----------
        pos : ndarray, shape (n, 3)
            Observation positions
        q : float
            Electron wavenumber
        ideps : ndarray
            Difference of inverse dielectric functions

        Returns
        -------
        e : ndarray, shape (n, 3, nimp)
            Electric field
        """
        b = self.impact
        vel = self.vel
        n_pos = pos.shape[0]
        n_imp = b.shape[0]

        e = np.zeros((n_pos, 3, n_imp), dtype = complex)

        if len(self._indmat) == 0:
            return e

        # MATLAB: field.m lines 65-73
        x = pos[:, 0:1] - b[:, 0:1].T
        y = pos[:, 1:2] - b[:, 1:2].T
        z = np.tile(pos[:, 2:3], (1, n_imp))

        r = msqrt(x ** 2 + y ** 2)
        rr = msqrt(r ** 2 + self.width ** 2)

        # MATLAB: field.m lines 76-78
        _, Ir, Iz = self._potwire(rr, z, q, 0, self._z[:, 0], self._z[:, 1])
        ideps_mat = ideps[self._indmat - 1]

        # MATLAB: field.m lines 80-85
        e[:, 0, :] = (Ir * ideps_mat[np.newaxis, :]) / vel * x / rr
        e[:, 1, :] = (Ir * ideps_mat[np.newaxis, :]) / vel * y / rr
        e[:, 2, :] = (Iz * ideps_mat[np.newaxis, :]) / vel

        return e

    def __call__(self,
            p: object,
            enei: float) -> CompStruct:
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
        return self.potential(p, enei)

    def __repr__(self) -> str:
        return 'EELSStat(n_impact={}, vel={:.4f})'.format(
            self.impact.shape[0], self.vel)

    def __str__(self) -> str:
        return ('EELS Excitation (Quasistatic):\n'
                '  Impact parameters: {}\n'
                '  Velocity: {:.4f} c\n'
                '  Width: {}').format(
            self.impact.shape[0], self.vel, self.width)
