"""
Spectrum class for quasistatic approximation.

Matches MATLAB MNPBEM @spectrumstat implementation exactly.
"""

import numpy as np
from .spectrum_ret import trisphere_unit, _PinftyStruct


class SpectrumStat(object):
    """
    Compute far-fields and scattering cross sections in quasistatic limit.

    Uses a discretized unit sphere to compute total and differential
    scattering cross sections by integrating the Poynting vector.

    Parameters
    ----------
    pinfty : Particle or ndarray or int, optional
        - Particle: Use its nvec and area properties
        - ndarray (ndir, 3): Light propagation directions
        - int: Number of faces for unit sphere (creates trisphere)
        - None: Creates trisphere(256, 2) like MATLAB default
    medium : int, optional
        Index of embedding medium (1-indexed, default: 1)

    Attributes
    ----------
    pinfty : object
        Discretized unit sphere (stores nvec, area)
    nvec : ndarray, shape (ndir, 3)
        Directions on unit sphere (outward normals)
    area : ndarray, shape (ndir,)
        Solid angles for each direction
    medium : int
        Embedding medium index

    Notes
    -----
    MATLAB equivalent: @spectrumstat class

    In quasistatic limit, far-field computed from induced dipole moment:
        dip = sum(pos * area * sig)  (dipole moment)
        H = nb * k^2 * cross(dir, dip)  (Jackson Eq. 9.19)
        E = -cross(dir, H) / nb

    Scattering cross section from Poynting vector:
        dsca = 0.5 * real(nvec · (E × conj(H)))
        sca = sum(area * dsca)

    Examples
    --------
    >>> spec = SpectrumStat(144)
    >>> field = spec.farfield(sig)
    >>> sca, dsca = spec.scattering(sig)
    """

    def __init__(self, pinfty=None, medium=1):
        """
        Initialize spectrum calculator.

        MATLAB: spectrumstat(pinfty, op, PropertyPair)

        Parameters
        ----------
        pinfty : Particle, ndarray, or int, optional
            Unit sphere specification (see class docstring)
        medium : int
            Embedding medium index (1-indexed)
        """
        self.medium = medium

        # Handle different input types like MATLAB init.m
        if pinfty is None:
            # Default: create trisphere(256, 2)
            _, _, nvec, area = trisphere_unit(256)
            self.pinfty = _PinftyStruct(nvec, area)
        elif isinstance(pinfty, int):
            # Integer: create unit sphere with given number of faces
            _, _, nvec, area = trisphere_unit(pinfty)
            self.pinfty = _PinftyStruct(nvec, area)
        elif isinstance(pinfty, np.ndarray):
            # Numeric array: treat as direction vectors
            nvec = np.atleast_2d(pinfty)
            # For direction vectors, compute area as uniform solid angles
            area = np.full(nvec.shape[0], 4 * np.pi / nvec.shape[0])
            self.pinfty = _PinftyStruct(nvec, area)
        elif hasattr(pinfty, 'nvec') and hasattr(pinfty, 'area'):
            # Particle or struct with nvec and area properties
            self.pinfty = pinfty
        else:
            # Try to use as particle
            _, _, nvec, area = trisphere_unit(256)
            self.pinfty = _PinftyStruct(nvec, area)

        # Expose nvec and area directly for convenience
        self.nvec = self.pinfty.nvec if hasattr(self.pinfty, 'nvec') else self.pinfty['nvec']
        self.area = self.pinfty.area if hasattr(self.pinfty, 'area') else self.pinfty['area']
        self.ndir = len(self.nvec)

    def farfield(self, sig, direction=None):
        """
        Compute far-field amplitudes from surface charges.

        MATLAB: @spectrumstat/farfield.m
            dip = matmul(bsxfun(@times, sig.p.pos, sig.p.area).', sig.sig)
            field.h = nb * k^2 * cross(dir, dip, 2)  % Jackson Eq. (9.19)
            field.e = -cross(dir, field.h, 2) / nb

        Parameters
        ----------
        sig : dict
            Solution containing:
            - 'sig': surface charges, shape (nfaces,) or (nfaces, npol)
            - 'p': particle
            - 'enei': wavelength
        direction : ndarray, optional
            Light propagation directions. Defaults to self.nvec.

        Returns
        -------
        field : dict
            Far-field with 'e' and 'h' arrays, shape (ndir, 3, npol)
        """
        if direction is None:
            direction = self.nvec

        p = sig['p']
        enei = sig['enei']
        surface_charge = sig['sig']

        # Wavenumber and refractive index
        # MATLAB: [epsb, k] = p.eps{obj.medium}(sig.enei); nb = sqrt(epsb)
        eps_val, k = p.eps[self.medium - 1](enei)
        nb = np.sqrt(eps_val)

        # Get particle properties
        pos = p.pos    # (nfaces, 3)
        area = p.area  # (nfaces,)

        # Ensure 2D surface charge
        if surface_charge.ndim == 1:
            surface_charge = surface_charge[:, np.newaxis]
        npol = surface_charge.shape[1]
        ndir = len(direction)

        # Compute dipole moment: dip = (area * pos)' @ sig
        # MATLAB: dip = matmul(bsxfun(@times, sig.p.pos, sig.p.area).', sig.sig)
        # weighted_pos: (nfaces, 3), each component weighted by area
        weighted_pos = area[:, np.newaxis] * pos  # (nfaces, 3)
        dip = weighted_pos.T @ surface_charge  # (3, npol)

        # Expand direction and dipole moment for cross product
        # MATLAB: dir = repmat(reshape(dir, [], 3, 1), [1, 1, size(dip, 2)])
        #         dip = repmat(reshape(dip, 1, 3, []), [size(dir, 1), 1, 1])
        # direction: (ndir, 3) -> (ndir, 3, npol)
        # dip: (3, npol) -> (ndir, 3, npol)
        dir_expanded = np.broadcast_to(
            direction[:, :, np.newaxis], (ndir, 3, npol)
        )
        dip_expanded = np.broadcast_to(
            dip[np.newaxis, :, :], (ndir, 3, npol)
        )

        # Magnetic field: H = nb * k^2 * cross(dir, dip)
        # Jackson Eq. (9.19)
        h = nb * k**2 * np.cross(dir_expanded, dip_expanded, axis=1)

        # Electric field: E = -cross(dir, H) / nb
        e = -np.cross(dir_expanded, h, axis=1) / nb

        return {
            'e': e,
            'h': h,
            'nvec': direction,
            'area': self.area if direction is self.nvec else np.full(ndir, 4*np.pi/ndir),
            'enei': enei,
            'k': k
        }

    def scattering(self, sig):
        """
        Compute scattering cross section.

        MATLAB:
            [sca, dsca] = scattering(farfield(obj, sig))

        Uses Poynting vector integration:
            dsca = 0.5 * real(nvec · (E × conj(H)))
            sca = sum(area * dsca)

        Parameters
        ----------
        sig : dict
            Solution from BEM solver

        Returns
        -------
        sca : ndarray
            Total scattering cross section
        dsca : ndarray
            Differential scattering (per solid angle)
        """
        # Get far-field
        field = self.farfield(sig)
        e = field['e']  # (ndir, 3, npol)
        h = field['h']  # (ndir, 3, npol)

        npol = e.shape[2] if e.ndim == 3 else 1
        if e.ndim == 2:
            e = e[:, :, np.newaxis]
            h = h[:, :, np.newaxis]

        # Poynting vector component in radial direction
        # MATLAB: dsca = 0.5 * real(inner(pinfty.nvec, cross(e, conj(h), 2)))
        dsca = np.zeros((self.ndir, npol))

        for ipol in range(npol):
            # Cross product E × conj(H)
            poynting = np.cross(e[:, :, ipol], np.conj(h[:, :, ipol]))  # (ndir, 3)
            # Dot with nvec (radial direction)
            dsca[:, ipol] = 0.5 * np.real(np.sum(self.nvec * poynting, axis=1))

        # Total scattering: integrate over sphere
        # MATLAB: sca = squeeze(matmul(reshape(area, 1, []), dsca))
        sca = np.dot(self.area, dsca)  # (npol,)

        if npol == 1:
            sca = sca[0]
            dsca = dsca[:, 0]

        return sca, dsca

    def __repr__(self):
        return "SpectrumStat(ndir={}, medium={})".format(self.ndir, self.medium)
