import os
import sys

from typing import List, Dict, Tuple, Optional, Union, Any, Callable

import numpy as np

from .spectrum_ret import trisphere_unit, _PinftyStruct, _load_pinfty_default
from ..greenfun import CompStruct


def trispheresegment_unit(n_phi: int = 21, n_theta: int = 21) -> Tuple[np.ndarray, np.ndarray]:
    """MATLAB trispheresegment(linspace(0,2*pi,n_phi), linspace(0,pi,n_theta), 2).

    Creates a (n_phi-1)*(n_theta-1) quadrilateral lat-lon mesh on unit sphere.
    Crucially includes theta=pi/2 equator line — needed for hemispheric split
    in SpectrumStatLayer.

    Returns
    -------
    nvec : (n_faces, 3) face centroids on unit sphere
    area : (n_faces,) solid angles, sum to exactly 4*pi
    """
    phi = np.linspace(0, 2 * np.pi, n_phi)
    theta = np.linspace(0, np.pi, n_theta)

    # Quadrilateral faces: (n_theta-1) rows x (n_phi-1) cols
    n_faces = (n_theta - 1) * (n_phi - 1)
    nvec = np.zeros((n_faces, 3))
    area = np.zeros(n_faces)

    idx = 0
    for i in range(n_theta - 1):
        # solid angle for this theta band
        d_omega_band = (np.cos(theta[i]) - np.cos(theta[i + 1]))  # positive
        dphi = (phi[1] - phi[0])
        for j in range(n_phi - 1):
            # 4 vertices of this quad
            t_mid = 0.5 * (theta[i] + theta[i + 1])
            p_mid = 0.5 * (phi[j] + phi[j + 1])
            # Face centroid on unit sphere
            nvec[idx, 0] = np.sin(t_mid) * np.cos(p_mid)
            nvec[idx, 1] = np.sin(t_mid) * np.sin(p_mid)
            nvec[idx, 2] = np.cos(t_mid)
            # Solid angle
            area[idx] = dphi * d_omega_band
            idx += 1

    return nvec, area


class SpectrumStatLayer(object):

    def __init__(self,
            pinfty: Optional[Any] = None,
            layer: Optional[Any] = None,
            medium: Optional[int] = None) -> None:

        self.medium = medium
        self.layer = layer

        # Handle different input types
        if pinfty is None:
            # MATLAB @spectrumstatlayer/init.m uses
            # trispheresegment(linspace(0,2*pi,21), linspace(0,pi,21), 2)
            # The MATLAB particle has CURVED quad face areas (computed via
            # midpoint interpolation) that differ from analytical solid angles
            # by ~1.1%. Using solid angles gave 3.86% scattering error.
            from ..geometry import trispheresegment
            phi_grid = np.linspace(0, 2 * np.pi, 21)
            theta_grid = np.linspace(0, np.pi, 21)
            p_inf = trispheresegment(phi_grid, theta_grid, 2)
            self.pinfty = _PinftyStruct(
                p_inf.nvec.copy(), p_inf.area.copy(), pos = p_inf.pos.copy())
        elif isinstance(pinfty, int):
            _, _, nvec, area = trisphere_unit(pinfty)
            self.pinfty = _PinftyStruct(nvec, area)
        elif isinstance(pinfty, np.ndarray):
            nvec = np.atleast_2d(pinfty)
            area = np.full(nvec.shape[0], 4 * np.pi / nvec.shape[0])
            self.pinfty = _PinftyStruct(nvec, area)
        elif hasattr(pinfty, 'nvec') and hasattr(pinfty, 'area'):
            self.pinfty = pinfty
        else:
            _, _, nvec, area = trisphere_unit(256)
            self.pinfty = _PinftyStruct(nvec, area)

        self.nvec = self.pinfty.nvec if hasattr(self.pinfty, 'nvec') else self.pinfty['nvec']
        self.area = self.pinfty.area if hasattr(self.pinfty, 'area') else self.pinfty['area']
        # MATLAB scattering.m uses pinfty.pos for the hemisphere split.
        # pos is the actual face centroid (radius ~2 for trispheresegment),
        # which can differ in sign from nvec at faces straddling the equator.
        self.pos = self.pinfty.pos if hasattr(self.pinfty, 'pos') and self.pinfty.pos is not None else self.nvec
        self.ndir = len(self.nvec)


        # Separate into upper and lower hemisphere
        self._init_hemispheres()

    def _init_hemispheres(self) -> None:

        # MATLAB: spectrumstatlayer/init.m
        # Upper hemisphere (z > 0) -> medium above layer
        # Lower hemisphere (z < 0) -> medium below layer
        # MATLAB uses pinfty.pos (face centroid position), NOT nvec.
        z = self.pos[:, 2]
        self.ind_up = np.where(z > 0)[0]
        self.ind_down = np.where(z < 0)[0]

        self.nvec_up = self.nvec[self.ind_up]
        self.area_up = self.area[self.ind_up]
        self.nvec_down = self.nvec[self.ind_down]
        self.area_down = self.area[self.ind_down]

    def efarfield(self,
            dip_or_sig: Any,
            enei: Optional[float] = None,
            dir: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:

        # MATLAB: spectrumstatlayer/efarfield.m
        # Compute electric far-fields using Novotny & Hecht Eqs. 10.31-38
        # with Fresnel coefficients for upper and lower hemispheres

        # Determine if first arg is a CompStruct (sig) or a dipole array
        if hasattr(dip_or_sig, 'sig') or (isinstance(dip_or_sig, dict) and 'sig' in dip_or_sig):
            # Called with sig (CompStruct)
            sig = dip_or_sig
            if enei is None:
                enei = sig.enei if hasattr(sig, 'enei') else sig['enei']

            p = sig.p if hasattr(sig, 'p') else sig['p']
            surface_charge = sig.sig if hasattr(sig, 'sig') else sig['sig']

            if surface_charge.ndim == 1:
                surface_charge = surface_charge[:, np.newaxis]

            pos = p.pos
            area = p.area

            # Induced dipole moment
            weighted_pos = area[:, np.newaxis] * pos  # (nfaces, 3)
            dip = weighted_pos.T @ surface_charge  # (3, npol)
        else:
            # Called with dip array directly (like MATLAB)
            dip = np.asarray(dip_or_sig, dtype = complex)
            if dip.ndim == 1:
                dip = dip.reshape(3, 1)
            elif dip.shape[0] != 3:
                # dip is (npol, 3) -> transpose to (3, npol)
                dip = dip.T

        npol = dip.shape[1]

        # Directions to evaluate far-field
        if dir is None:
            directions = self.nvec
            ndir = self.ndir
        else:
            directions = np.atleast_2d(dir)
            ndir = directions.shape[0]

        layer = self.layer
        if layer is None:
            return self._farfield_free(dip, enei, npol)

        # Get dielectric functions and wavenumbers
        eps1, k1 = layer.eps[0](enei)
        eps2, k2 = layer.eps[1](enei)

        k0 = 2 * np.pi / enei

        # Allocate field array
        field = np.zeros((ndir, 3, npol), dtype = complex)
        # Wavenumber array for each direction
        k_arr = np.zeros(ndir, dtype = complex)

        # Dipole components (each is shape (npol,))
        dip1 = dip[0, :]  # x-component
        dip2 = dip[1, :]  # y-component
        dip3 = dip[2, :]  # z-component

        # --- Upper hemisphere (z > 0, MATLAB strict) ---
        ind_up = np.where(
            (directions[:, 2] > 0)
            & ~np.any(np.abs(np.imag(directions)) > 1e-10, axis = 1)
        )[0]

        if np.abs(np.imag(k1)) < 1e-10 and len(ind_up) > 0:
            d_up = directions[ind_up]  # (n_up, 3)
            k_arr[ind_up] = k1

            # Spherical coordinates (MATLAB cart2sph convention + correction)
            phi_s = np.arctan2(d_up[:, 1], d_up[:, 0])
            theta_s = np.pi / 2 - np.arctan2(d_up[:, 2],
                np.sqrt(d_up[:, 0] ** 2 + d_up[:, 1] ** 2))

            sinp = np.sin(phi_s)
            cosp = np.cos(phi_s)
            sint = np.sin(theta_s)
            cost = np.cos(theta_s)

            # z-components of wavevectors
            k1z = k1 * d_up[:, 2]  # (n_up,)
            k2z = np.sqrt(k2 ** 2 - k1 ** 2 + k1z ** 2 + 0j)

            # Fresnel reflection coefficients, Novotny & Hecht Eq. (2.49)
            rte = (k1z - k2z) / (k1z + k2z)
            rtm = (eps2 * k1z - eps1 * k2z) / (eps2 * k1z + eps1 * k2z)

            # Quasistatic coefficients, Eqs. (10.33-35)
            c_phi1 = 1 + rtm  # for dip_z
            c_phi2 = 1 - rtm  # for dip_xy theta
            c_phi3 = 1 + rte  # for TE (phi) component

            # Electric field components for each polarization
            for ipol in range(npol):
                # etheta and ephi (vectorized over directions)
                etheta = ((cosp * dip1[ipol] + sinp * dip2[ipol]) * cost * c_phi2
                          - sint * c_phi1 * dip3[ipol])
                ephi = -(sinp * dip1[ipol] - cosp * dip2[ipol]) * c_phi3

                # Convert to Cartesian
                field[ind_up, 0, ipol] = etheta * cost * cosp - ephi * sinp
                field[ind_up, 1, ipol] = etheta * cost * sinp + ephi * cosp
                field[ind_up, 2, ipol] = -etheta * sint
        else:
            k_arr[ind_up] = k1

        # --- Lower hemisphere (z < 0) ---
        ind_down = np.where(
            (directions[:, 2] < 0)
            & ~np.any(np.abs(np.imag(directions)) > 1e-10, axis = 1)
        )[0]

        if np.abs(np.imag(k2)) < 1e-10 and len(ind_down) > 0:
            d_down = directions[ind_down]  # (n_down, 3)
            k_arr[ind_down] = k2

            # Spherical coordinates
            phi_s = np.arctan2(d_down[:, 1], d_down[:, 0])
            theta_s = np.pi / 2 - np.arctan2(d_down[:, 2],
                np.sqrt(d_down[:, 0] ** 2 + d_down[:, 1] ** 2))

            sinp = np.sin(phi_s)
            cosp = np.cos(phi_s)
            sint = np.sin(theta_s)
            cost = np.cos(theta_s)

            # z-components of wavevectors (Novotny & Hecht convention)
            k2z = -k2 * d_down[:, 2]  # positive (since dir_z < 0)
            k1z = np.sqrt(k1 ** 2 - k2 ** 2 + k2z ** 2 + 0j) + 1e-10j
            k1z = k1z * np.sign(np.imag(k1z))

            # Novotny & Hecht Eq. (10.31)
            stilde = k1z / k2

            # Fresnel transmission coefficients, Eq. (2.50)
            tte = 2 * k1z / (k1z + k2z)
            ttm = (2 * eps2 * k1z / (eps2 * k1z + eps1 * k2z)
                   * np.sqrt(eps1 / eps2))

            # Quasistatic coefficients, Eqs. (10.36-38)
            c_phi1 = np.sqrt(eps2 / eps1) * cost / stilde * ttm
            c_phi2 = -np.sqrt(eps2 / eps1) * ttm
            c_phi3 = cost / stilde * tte

            # Sign correction (see MATLAB comment about sign error)
            c_phi1 = -c_phi1
            c_phi2 = -c_phi2
            c_phi3 = -c_phi3

            # Electric field components
            for ipol in range(npol):
                etheta = ((cosp * dip1[ipol] + sinp * dip2[ipol]) * cost * c_phi2
                          - sint * c_phi1 * dip3[ipol])
                ephi = -(sinp * dip1[ipol] - cosp * dip2[ipol]) * c_phi3

                # Convert to Cartesian
                field[ind_down, 0, ipol] = etheta * cost * cosp - ephi * sinp
                field[ind_down, 1, ipol] = etheta * cost * sinp + ephi * cosp
                field[ind_down, 2, ipol] = -etheta * sint
        else:
            k_arr[ind_down] = k2

        # Prefactor for electric field, Eq. (10.32)
        # Note: MATLAB neglects 1/eps1 factor intentionally
        field = k1 ** 2 * field

        return field, k_arr

    def _farfield_free(self,
            dip: np.ndarray,
            enei: float,
            npol: int) -> Tuple[np.ndarray, np.ndarray]:

        layer = self.layer
        if layer is not None:
            eps_val, k = layer.eps[0](enei)
        else:
            k = 2 * np.pi / enei
            eps_val = 1.0

        e = np.zeros((self.ndir, 3, npol), dtype = complex)

        for ipol in range(npol):
            dip_i = dip[:, ipol]
            dir_expanded = self.nvec
            dip_expanded = np.tile(dip_i, (self.ndir, 1))

            cross1 = np.cross(dir_expanded, dip_expanded)
            e[:, :, ipol] = k ** 2 * np.cross(cross1, dir_expanded) / eps_val

        k_arr = np.full(self.ndir, k, dtype = complex)
        return e, k_arr

    def farfield(self,
            sig: Any,
            direction: Optional[np.ndarray] = None) -> CompStruct:

        e_total, k_arr = self.efarfield(sig)
        enei = sig.enei if hasattr(sig, 'enei') else sig['enei']

        npol = e_total.shape[2]
        if npol == 1:
            e_total = e_total[:, :, 0]

        # MATLAB @spectrumstatlayer/farfield.m:
        #   nb = sqrt( k / ( 2 * pi / sig.enei ) );
        #   field.h = matmul( diag( nb ), matcross( dir, field.e ) );
        # k_arr is per-direction (k1 upper, k2 lower); nb = eps^(1/4).
        k0 = 2 * np.pi / enei
        nb = np.sqrt(k_arr / k0)  # (ndir,), complex

        if e_total.ndim == 2:
            h = nb[:, np.newaxis] * np.cross(self.nvec, e_total)
        else:
            h = np.zeros_like(e_total)
            for ipol in range(npol):
                h[:, :, ipol] = nb[:, np.newaxis] * np.cross(self.nvec, e_total[:, :, ipol])

        # For CompStruct metadata: k scalar for upper medium (MATLAB legacy).
        if self.layer is not None:
            _, k = self.layer.eps[0](enei)
        else:
            k = k0
        field = CompStruct(self.pinfty, enei, e = e_total, h = h,
            nvec = self.nvec, area = self.area, k = k)
        return field

    def scattering(self,
            dip_or_sig: Any,
            enei: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:

        # MATLAB: spectrumstatlayer/scattering.m
        # Integrate |E|^2 weighted by 0.5*k/k0 over the sphere

        # Determine if called with sig or dip
        if hasattr(dip_or_sig, 'sig') or (isinstance(dip_or_sig, dict) and 'sig' in dip_or_sig):
            sig = dip_or_sig
            if enei is None:
                enei = sig.enei if hasattr(sig, 'enei') else sig['enei']
            field, k_arr = self.efarfield(sig, enei)
        else:
            # Called with dip directly
            dip = dip_or_sig
            field, k_arr = self.efarfield(dip, enei)

        k0 = 2 * np.pi / enei
        npol = field.shape[2]

        # Differential radiated power, Jackson Eq. (9.22)
        dsca = np.sum(np.abs(field) ** 2, axis = 1)  # (ndir, npol)

        # Multiply by 0.5 * k/k0 (MATLAB: 0.5 * k(:) / k0)
        k_real = np.real(k_arr)
        weight = 0.5 * k_real / k0  # (ndir,)
        dsca = dsca * weight[:, np.newaxis]

        # Damp fields in media with complex dielectric functions
        dsca[np.imag(dsca) != 0] = 0
        dsca = np.real(dsca)

        # Select indices for integration based on medium
        if self.medium is None or not hasattr(self, 'medium') or self.medium is None:
            # Use all directions
            ind = np.arange(self.ndir)
        else:
            if self.layer is not None:
                # MATLAB: uses pinfty.pos[:,3], not nvec[:,3]
                if self.medium == self.layer.ind[0]:
                    ind = np.where(self.pos[:, 2] > 0)[0]
                elif self.medium == self.layer.ind[-1]:
                    ind = np.where(self.pos[:, 2] < 0)[0]
                else:
                    ind = np.arange(self.ndir)
            else:
                ind = np.arange(self.ndir)

        # Total cross section: integrate area * dsca over selected indices
        sca = np.dot(self.area[ind], dsca[ind])

        if npol == 1:
            sca = sca[0]
            dsca = dsca[:, 0]

        return sca, dsca

    def __repr__(self) -> str:
        return 'SpectrumStatLayer(ndir={}, medium={})'.format(self.ndir, self.medium)
