import os
import sys

from typing import List, Dict, Tuple, Optional, Union, Any, Callable

import numpy as np

from ..greenfun import CompStruct
from ..utils.matlab_compat import msqrt


class DipoleStatLayer(object):

    name = 'dipole'
    needs = {'sim': 'stat'}

    def __init__(self,
            pt: Any,
            layer: Any,
            dip: Optional[np.ndarray] = None,
            full: bool = False,
            pinfty: Optional[Any] = None,
            **options: Any) -> None:

        self.pt = pt
        self.layer = layer
        self.varargin = options

        self._init(dip, full, pinfty, **options)

    def _init(self,
            dip: Optional[np.ndarray] = None,
            full: bool = False,
            pinfty: Optional[Any] = None,
            **options: Any) -> None:

        if dip is None:
            dip = np.eye(3)
            full = False

        dip = np.asarray(dip, dtype = float)

        if full:
            if dip.ndim == 2:
                dip = dip.reshape(dip.shape + (1,))
            self.dip = dip
        else:
            if dip.ndim == 1:
                dip = dip.reshape(1, -1)

            dip_reshaped = dip.T.reshape(1, dip.shape[1], dip.shape[0])
            self.dip = np.tile(dip_reshaped, (self.pt.n, 1, 1))

        # MATLAB: spectrumstatlayer for radiative decay rate
        from ..spectrum import SpectrumStatLayer
        self.spec = SpectrumStatLayer(pinfty, layer = self.layer)

    def _image_positions(self) -> np.ndarray:

        z_layer = self.layer.z[0]
        pos = self.pt.pos.copy()
        pos_image = pos.copy()
        pos_image[:, 2] = 2 * z_layer - pos[:, 2]
        return pos_image

    def _image_factors(self,
            enei: float) -> Tuple[complex, complex]:

        # Jackson Eq. (4.45): image charge factors
        eps1, _ = self.layer.eps[0](enei)
        eps2, _ = self.layer.eps[1](enei)

        # Reflection coefficient for image dipole
        q1 = (eps1 - eps2) / (eps1 + eps2)  # parallel component
        q2 = -(eps1 - eps2) / (eps1 + eps2)  # perpendicular component

        return q1, q2

    def field2(self,
            p: Any,
            enei: float) -> CompStruct:

        # MATLAB: dipolestatlayer/field2.m
        # Electric field for mirror dipole excitation (reflected part only)

        eps1, _ = self.layer.eps[0](enei)
        eps2, _ = self.layer.eps[1](enei)
        # Image charge factors, Jackson Eq. (4.45)
        q1 = -(eps2 - eps1) / (eps2 + eps1)
        q2 = 2 * eps2 / (eps2 + eps1)

        pos = p.pos if hasattr(p, 'pos') else p.pc.pos
        z_layer = self.layer.z[0]

        # Image dipole: mirrored positions and flipped z-component
        pos_image = self._image_positions()
        dip_image = self.dip.astype(complex).copy()
        dip_image[:, 2, :] = -dip_image[:, 2, :]
        eps_at_dip = self.pt.eps1(enei)

        if np.all(pos[:, 2] > z_layer):
            # All points above substrate
            e_image = self._efield(pos, pos_image, dip_image, eps_at_dip)
            e = q1 * e_image
            return CompStruct(p, enei, e = e)

        # Points above and below substrate
        ind1 = pos[:, 2] > z_layer
        ind2 = pos[:, 2] < z_layer

        n_obs = pos.shape[0]
        n_dip = self.pt.n
        ndip = self.dip.shape[2]
        e = np.zeros((n_obs, 3, n_dip, ndip), dtype = complex)

        if np.any(ind1):
            e1 = self._efield(pos[ind1, :], pos_image, dip_image, eps_at_dip)
            e[ind1, :, :, :] = q1 * e1

        if np.any(ind2):
            e2 = self._efield(pos[ind2, :], self.pt.pos, self.dip, eps_at_dip)
            e[ind2, :, :, :] = q2 * e2

        return CompStruct(p, enei, e = e)

    def field(self,
            p: Any,
            enei: float,
            key: Optional[str] = None) -> CompStruct:

        # MATLAB: dipolestatlayer/field.m
        # Electric field from direct dipole + image dipole
        if key == 'refl':
            return self.field2(p, enei)

        pt = self.pt
        pos1 = p.pos if hasattr(p, 'pos') else p.pc.pos
        pos2 = pt.pos
        eps_at_dip = pt.eps1(enei)

        # Direct contribution
        e_direct = self._efield(pos1, pos2, self.dip, eps_at_dip)

        # Image contribution
        pos2_image = self._image_positions()
        q1, q2 = self._image_factors(enei)

        # Image dipole: parallel components multiplied by q1,
        # perpendicular (z) component multiplied by q2
        dip_image = self.dip.astype(complex).copy()
        dip_image[:, 0, :] *= q1  # x-component
        dip_image[:, 1, :] *= q1  # y-component
        dip_image[:, 2, :] *= q2  # z-component (flipped sign convention)

        e_image = self._efield(pos1, pos2_image, dip_image, eps_at_dip)

        e = e_direct + e_image

        return CompStruct(p, enei, e = e)

    def _efield(self,
            pos1: np.ndarray,
            pos2: np.ndarray,
            dip: np.ndarray,
            eps: np.ndarray) -> np.ndarray:

        n1 = pos1.shape[0]
        n2 = pos2.shape[0]
        ndip = dip.shape[2]

        e = np.zeros((n1, 3, n2, ndip), dtype = complex)

        x = pos1[:, 0:1] - pos2[:, 0].T
        y = pos1[:, 1:2] - pos2[:, 1].T
        z = pos1[:, 2:3] - pos2[:, 2].T

        r = msqrt(x ** 2 + y ** 2 + z ** 2)
        r = np.maximum(r, np.finfo(float).eps)

        x = x / r
        y = y / r
        z = z / r

        for i in range(ndip):
            dx = np.tile(dip[:, 0, i], (n1, 1))
            dy = np.tile(dip[:, 1, i], (n1, 1))
            dz = np.tile(dip[:, 2, i], (n1, 1))

            inner = x * dx + y * dy + z * dz

            e[:, 0, :, i] = (3 * x * inner - dx) / (r ** 3 * eps)
            e[:, 1, :, i] = (3 * y * inner - dy) / (r ** 3 * eps)
            e[:, 2, :, i] = (3 * z * inner - dz) / (r ** 3 * eps)

        return e

    def potential(self,
            p: Any,
            enei: float) -> CompStruct:

        exc = self.field(p, enei)
        e = exc.e

        nvec = p.nvec if hasattr(p, 'nvec') else p.pc.nvec
        phip = -np.einsum('ij,ij...->i...', nvec, e)

        # Reshape from (nfaces, npt, ndip) to (nfaces, npol)
        n = p.n if hasattr(p, 'n') else phip.shape[0]
        if phip.ndim > 2:
            phip = phip.reshape(n, -1)

        return CompStruct(p, enei, phip = phip)

    def decayrate(self,
            sig: CompStruct) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        p, enei = sig.p, sig.enei

        from ..greenfun import CompGreenStatLayer
        g = CompGreenStatLayer(self.pt, sig.p, self.layer, **self.varargin)

        # A5 fix: materialize cupy sig members on host before invoking Green
        # function so numpy matmul does not raise on a cupy operand.
        _sig_raw = sig.sig
        if hasattr(_sig_raw, 'get') and not isinstance(_sig_raw, np.ndarray):
            sig.sig = _sig_raw.get()
        field_struct = g.field(sig)
        _e_raw = field_struct.e
        e = (_e_raw.get() if (hasattr(_e_raw, 'get')
            and not isinstance(_e_raw, np.ndarray)) else np.asarray(_e_raw))

        gamma = 4 / 3 * (2 * np.pi / sig.enei) ** 3

        npt = self.pt.n
        ndip = self.dip.shape[2]

        area_pos = sig.p.pos * sig.p.area[:, np.newaxis]

        _sig_raw2 = sig.sig
        sig_arr = (_sig_raw2.get() if (hasattr(_sig_raw2, 'get')
            and not isinstance(_sig_raw2, np.ndarray)) else np.asarray(_sig_raw2))
        indip = area_pos.T @ sig_arr
        # indip shape: (3, npol) where npol = npt * ndip
        indip = indip.reshape(3, npt, ndip)
        tot = np.zeros((npt, ndip))
        rad = np.zeros((npt, ndip))
        rad0 = np.zeros((npt, ndip))

        # Reshape e from (npt, 3, npol) to (npt, 3, npt, ndip) if needed
        if e.ndim == 3:
            e = e.reshape(npt, 3, npt, ndip)
        elif e.ndim == 2:
            e = e.reshape(npt, 3, npt, ndip)

        k0 = 2 * np.pi / sig.enei

        for ipos in range(npt):
            for idip in range(ndip):
                nb = np.sqrt(self.pt.eps1(sig.enei)[ipos])

                dip = self.dip[ipos, :, idip]
                indip_i = indip[:, ipos, idip]

                # MATLAB: sca = scattering(obj.spec, indip + dip/nb^2, enei)
                sca, _ = self.spec.scattering(indip_i.reshape(1, 3) + dip.reshape(1, 3) / nb ** 2, sig.enei)
                # MATLAB: rad = (sca / (2*pi*k0)) / (0.5*nb*gamma)
                rad[ipos, idip] = (sca / (2 * np.pi * k0)) / (0.5 * nb * gamma)

                # MATLAB: tot = 1 + rad + imag(e * dip') / (0.5*nb*gamma)
                e_i = e[ipos, :, ipos, idip]
                tot[ipos, idip] = np.real(1 + rad[ipos, idip] + np.imag(e_i @ dip) / (0.5 * nb * gamma))

                rad0[ipos, idip] = np.real(nb * gamma)

        return tot, rad, rad0

    def decayrate0(self,
            enei: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        # MATLAB: dipolestatlayer/decayrate0.m
        # Total and radiative decay rate for oscillating dipole above
        # layer structure (w/o nanoparticle) in units of the free-space decay rate

        # TRUE if all dielectric functions real
        ir = all(np.isreal(eps_func(enei)[0]) for eps_func in self.layer.eps)

        # Induced electric field from image dipole at dipole positions
        e = self.field2(self.pt, enei).e

        k0 = 2 * np.pi / enei
        # Wigner-Weisskopf decay rate in free space
        gamma = 4 / 3 * k0 ** 3

        pt = self.pt
        npt = pt.n
        ndip = self.dip.shape[2]
        tot = np.zeros((npt, ndip))
        rad = np.zeros((npt, ndip))
        rad0 = np.zeros((npt, ndip))

        for ipos in range(npt):
            for idip in range(ndip):
                nb = np.sqrt(pt.eps1(enei)[ipos])
                if np.imag(nb) != 0:
                    import warnings
                    warnings.warn('Dipole embedded in medium with complex dielectric function')

                dip = self.dip[ipos, :, idip]

                # Scattering cross section from spectrum
                sca, _ = self.spec.scattering(dip / nb ** 2, enei)
                # Radiative decay rate in units of free-space decay rate
                rad[ipos, idip] = (sca / (2 * np.pi * k0)) / (0.5 * nb * gamma)
                # Free-space decay rate
                rad0[ipos, idip] = np.real(nb * gamma)

                # Total decay rate
                if ir:
                    tot[ipos, idip] = rad[ipos, idip]
                else:
                    tot[ipos, idip] = rad[ipos, idip] + np.imag(
                        np.squeeze(e[ipos, :, ipos, idip]) @ dip) / (0.5 * nb * gamma)

        return tot, rad, rad0

    def farfield(self,
            spec: Any,
            enei: float) -> CompStruct:

        dir = spec.pinfty.nvec if hasattr(spec.pinfty, 'nvec') else spec.nvec
        epstab = self.pt.eps
        eps_val, k = epstab[spec.medium - 1](enei)
        nb = np.sqrt(eps_val)

        pt = self.pt
        dip = self.dip.copy()

        screening = eps_val / pt.eps1(enei)
        dip = screening[:, np.newaxis, np.newaxis] * dip

        n1 = dir.shape[0]
        n2 = dip.shape[0]
        n3 = dip.shape[2]

        e = np.zeros((n1, 3, n2, n3), dtype = complex)
        h = np.zeros((n1, 3, n2, n3), dtype = complex)

        g = np.exp(-1j * k * (dir @ pt.pos.T))
        g = g[:, np.newaxis, :, np.newaxis]
        g = np.tile(g, (1, 3, 1, n3))

        dir_rep = dir[:, :, np.newaxis, np.newaxis]
        dir_rep = np.tile(dir_rep, (1, 1, n2, n3))

        dip_perm = dip.transpose(1, 0, 2)
        dip_rep = dip_perm[np.newaxis, :, :, :]
        dip_rep = np.tile(dip_rep, (n1, 1, 1, 1))

        h = np.cross(dir_rep, dip_rep, axis = 1) * g
        e = np.cross(h, dir_rep, axis = 1)

        e = k ** 2 * e / eps_val
        h = k ** 2 * h / nb

        field = CompStruct(spec.pinfty, enei, e = e, h = h)
        return field

    def __call__(self,
            p: Any,
            enei: float) -> CompStruct:

        return self.potential(p, enei)

    def __repr__(self) -> str:
        return 'DipoleStatLayer(npt={}, ndip={})'.format(
            self.pt.n, self.dip.shape[2])
