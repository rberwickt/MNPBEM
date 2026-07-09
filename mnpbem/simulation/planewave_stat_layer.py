import os
import sys

from typing import List, Dict, Tuple, Optional, Union, Any, Callable

import numpy as np

from ..greenfun import CompStruct
from ..spectrum.spectrum_stat_layer import SpectrumStatLayer
from ..utils.matlab_compat import msqrt


class PlaneWaveStatLayer(object):

    name = 'planewave'
    needs = {'sim': 'stat'}

    def __init__(self,
            pol: np.ndarray,
            *args: Any,
            medium: int = 1,
            **options: Any) -> None:

        # Support call signatures:
        #   PlaneWaveStatLayer(pol, layer, ...)                  (legacy)
        #   PlaneWaveStatLayer(pol, dir, layer=layer, ...)       (MATLAB style)
        layer = options.pop('layer', None)
        dir = options.pop('dir', None)
        for a in args:
            if a is None:
                continue
            # A direction is a 3-vector (shape (3,) or (N,3)); anything else
            # with an `eps` attribute is treated as the layer structure.
            if hasattr(a, 'eps'):
                layer = a
                continue
            arr = np.asarray(a)
            if arr.ndim in (1, 2) and arr.shape[-1] == 3 and np.issubdtype(arr.dtype, np.number):
                dir = arr
            else:
                layer = a

        self.pol = np.asarray(pol, dtype = float)
        if self.pol.ndim == 1:
            self.pol = self.pol.reshape(1, -1)

        self.layer = layer
        self.medium = options.get('medium', medium)

        # Propagation direction. MATLAB: planewavestatlayer stores dir as
        # (npol, 3). Default to normal incidence (downward) if not supplied.
        npol = self.pol.shape[0]
        if dir is None:
            self.dir = np.zeros((npol, 3))
            self.dir[:, 2] = -1.0
        else:
            dir_arr = np.asarray(dir, dtype = float)
            if dir_arr.ndim == 1:
                dir_arr = np.tile(dir_arr.reshape(1, -1), (npol, 1))
            self.dir = dir_arr

        # Initialize spectrum object for far-field calculations
        pinfty = options.get('pinfty', None)
        if pinfty is not None:
            self.spec = SpectrumStatLayer(pinfty, layer = layer)
        else:
            self.spec = SpectrumStatLayer(layer = layer)

    def decompose(self,
            pol: Optional[np.ndarray] = None,
            dir: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        # MATLAB: planewavestatlayer/decompose.m
        # Decompose polarization into TE and TM components
        # Uses the MATLAB algorithm exactly

        if pol is None:
            pol = self.pol
        if dir is None:
            dir = self.dir

        pol = np.asarray(pol)
        dir = np.asarray(dir)

        if pol.ndim == 1:
            pol = pol.reshape(1, -1)
        if dir.ndim == 1:
            dir = dir.reshape(1, -1)

        # Extract components
        ex = pol[:, 0]
        ey = pol[:, 1]
        kx = dir[:, 0]
        ky = dir[:, 1]

        # Parallel wavenumber component
        kt = np.maximum(msqrt(kx ** 2 + ky ** 2), np.finfo(float).eps)

        # Normal vector
        nvec = np.column_stack([ky / kt, -kx / kt, np.zeros_like(kx)])

        # TE component: projection onto nvec
        te_coeff = (ky * ex - kx * ey) / kt
        te = nvec * te_coeff[:, np.newaxis]

        # TM component: remainder
        tm = pol - te

        return te, tm, dir

    def fresnel(self,
            dir_or_enei: Any = None,
            enei: Optional[float] = None) -> Any:

        # Backward-compatible Fresnel coefficient computation.
        # Old API: fresnel(dir, enei) -> (rp, rs, tp, ts)
        # New API: fresnel(enei) -> dict with re, te, rm, tm, ki, kr, kt

        if dir_or_enei is None:
            # fresnel() with no args - use defaults
            return self._fresnel_full(500.0)

        if isinstance(dir_or_enei, np.ndarray):
            # Old API: fresnel(dir, enei)
            dir_arr = np.asarray(dir_or_enei)
            if dir_arr.ndim == 1:
                dir_arr = dir_arr.reshape(1, -1)
            enei_val = float(enei)

            refl = self._fresnel_full(enei_val, dir_arr)

            # Convert to old-style (rp, rs, tp, ts) return
            # Old API: rs = TE reflection, rp = TM reflection
            # New API: re = TE reflection, rm = TM reflection
            # At normal incidence in old code:
            #   rs = (n1*cos-n2*cost)/(n1*cos+n2*cost)
            #   rp = (n2*cos-n1*cost)/(n2*cos+n1*cost)
            # In new code (Chew convention):
            #   re = (k1z-k2z)/(k1z+k2z) [TE]
            #   rm = (eps2*k1z-eps1*k2z)/(eps2*k1z+eps1*k2z) [TM]
            # These are equivalent for the old calling convention.
            return refl['rm'], refl['re'], refl['tm'], refl['te']

        # New API: fresnel(enei)
        return self._fresnel_full(float(dir_or_enei))

    def _fresnel_full(self,
            enei: float,
            dir: Optional[np.ndarray] = None) -> dict:

        # MATLAB: planewavestatlayer/fresnel.m
        # Compute Fresnel reflection and transmission coefficients
        # Returns dict with re, te, rm, tm, ki, kr, kt

        if dir is None:
            dir = self.dir

        dir = np.asarray(dir)
        if dir.ndim == 1:
            dir = dir.reshape(1, -1)

        layer = self.layer
        npol = dir.shape[0]

        # Get dielectric functions and wavenumbers
        eps1, k1 = layer.eps[0](enei)
        eps2, k2 = layer.eps[1](enei)

        re = np.zeros(npol, dtype = complex)
        te = np.zeros(npol, dtype = complex)
        rm = np.zeros(npol, dtype = complex)
        tm = np.zeros(npol, dtype = complex)
        ki = np.zeros((npol, 3), dtype = complex)
        kr = np.zeros((npol, 3), dtype = complex)
        kt = np.zeros((npol, 3), dtype = complex)

        for i in range(npol):
            if dir[i, 2] < 0:
                # Downgoing wave
                k1z = -k1 * dir[i, 2]
                k2z = np.sqrt(k2 ** 2 - k1 ** 2 + k1z ** 2 + 0j)
                # Proper sign for evanescent fields
                if np.imag(k2z) < 0:
                    k2z = np.conj(k2z)

                # Fresnel coefficients for TE modes, Eq. (2.1.13)
                re[i] = (k1z - k2z) / (k1z + k2z)
                te[i] = 2 * k1z / (k1z + k2z)
                # Fresnel coefficients for TM modes, Eq. (2.1.4)
                rm[i] = (eps2 * k1z - eps1 * k2z) / (eps2 * k1z + eps1 * k2z)
                tm[i] = 2 * eps2 * k1z / (eps2 * k1z + eps1 * k2z)

                # Wavevectors
                ki[i, :] = k1 * dir[i, :]
                kr[i, :] = np.array([ki[i, 0], ki[i, 1], k1z])
                kt[i, :] = np.array([ki[i, 0], ki[i, 1], -k2z])
            else:
                # Upgoing wave
                k2z = k2 * dir[i, 2]
                k1z = np.sqrt(k1 ** 2 - k2 ** 2 + k2z ** 2 + 0j)
                # Proper sign for evanescent fields
                if np.imag(k1z) < 0:
                    k1z = np.conj(k1z)

                # Fresnel coefficients for TE modes
                re[i] = (k2z - k1z) / (k2z + k1z)
                te[i] = 2 * k2z / (k2z + k1z)
                # Fresnel coefficients for TM modes
                rm[i] = (eps1 * k2z - eps2 * k1z) / (eps1 * k2z + eps2 * k1z)
                tm[i] = 2 * eps1 * k2z / (eps1 * k2z + eps2 * k1z)

                # Wavevectors
                ki[i, :] = k2 * dir[i, :]
                kr[i, :] = np.array([ki[i, 0], ki[i, 1], -k2z])
                kt[i, :] = np.array([ki[i, 0], ki[i, 1], k1z])

        return {'re': re, 'te': te, 'rm': rm, 'tm': tm,
                'ki': ki, 'kr': kr, 'kt': kt}

    def field(self,
            p: Any,
            enei: float) -> CompStruct:

        # MATLAB: planewavestatlayer/field.m
        # Electric field including reflected and transmitted components
        # Matches MATLAB implementation exactly

        n = p.n if hasattr(p, 'n') else p.nfaces
        npol = self.pol.shape[0]

        k0 = 2 * np.pi / enei
        layer = self.layer
        z_layer = layer.z[0]

        e = np.zeros((n, 3, npol), dtype = complex)

        # Decompose and compute Fresnel
        te_pol, tm_pol, _ = self.decompose()
        refl = self._fresnel_full(enei)

        # Magnetic field of TM mode: H_TM = ki x E_TM / k0
        tm_h = np.cross(refl['ki'], tm_pol) / k0

        pos = p.pos if hasattr(p, 'pos') else p.pc.pos

        # Index to points above and below layer
        ind1 = np.where(pos[:, 2] >= z_layer)[0]
        ind2 = np.where(pos[:, 2] < z_layer)[0]

        for i in range(npol):
            # Electric field of reflected TM mode
            kr_i = refl['kr'][i, :]
            kt_i = refl['kt'][i, :]
            er_tm = np.cross(tm_h[i, :], kr_i) * k0 / np.sum(kr_i ** 2)
            et_tm = np.cross(tm_h[i, :], kt_i) * k0 / np.sum(kt_i ** 2)

            if self.dir[i, 2] < 0:
                # Downgoing wave
                e1 = (self.pol[i, :]
                      + refl['re'][i] * te_pol[i, :]
                      + refl['rm'][i] * er_tm)
                e2 = (refl['te'][i] * te_pol[i, :]
                      + refl['tm'][i] * et_tm)
            else:
                # Upgoing wave
                e1 = (refl['te'][i] * te_pol[i, :]
                      + refl['tm'][i] * et_tm)
                e2 = (self.pol[i, :]
                      + refl['re'][i] * te_pol[i, :]
                      + refl['rm'][i] * er_tm)

            if len(ind1) > 0:
                e[ind1, :, i] = np.tile(e1, (len(ind1), 1))
            if len(ind2) > 0:
                e[ind2, :, i] = np.tile(e2, (len(ind2), 1))

        if npol == 1:
            e = e[:, :, 0]

        return CompStruct(p, enei, e = e)

    def potential(self,
            p: Any,
            enei: float) -> CompStruct:

        # MATLAB: planewavestatlayer/potential.m
        # Surface derivative of scalar potential: phip = -nvec . E

        exc = self.field(p, enei)
        e = exc.e

        nvec = p.nvec if hasattr(p, 'nvec') else p.pc.nvec

        if e.ndim == 2:
            phip = -np.sum(nvec * e, axis = 1)
        else:
            npol = e.shape[2]
            phip = np.zeros((nvec.shape[0], npol), dtype = complex)
            for ipol in range(npol):
                phip[:, ipol] = -np.sum(nvec * e[:, :, ipol], axis = 1)

        return CompStruct(p, enei, phip = phip)

    def extinction(self,
            sig: CompStruct,
            key: str = 'full') -> np.ndarray:

        # MATLAB: planewavestatlayer/extinction.m
        # Extinction cross section using Dahan & Greffet, Opt. Expr. 20, A530 (2012)

        k0 = 2 * np.pi / sig.enei

        # Dipole moment of surface charge distribution.
        # A5 fix: materialize cupy sig on host so numpy matmul does not raise.
        _sig_raw = sig.sig
        sig_arr = _sig_raw.get() if (hasattr(_sig_raw, 'get')
            and not isinstance(_sig_raw, np.ndarray)) else np.asarray(_sig_raw)
        area_pos = sig.p.area[:, np.newaxis] * sig.p.pos  # (nfaces, 3)
        if sig_arr.ndim == 1:
            dip = (area_pos.T @ sig_arr).reshape(1, 3)  # (1, 3)
        else:
            dip = (area_pos.T @ sig_arr).T  # (npol, 3)

        # Decompose electric fields into TE and TM
        te_pol, tm_pol, _ = self.decompose()

        # Compute Fresnel coefficients
        refl = self._fresnel_full(sig.enei)

        # Magnetic field of TM mode
        tm_h = np.cross(refl['ki'], tm_pol) / k0

        npol_dir = self.dir.shape[0]
        ext = np.zeros(npol_dir)

        for i in range(npol_dir):
            kr_i = refl['kr'][i, :]
            kt_i = refl['kt'][i, :]

            # Reflected and transmitted electric fields
            er_tm = np.cross(tm_h[i, :], kr_i) * k0 / np.dot(kr_i, kr_i)
            et_tm = np.cross(tm_h[i, :], kt_i) * k0 / np.dot(kt_i, kt_i)

            er = refl['re'][i] * te_pol[i, :] + refl['rm'][i] * er_tm
            et = refl['te'][i] * te_pol[i, :] + refl['tm'][i] * et_tm

            # Scattered far-fields in directions of reflected and transmitted light
            kr_norm = kr_i / np.linalg.norm(kr_i)
            kt_norm = kt_i / np.linalg.norm(kt_i)

            # Use SpectrumStatLayer.efarfield with specific directions
            dip_i = dip[i, :] if dip.shape[0] > 1 else dip[0, :]

            field_r, _ = self.spec.efarfield(dip_i.reshape(1, 3), sig.enei,
                dir = np.real(kr_norm).reshape(1, 3))
            field_t, _ = self.spec.efarfield(dip_i.reshape(1, 3), sig.enei,
                dir = np.real(kt_norm).reshape(1, 3))

            esr = field_r[0, :, 0]  # (3,)
            est = field_t[0, :, 0]  # (3,)

            # Zero out evanescent transmitted fields
            if np.abs(np.imag(kt_i[2])) > 1e-10:
                est = 0 * est

            norm_kr = np.linalg.norm(kr_i)

            # Extinction of reflected and transmitted beams
            extr = 4 * np.pi / norm_kr * np.imag(np.dot(er, esr))
            extt = 4 * np.pi / norm_kr * np.imag(np.dot(et, est))

            if key == 'refl':
                ext[i] = extr
            elif key == 'trans':
                ext[i] = extt
            else:
                ext[i] = extr + extt

        if ext.size == 1:
            return float(np.real(ext[0]))
        return np.real(ext)

    def scattering(self,
            sig: CompStruct) -> Any:

        # MATLAB: planewavestatlayer/scattering.m
        # Uses SpectrumStatLayer.scattering for Fresnel-aware far-field integration

        # Dipole moment.
        # A5 fix: materialize cupy sig on host so numpy matmul does not raise.
        _sig_raw = sig.sig
        sig_arr = _sig_raw.get() if (hasattr(_sig_raw, 'get')
            and not isinstance(_sig_raw, np.ndarray)) else np.asarray(_sig_raw)
        area_pos = sig.p.area[:, np.newaxis] * sig.p.pos
        if sig_arr.ndim == 1:
            dip = (area_pos.T @ sig_arr).reshape(1, 3)  # (1, 3)
        else:
            dip = (area_pos.T @ sig_arr).T  # (npol, 3)

        # Get refractive indices
        nb = msqrt(np.array([
            np.real(self.layer.eps[j](sig.enei)[0])
            for j in range(len(self.layer.eps))
        ]))

        # Total and differential radiated power via spectrum
        sca, dsca = self.spec.scattering(dip, sig.enei)

        # Normalize to incoming power: proportional to 0.5 * nb
        npol_dir = self.dir.shape[0]
        if np.isscalar(sca):
            sca = np.array([sca])

        for i in range(npol_dir):
            if self.dir[i, 2] < 0:
                sca_norm = 0.5 * nb[0]
            else:
                sca_norm = 0.5 * nb[-1]
            if npol_dir == 1:
                sca = sca / sca_norm
            else:
                sca[i] = sca[i] / sca_norm

        if sca.size == 1:
            return float(np.real(sca[0]))
        return np.real(sca)

    def absorption(self,
            sig: CompStruct) -> np.ndarray:

        # MATLAB: planewavestatlayer/absorption.m
        # Note: this equation only works for non-absorbing layer materials
        return self.extinction(sig) - self.scattering(sig)

    def __call__(self,
            p: Any,
            enei: float) -> CompStruct:

        return self.potential(p, enei)

    def __repr__(self) -> str:
        return 'PlaneWaveStatLayer(pol={}, medium={})'.format(
            self.pol.tolist(), self.medium)
