import os
import sys

from typing import List, Dict, Tuple, Optional, Union, Any, Callable

import numpy as np

from ..greenfun import CompStruct
from ..utils.matlab_compat import msqrt


class PlaneWaveRetLayer(object):

    name = 'planewave'
    needs = {'sim': 'ret'}

    def __init__(self,
            pol: np.ndarray,
            dir: np.ndarray,
            layer: Any,
            medium: int = 1,
            **options: Any) -> None:

        self.pol = np.asarray(pol, dtype = float)
        self.dir = np.asarray(dir, dtype = float)

        if self.pol.ndim == 1:
            self.pol = self.pol.reshape(1, -1)
        if self.dir.ndim == 1:
            self.dir = self.dir.reshape(1, -1)

        self.layer = layer

        # Spectrum for scattering calculations
        pinfty_arg = options.get('pinfty', None)
        from ..spectrum import SpectrumRetLayer
        self.spec = SpectrumRetLayer(pinfty_arg, layer)

    def _poslayer(self,
            z1: np.ndarray,
            z2: float) -> Dict[str, Any]:
        # MATLAB: poslayer() helper in potential.m
        z1 = np.atleast_1d(z1)
        z2_arr = np.atleast_1d(z2)
        ind1, _ = self.layer.indlayer(z1)
        ind2, _ = self.layer.indlayer(z2_arr)
        return {'r': 0, 'z1': z1, 'z2': z2_arr, 'ind1': ind1, 'ind2': ind2}

    def potential(self,
            p: Any,
            enei: float) -> CompStruct:
        # MATLAB: @planewaveretlayer/potential.m

        layer = self.layer
        k0 = 2 * np.pi / enei

        # Inside and outside medium of particle (MATLAB: p.inout expanded to faces)
        inout = p.inout_faces  # (nfaces, 2)

        # Find boundary elements connected to layer structure
        # MATLAB: ind1 = any(bsxfun(@eq, inout(:,1), layer.ind), 2)
        ind1 = np.zeros(p.nfaces, dtype = bool)
        ind2 = np.zeros(p.nfaces, dtype = bool)
        for li in layer.ind:
            ind1 |= (inout[:, 0] == li)
            ind2 |= (inout[:, 1] == li)
        ind = ind1 | ind2

        eta = 1e-8
        pos = p.pos
        nvec = p.nvec

        # Positions and displaced positions for faces connected to layer
        pos1 = pos[ind, :]
        pos2 = pos[ind, :] + eta * nvec[ind, :]

        npol = self.dir.shape[0]
        nfaces = p.nfaces

        # Initialize output arrays
        phi = np.zeros((nfaces, npol), dtype = complex)
        phip = np.zeros((nfaces, npol), dtype = complex)
        a = np.zeros((nfaces, 3, npol), dtype = complex)
        ap = np.zeros((nfaces, 3, npol), dtype = complex)

        for i in range(npol):
            pol_i = self.pol[i, :]
            dir_i = self.dir[i, :]

            # Excitation through upper or lower layer
            if dir_i[2] < 0:
                medium = layer.ind[0]
                z2 = layer.z[0] + 1e-10
            else:
                medium = layer.ind[-1]
                z2 = layer.z[-1] - 1e-10

            # Index to embedding medium faces
            indb = np.zeros(nfaces, dtype = bool)
            for col in range(inout.shape[1]):
                indb |= (inout[:, col] == medium)

            # Refractive index of embedding medium
            eps_val, _ = p.eps[medium - 1](enei)
            nb = np.sqrt(eps_val)

            # Parallel component of wavevector
            kpar = nb * k0 * msqrt(dir_i[0] ** 2 + dir_i[1] ** 2)

            # Reflection and transmission coefficients at pos1 and pos2
            pos_r1 = self._poslayer(pos1[:, 2], z2)
            pos_r2 = self._poslayer(pos2[:, 2], z2)

            r1 = layer.fresnel(enei, kpar, pos_r1)
            r2 = layer.fresnel(enei, kpar, pos_r2)

            # Inner product: pos_xy * dir_xy' + z2 * dir_z
            in1 = pos1[:, 0:2] @ dir_i[0:2] + z2 * dir_i[2]
            in2 = pos2[:, 0:2] @ dir_i[0:2] + z2 * dir_i[2]

            # Factor: exp(i * k0 * nb * inner) / (i * k0) * pol
            fac1 = np.exp(1j * k0 * nb * in1)[:, np.newaxis] / (1j * k0) * pol_i[np.newaxis, :]  # (nind, 3)
            fac2 = np.exp(1j * k0 * nb * in2)[:, np.newaxis] / (1j * k0) * pol_i[np.newaxis, :]

            # Vector potential from reflection
            # a(ind, 1:2) = fac1(:, 1:2) .* r1.p
            r1_p = np.atleast_1d(r1['p']).ravel()
            r1_hh = np.atleast_1d(r1['hh']).ravel()
            r1_sh = np.atleast_1d(r1['sh']).ravel()
            r2_p = np.atleast_1d(r2['p']).ravel()
            r2_hh = np.atleast_1d(r2['hh']).ravel()
            r2_sh = np.atleast_1d(r2['sh']).ravel()

            a[ind, 0, i] = fac1[:, 0] * r1_p
            a[ind, 1, i] = fac1[:, 1] * r1_p
            a[ind, 2, i] = fac1[:, 2] * r1_hh

            # Scalar potential from reflection
            phi[ind, i] = fac1[:, 2] * r1_sh

            # Derivative of vector potential (finite difference)
            ap[ind, 0, i] = (fac2[:, 0] * r2_p - a[ind, 0, i]) / eta
            ap[ind, 1, i] = (fac2[:, 1] * r2_p - a[ind, 1, i]) / eta
            ap[ind, 2, i] = (fac2[:, 2] * r2_hh - a[ind, 2, i]) / eta

            # Derivative of scalar potential (finite difference)
            phip[ind, i] = (fac2[:, 2] * r2_sh - phi[ind, i]) / eta

            # Direct excitation for faces in embedding medium
            pos_b = pos[indb, :]
            nvec_b = nvec[indb, :]

            a0 = np.exp(1j * k0 * nb * (pos_b @ dir_i))[:, np.newaxis] / (1j * k0) * pol_i[np.newaxis, :]
            a0p = a0 * (1j * k0 * nb * (nvec_b @ dir_i))[:, np.newaxis]

            a[indb, :, i] += a0
            ap[indb, :, i] += a0p

        # Build excitation output
        # MATLAB: exc arrays are full-sized (nfaces), only ind1/ind2 entries are non-zero
        phi1_out = np.zeros_like(phi)
        phi1p_out = np.zeros_like(phip)
        phi2_out = np.zeros_like(phi)
        phi2p_out = np.zeros_like(phip)
        a1_out = np.zeros_like(a)
        a1p_out = np.zeros_like(ap)
        a2_out = np.zeros_like(a)
        a2p_out = np.zeros_like(ap)

        phi1_out[ind1, :] = phi[ind1, :]
        phi1p_out[ind1, :] = phip[ind1, :]
        phi2_out[ind2, :] = phi[ind2, :]
        phi2p_out[ind2, :] = phip[ind2, :]
        a1_out[ind1, :, :] = a[ind1, :, :]
        a1p_out[ind1, :, :] = ap[ind1, :, :]
        a2_out[ind2, :, :] = a[ind2, :, :]
        a2p_out[ind2, :, :] = ap[ind2, :, :]

        exc = CompStruct(p, enei)

        if npol == 1:
            exc = exc.set(
                phi1 = phi1_out[:, 0], phi1p = phi1p_out[:, 0],
                phi2 = phi2_out[:, 0], phi2p = phi2p_out[:, 0],
                a1 = a1_out[:, :, 0], a1p = a1p_out[:, :, 0],
                a2 = a2_out[:, :, 0], a2p = a2p_out[:, :, 0])
        else:
            exc = exc.set(
                phi1 = phi1_out, phi1p = phi1p_out,
                phi2 = phi2_out, phi2p = phi2p_out,
                a1 = a1_out, a1p = a1p_out,
                a2 = a2_out, a2p = a2p_out)

        return exc

    def field(self,
            p: Any,
            enei: float,
            inout: int = 1) -> CompStruct:
        # MATLAB: @planewaveretlayer/field.m

        layer = self.layer
        k0 = 2 * np.pi / enei

        # Inside and outside medium per point. ComParticle exposes a
        # (nfaces, 2) inout_faces array; ComPoint (used when evaluating fields
        # at arbitrary positions) exposes per-group medium via `inout`. MATLAB
        # unifies these via `p.expand`; we emulate that here.
        if hasattr(p, 'inout_faces'):
            inout_arr = p.inout_faces  # (n, 2)
            npts = p.nfaces
        else:
            group_inout = np.atleast_1d(np.asarray(p.inout, dtype = int)).ravel()
            expanded = np.zeros(p.n, dtype = int)
            offset = 0
            for gi, grp in enumerate(p.p):
                expanded[offset:offset + grp.n] = group_inout[gi]
                offset += grp.n
            inout_arr = np.column_stack([expanded, expanded])
            npts = p.n

        # Find boundary elements connected to layer structure
        ind1 = np.zeros(npts, dtype = bool)
        ind2 = np.zeros(npts, dtype = bool)
        for li in layer.ind:
            ind1 |= (inout_arr[:, 0] == li)
            ind2 |= (inout_arr[:, 1] == li)
        ind = ind1 | ind2

        eta = 1e-8
        pos_all = p.pos
        npol = self.dir.shape[0]
        nfaces = npts

        # 4D arrays: (nfaces, 3, npol, 4) for undisplaced + 3 displaced
        phi = np.zeros((nfaces, npol, 4), dtype = complex)
        a = np.zeros((nfaces, 3, npol, 4), dtype = complex)

        for i in range(npol):
            pol_i = self.pol[i, :]
            dir_i = self.dir[i, :]

            if dir_i[2] < 0:
                medium = layer.ind[0]
                z2 = layer.z[0] + 1e-10
            else:
                medium = layer.ind[-1]
                z2 = layer.z[-1] - 1e-10

            indb = np.zeros(nfaces, dtype = bool)
            for col in range(inout_arr.shape[1]):
                indb |= (inout_arr[:, col] == medium)

            eps_val, _ = p.eps[medium - 1](enei)
            nb = np.sqrt(eps_val)
            kpar = nb * k0 * msqrt(dir_i[0] ** 2 + dir_i[1] ** 2)

            # Loop over Cartesian displacements (0=undisplaced, 1=dx, 2=dy, 3=dz)
            for kk in range(4):
                pos_disp = pos_all[ind, :].copy()
                if kk > 0:
                    pos_disp[:, kk - 1] += eta

                pos_r = self._poslayer(pos_disp[:, 2], z2)
                r = layer.fresnel(enei, kpar, pos_r)

                inner = pos_disp[:, 0:2] @ dir_i[0:2] + z2 * dir_i[2]
                fac = np.exp(1j * k0 * nb * inner)[:, np.newaxis] / (1j * k0) * pol_i[np.newaxis, :]

                r_p = np.atleast_1d(r['p']).ravel()
                r_hh = np.atleast_1d(r['hh']).ravel()
                r_sh = np.atleast_1d(r['sh']).ravel()

                a[ind, 0, i, kk] = fac[:, 0] * r_p
                a[ind, 1, i, kk] = fac[:, 1] * r_p
                a[ind, 2, i, kk] = fac[:, 2] * r_hh
                phi[ind, i, kk] = fac[:, 2] * r_sh

                # Direct excitation
                pos_dir = pos_all[indb, :].copy()
                if kk > 0:
                    pos_dir[:, kk - 1] += eta

                a0 = np.exp(1j * k0 * nb * (pos_dir @ dir_i))[:, np.newaxis] / (1j * k0) * pol_i[np.newaxis, :]
                a[indb, :, i, kk] += a0

        # Electric field: E = i*k0*A - grad(phi)
        e = 1j * k0 * a[:, :, :, 0]

        for kk in range(3):
            dphi = (phi[:, :, kk + 1] - phi[:, :, 0]) / eta  # (nfaces, npol)
            e[:, kk, :] -= dphi

        # Magnetic field: H = curl(A)
        h = np.zeros_like(e)
        for kk in range(3):
            for ii in range(3):
                da = (a[:, ii, :, kk + 1] - a[:, ii, :, 0]) / eta  # (nfaces, npol)
                # H_x = dA_z/dy - dA_y/dz, etc (Levi-Civita)
                if (kk, ii) == (1, 2):
                    h[:, 0, :] += da
                elif (kk, ii) == (2, 1):
                    h[:, 0, :] -= da
                elif (kk, ii) == (2, 0):
                    h[:, 1, :] += da
                elif (kk, ii) == (0, 2):
                    h[:, 1, :] -= da
                elif (kk, ii) == (0, 1):
                    h[:, 2, :] += da
                elif (kk, ii) == (1, 0):
                    h[:, 2, :] -= da

        if npol == 1:
            e = e[:, :, 0]
            h = h[:, :, 0]

        return CompStruct(p, enei, e = e, h = h)

    def extinction(self,
            sig: CompStruct) -> np.ndarray:
        # MATLAB: @planewaveretlayer/extinction.m
        # Uses efresnel to get reflected/transmitted fields and wavevectors

        layer = self.layer
        npol = self.dir.shape[0]
        ext = np.zeros(npol)

        e, k = layer.efresnel(self.pol, self.dir, sig.enei)

        for i in range(npol):
            # Scattered far-field in reflection direction
            kr = k['r'][i, :]
            kr_norm = msqrt(np.sum(np.abs(kr) ** 2))
            kr_hat = kr / kr_norm
            esr_field = self.spec.farfield(sig, kr_hat.reshape(1, -1))
            esr = esr_field.e
            if esr.ndim == 3:
                esr = esr[0, :, i]
            elif esr.ndim == 2:
                esr = esr[0, :]

            # Scattered far-field in transmission direction
            kt = k['t'][i, :]
            if np.abs(np.imag(kt[2])) > 1e-10:
                # Evanescent field
                est = np.zeros(3, dtype = complex)
            else:
                kt_norm = msqrt(np.sum(np.abs(kt) ** 2))
                kt_hat = kt / kt_norm
                est_field = self.spec.farfield(sig, kt_hat.reshape(1, -1))
                est = est_field.e
                if est.ndim == 3:
                    est = est[0, :, i]
                elif est.ndim == 2:
                    est = est[0, :]

            # Extinction of reflected and transmitted beam
            # MATLAB: 4*pi/norm(k.r(i,:)) * imag(dot(e.r(i,:), esr))
            # MATLAB dot(a,b) = sum(conj(a).*b), so conjugate of e is needed
            extr = 4 * np.pi / kr_norm * np.imag(np.sum(np.conj(e['r'][i, :]) * esr))
            extt = 4 * np.pi / kr_norm * np.imag(np.sum(np.conj(e['t'][i, :]) * est))
            ext[i] = extr + extt

        if npol == 1:
            return float(ext[0])
        return ext

    def absorption(self,
            sig: CompStruct) -> np.ndarray:
        # MATLAB: @planewaveretlayer/absorption.m
        return self.extinction(sig) - self.scattering(sig)[0]

    def scattering(self,
            sig: CompStruct) -> Tuple[np.ndarray, Any]:
        # MATLAB: @planewaveretlayer/scattering.m

        sca, dsca = self.spec.scattering(sig)

        # Refractive indices of each layer
        nb = np.zeros(len(self.layer.eps), dtype = complex)
        for i, eps_func in enumerate(self.layer.eps):
            eps_val, _ = eps_func(sig.enei)
            nb[i] = np.sqrt(eps_val)

        npol = self.dir.shape[0]
        sca = np.atleast_1d(np.real(sca).astype(float))

        for i in range(npol):
            if self.dir[i, 2] < 0:
                # Excitation through upper medium
                sca[i] = sca[i] / np.real(0.5 * nb[0])
            else:
                # Excitation through lower medium
                sca[i] = sca[i] / np.real(0.5 * nb[-1])
        if npol == 1:
            return float(sca[0]), dsca
        return sca, dsca

    def __call__(self,
            p: Any,
            enei: float) -> CompStruct:

        return self.potential(p, enei)

    def __repr__(self) -> str:
        return 'PlaneWaveRetLayer(pol={}, dir={})'.format(
            self.pol.tolist(), self.dir.tolist())
