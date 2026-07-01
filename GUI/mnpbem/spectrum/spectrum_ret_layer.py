import os
import sys

from typing import List, Dict, Tuple, Optional, Union, Any, Callable

import numpy as np

from .spectrum_ret import SpectrumRet, trisphere_unit, _PinftyStruct, _load_pinfty_default
from ..greenfun import CompStruct


class SpectrumRetLayer(object):

    def __init__(self,
            pinfty: Optional[Any] = None,
            layer: Optional[Any] = None,
            medium: int = 1) -> None:

        self.layer = layer

        # Track whether pinfty came from the user (relevant for MATLAB
        # spectrumret isstruct-fallback bug-compat in DipoleRetLayer.farfield).
        self._user_pinfty = pinfty is not None

        # Handle different input types
        if pinfty is None:
            # MATLAB @spectrumretlayer/init.m uses
            # trispheresegment(linspace(0,2*pi,21), linspace(0,pi,21), 2).
            # Load the MATLAB-exported pinfty for bit-identical face ordering
            # (Python trispheresegment's mesh-cleaning step reorders faces).
            data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
            pinfty_file = os.path.join(data_dir, 'pinfty_layer_default.bin')
            if os.path.exists(pinfty_file):
                with open(pinfty_file, 'rb') as f:
                    n = np.fromfile(f, dtype=np.int32, count=1)[0]
                    nx = np.fromfile(f, dtype=np.float64, count=n)
                    ny = np.fromfile(f, dtype=np.float64, count=n)
                    nz = np.fromfile(f, dtype=np.float64, count=n)
                    px = np.fromfile(f, dtype=np.float64, count=n)
                    py = np.fromfile(f, dtype=np.float64, count=n)
                    pz = np.fromfile(f, dtype=np.float64, count=n)
                    area = np.fromfile(f, dtype=np.float64, count=n)
                nvec = np.column_stack([nx, ny, nz])
                pos = np.column_stack([px, py, pz])
                self.pinfty = _PinftyStruct(nvec, area, pos=pos)
            else:
                from ..geometry import trispheresegment
                phi_grid = np.linspace(0, 2 * np.pi, 21)
                theta_grid = np.linspace(0, np.pi, 21)
                p_inf = trispheresegment(phi_grid, theta_grid, 2)
                self.pinfty = _PinftyStruct(p_inf.nvec.copy(), p_inf.area.copy())
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
        self.ndir = len(self.nvec)

        # Upper and lower medium indices from layer
        if layer is not None:
            self.medium = [layer.ind[0], layer.ind[-1]]
        else:
            self.medium = [medium, medium]

        # Separate into upper and lower hemisphere
        self._init_hemispheres()

    def _init_hemispheres(self) -> None:
        z = self.nvec[:, 2]
        self.ind_up = np.where(z >= 0)[0]
        self.ind_down = np.where(z < 0)[0]

    def farfield(self,
            sig: Any,
            direction: Optional[np.ndarray] = None) -> CompStruct:
        # MATLAB: @spectrumretlayer/farfield.m

        if direction is None:
            direction = self.nvec

        direction = np.atleast_2d(direction)

        p = sig.p if hasattr(sig, 'p') else sig['p']
        enei = sig.enei if hasattr(sig, 'enei') else sig['enei']

        layer = self.layer
        k0 = 2 * np.pi / enei

        # Wavenumbers in layer media
        k_vals = np.empty(len(layer.eps), dtype = complex)
        for i, eps_func in enumerate(layer.eps):
            _, k_vals[i] = eps_func(enei)

        # Upper and lower medium
        medium = [layer.ind[0], layer.ind[-1]]

        # Indices for upper (z >= 0) and lower (z < 0) hemispheres
        ind1 = np.where(direction[:, 2] >= 0)[0]
        ind2 = np.where(direction[:, 2] < 0)[0]

        # Get charges and currents
        sig2 = sig.sig2 if hasattr(sig, 'sig2') else np.zeros(p.nfaces)
        h2 = sig.h2 if hasattr(sig, 'h2') else np.zeros((p.nfaces, 3))

        if sig2.ndim == 1:
            sig2 = sig2[:, np.newaxis]
        if h2.ndim == 2:
            h2 = h2[:, :, np.newaxis]

        npol = h2.shape[2]
        ndir = len(direction)

        # Allocate reflected/transmitted vector potential
        a = np.zeros((ndir, 3, npol), dtype = complex)

        # Boundary elements connected to layer structure
        inout_arr = p.inout_faces
        ind_layer = np.zeros(p.nfaces, dtype = bool)
        for li in layer.ind:
            ind_layer |= (inout_arr[:, 1] == li)

        # z-values of particle centroids for connected faces
        z2_part = p.pos[ind_layer, 2]

        # Position structures for upper and lower hemisphere
        # MATLAB: pos1 for upper (z1 = layer.z[0], ind1_pos = 1)
        # MATLAB: pos2 for lower (z1 = layer.z[-1], ind1_pos = n+1)
        ind2_layer, _ = layer.indlayer(z2_part)

        pos_up = {
            'z1': np.atleast_1d(layer.z[0]),
            'ind1': np.atleast_1d(1),
            'z2': z2_part,
            'ind2': ind2_layer
        }
        pos_down = {
            'z1': np.atleast_1d(layer.z[-1]),
            'ind1': np.atleast_1d(layer.n + 1),
            'z2': z2_part,
            'ind2': ind2_layer
        }

        # Upper hemisphere (positive z propagation)
        if np.imag(k_vals[0]) == 0 and len(ind1) > 0:
            for idx in ind1:
                d = direction[idx, :]
                kpar = np.real(k_vals[0]) * np.sqrt(1 - d[2] ** 2)

                r, _ = layer.reflection(enei, kpar, pos_up)

                # Distance for phase factor
                dist = p.pos[ind_layer, 0:2] @ d[0:2] + layer.z[0] * d[2]

                # Phase factor: exp(-i*k*dist) * area
                phase_arr = np.exp(-1j * k_vals[0] * dist) * p.area[ind_layer]  # (nind,)

                r_p = np.asarray(r['p']).ravel()     # (nind,)
                r_hh = np.asarray(r['hh']).ravel()
                r_hs = np.asarray(r['hs']).ravel()

                rp_phase = r_p * phase_arr    # (nind,)
                rhh_phase = r_hh * phase_arr
                rhs_phase = r_hs * phase_arr

                for ipol in range(npol):
                    h2_x = h2[ind_layer, 0, ipol]
                    h2_y = h2[ind_layer, 1, ipol]
                    h2_z = h2[ind_layer, 2, ipol]
                    sig2_pol = sig2[ind_layer, ipol]

                    a[idx, 0, ipol] = np.dot(rp_phase, h2_x)
                    a[idx, 1, ipol] = np.dot(rp_phase, h2_y)
                    a[idx, 2, ipol] = np.dot(rhh_phase, h2_z) + np.dot(rhs_phase, sig2_pol)

        # Lower hemisphere (negative z propagation)
        if np.imag(k_vals[-1]) == 0 and len(ind2) > 0:
            for idx in ind2:
                d = direction[idx, :]
                kpar = np.real(k_vals[-1]) * np.sqrt(1 - d[2] ** 2)

                r, _ = layer.reflection(enei, kpar, pos_down)

                dist = p.pos[ind_layer, 0:2] @ d[0:2] + layer.z[-1] * d[2]
                phase_arr = np.exp(-1j * k_vals[-1] * dist) * p.area[ind_layer]

                r_p = np.asarray(r['p']).ravel()
                r_hh = np.asarray(r['hh']).ravel()
                r_hs = np.asarray(r['hs']).ravel()

                rp_phase = r_p * phase_arr
                rhh_phase = r_hh * phase_arr
                rhs_phase = r_hs * phase_arr

                for ipol in range(npol):
                    h2_x = h2[ind_layer, 0, ipol]
                    h2_y = h2[ind_layer, 1, ipol]
                    h2_z = h2[ind_layer, 2, ipol]
                    sig2_pol = sig2[ind_layer, ipol]

                    a[idx, 0, ipol] = np.dot(rp_phase, h2_x)
                    a[idx, 1, ipol] = np.dot(rp_phase, h2_y)
                    a[idx, 2, ipol] = np.dot(rhh_phase, h2_z) + np.dot(rhs_phase, sig2_pol)

        # Electric field from reflected/transmitted potentials
        e = 1j * k0 * a

        # Direct far-fields in upper and lower medium
        if len(ind1) > 0:
            spec1 = SpectrumRet(self.pinfty, medium = medium[0])
            field1 = spec1.farfield(sig, direction[ind1, :])
            e1 = field1.e
            if e1.ndim == 2:
                e1 = e1[:, :, np.newaxis]
            e[ind1, :, :] += e1

        if len(ind2) > 0:
            spec2 = SpectrumRet(self.pinfty, medium = medium[1])
            field2 = spec2.farfield(sig, direction[ind2, :])
            e2 = field2.e
            if e2.ndim == 2:
                e2 = e2[:, :, np.newaxis]
            e[ind2, :, :] += e2

        # Make electric field transversal: e = e - dir * dot(dir, e)
        for ipol in range(npol):
            dot_de = np.sum(direction * e[:, :, ipol], axis = 1)  # (ndir,)
            e[:, :, ipol] -= direction * dot_de[:, np.newaxis]

        # Magnetic field: H = (k/k0) * cross(dir, E)
        h = np.zeros_like(e)
        for ipol in range(npol):
            if len(ind1) > 0:
                dir_up = direction[ind1, :]
                h[ind1, :, ipol] = (k_vals[0] / k0) * np.cross(dir_up, e[ind1, :, ipol])
            if len(ind2) > 0:
                dir_down = direction[ind2, :]
                h[ind2, :, ipol] = (k_vals[-1] / k0) * np.cross(dir_down, e[ind2, :, ipol])

        if npol == 1:
            e = e[:, :, 0]
            h = h[:, :, 0]

        field = CompStruct(self.pinfty, enei, e = e, h = h)
        return field

    def scattering(self,
            sig: Any) -> Tuple[np.ndarray, Any]:
        # MATLAB: @spectrumretlayer/scattering.m
        # scattering(farfield(obj, sig), obj.medium)

        field = self.farfield(sig)
        e = field.e
        h_field = field.h
        enei = sig.enei if hasattr(sig, 'enei') else sig['enei']

        if e.ndim == 2:
            e = e[:, :, np.newaxis]
            h_field = h_field[:, :, np.newaxis]

        npol = e.shape[2]

        dsca_arr = np.zeros((self.ndir, npol))

        for ipol in range(npol):
            poynting = np.cross(e[:, :, ipol], np.conj(h_field[:, :, ipol]))
            dsca_arr[:, ipol] = 0.5 * np.real(np.sum(self.nvec * poynting, axis = 1))

        sca = np.dot(self.area, dsca_arr)

        if npol == 1:
            sca = sca[0]
            dsca_arr = dsca_arr[:, 0]

        dsca = CompStruct(field.p, enei, dsca = dsca_arr)

        return sca, dsca

    def __repr__(self) -> str:
        return 'SpectrumRetLayer(ndir={})'.format(self.ndir)
