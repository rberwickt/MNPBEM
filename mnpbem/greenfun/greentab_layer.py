import os
import sys

from typing import List, Dict, Tuple, Optional, Union, Any, Callable

import numpy as np
from scipy.interpolate import RegularGridInterpolator


class GreenTabLayer(object):

    name = 'greentablayer'

    def __new__(cls,
            layer: Any,
            tab: Optional[Any] = None,
            **options: Any):
        # MATLAB compgreentablayer treats a cell-array of tabs as separate
        # sub-tables and dispatches each query to the owning sub-tab via
        # inside.m.  When given a list with more than one entry we route
        # through _MultiGreenTabLayer to mirror that behaviour; merging them
        # into a single union grid (as before) caused 2D MATLAB queries to
        # become 3D in Python and produced systematically wrong z2 finite
        # differences in dipoleretlayer (Wave 45).
        if cls is GreenTabLayer and isinstance(tab, list) and len(tab) > 1:
            from .compgreentab_layer import _MultiGreenTabLayer
            return _MultiGreenTabLayer(layer, tab)
        return super().__new__(cls)

    def __init__(self,
            layer: Any,
            tab: Optional[Dict[str, Any]] = None,
            **options: Any) -> None:

        self.layer = layer

        if tab is not None:
            # Handle list of tabs (from tabspace with particle argument)
            if isinstance(tab, list):
                tab = self._merge_tabs(tab)
            r_in = tab.get('r', None)
            z1_in = tab.get('z1', None)
            z2_in = tab.get('z2', None)
            # MATLAB @greentablayer/private/init.m L12-L15:
            #   r  = unique(sort(max(layer.rmin, tab.r)))
            #   z1 = unique(sort(round(layer, tab.z1)))
            #   z2 = unique(sort(round(layer, tab.z2)))
            if r_in is not None:
                r_arr = np.atleast_1d(np.asarray(r_in, dtype=float)).ravel()
                r_arr = np.maximum(layer.rmin, r_arr)
                self.r = np.unique(np.sort(r_arr))
            else:
                self.r = None
            if z1_in is not None:
                z1_arr = np.atleast_1d(np.asarray(z1_in, dtype=float)).ravel()
                z1_rounded, = layer.round_z(z1_arr)
                self.z1 = np.unique(np.sort(np.asarray(z1_rounded).ravel()))
            else:
                self.z1 = None
            if z2_in is not None:
                z2_arr = np.atleast_1d(np.asarray(z2_in, dtype=float)).ravel()
                z2_rounded, = layer.round_z(z2_arr)
                self.z2 = np.unique(np.sort(np.asarray(z2_rounded).ravel()))
            else:
                self.z2 = None
        else:
            self.r = None
            self.z1 = None
            self.z2 = None

        # Per-component 3D tables: dict[name] -> (nr, nz1, nz2) complex array
        # These store NORMALIZED values (multiplied by distance-dependent factors)
        self._Gsav = None
        self._Frsav = None
        self._Fzsav = None

        self.enei = None
        self.G = None
        self.Fr = None
        self.Fz = None
        self._pos = None

        # Per-component caches (for eval_components path)
        self._enei_comp = None
        self._Gsav_comp = None
        self._Frsav_comp = None
        self._Fzsav_comp = None

    @staticmethod
    def _merge_tabs(tabs):
        """Merge a list of tabspace dicts into a single dict.

        When tabspace is called with a particle, it returns a list of dicts
        (one per layer combination). This merges them by taking the union
        of all r, z1, z2 grids.
        """
        if len(tabs) == 1:
            return tabs[0]

        # Merge grids by taking the union
        all_r = np.concatenate([t['r'] for t in tabs])
        all_z1 = np.concatenate([np.atleast_1d(t['z1']) for t in tabs])
        all_z2 = np.concatenate([np.atleast_1d(t['z2']) for t in tabs])

        r = np.sort(np.unique(all_r))
        z1 = np.sort(np.unique(all_z1))
        z2 = np.sort(np.unique(all_z2))

        return {'r': r, 'z1': z1, 'z2': z2}

    # ------------------------------------------------------------------
    # Grid normalization helpers (MATLAB: @greentablayer/norm.m, interp3.m)
    #
    # The reflected Green function G(r, z1, z2) varies rapidly near layer
    # interfaces.  To improve interpolation accuracy we multiply by
    # distance-dependent factors BEFORE tabulation (norm) and divide them
    # out AFTER interpolation (denorm).
    #
    # Normalization factors on the grid:
    #   d_grid  = sqrt(r_grid^2 + zmin_grid^2)
    #   G_norm  = G * d_grid
    #   Fr_norm = Fr * d_grid^3 / r_grid
    #   Fz_norm = Fz * d_grid^3 / zmin_grid
    #
    # Denormalization at query points:
    #   d_q    = sqrt(r_q^2 + zmin_q^2)
    #   G      = G_interp / d_q
    #   Fr     = Fr_interp * r_q / d_q^3
    #   Fz     = Fz_interp * zmin_q / d_q^3
    # ------------------------------------------------------------------

    def _grid_norm_factors(self):
        """Compute normalization factors (d, d^3/r, d^3/zmin) on the 3D grid.

        Returns
        -------
        d_grid : ndarray (nr, nz1, nz2)
        d3_over_r : ndarray (nr, nz1, nz2)
        d3_over_zmin : ndarray (nr, nz1, nz2)
        """
        nr = len(self.r)
        nz1 = len(self.z1)
        nz2 = len(self.z2)

        # mindist for each z1 and z2 value
        zmin_z1, _ = self.layer.mindist(self.z1)  # (nz1,)
        zmin_z2, _ = self.layer.mindist(self.z2)  # (nz2,)

        # Broadcast to 3D: zmin(iz1, iz2) = mindist(z1) + mindist(z2)
        # MATLAB interp3.m line 19: zmin = mindist(layer, z1) + mindist(layer, z2)
        zmin_grid = zmin_z1[np.newaxis, :, np.newaxis] + zmin_z2[np.newaxis, np.newaxis, :]
        # shape: (1, nz1, nz2) -> will broadcast with r

        r_grid = self.r[:, np.newaxis, np.newaxis]  # (nr, 1, 1)

        d_grid = np.sqrt(r_grid ** 2 + zmin_grid ** 2)  # (nr, nz1, nz2)

        # Avoid division by zero for r=0 or zmin=0
        r_safe = np.maximum(r_grid, np.finfo(float).eps)
        zmin_safe = np.maximum(zmin_grid, np.finfo(float).eps)

        d3_over_r = d_grid ** 3 / r_safe
        d3_over_zmin = d_grid ** 3 / zmin_safe

        return d_grid, d3_over_r, d3_over_zmin

    def _query_denorm_factors(self, r_q, z1_q, z2_q):
        """Compute denormalization factors at query points.

        Parameters
        ----------
        r_q, z1_q, z2_q : ndarray (n,)
            Query point coordinates (already clipped/rounded).

        Returns
        -------
        inv_d : ndarray (n,)      — 1/d
        r_over_d3 : ndarray (n,)  — r/d^3
        zmin_over_d3 : ndarray (n,) — zmin/d^3
        """
        zmin_z1, _ = self.layer.mindist(z1_q)
        zmin_z2, _ = self.layer.mindist(z2_q)
        zmin = zmin_z1 + zmin_z2

        d = np.sqrt(r_q ** 2 + zmin ** 2)
        d_safe = np.maximum(d, np.finfo(float).eps)

        inv_d = 1.0 / d_safe
        r_over_d3 = r_q / d_safe ** 3
        zmin_over_d3 = zmin / d_safe ** 3

        return inv_d, r_over_d3, zmin_over_d3

    def _query_denorm_factors_2d(self, r_q, z_eff):
        """Compute denormalization factors for 2D (single-z2) case.

        MATLAB interp2.m: zmin = mindist(layer, z_eff) where z_eff already
        folds z1 and z2 together.

        Parameters
        ----------
        r_q : ndarray (n,)
        z_eff : ndarray (n,)  — combined z1 + mindist(z2) or z1 - mindist(z2)

        Returns
        -------
        inv_d, r_over_d3, zmin_over_d3 : ndarray (n,)
        """
        zmin, _ = self.layer.mindist(z_eff)
        d = np.sqrt(r_q ** 2 + zmin ** 2)
        d_safe = np.maximum(d, np.finfo(float).eps)

        inv_d = 1.0 / d_safe
        r_over_d3 = r_q / d_safe ** 3
        zmin_over_d3 = zmin / d_safe ** 3

        return inv_d, r_over_d3, zmin_over_d3

    # ------------------------------------------------------------------
    # Multi-wavelength pre-computation
    # ------------------------------------------------------------------

    def set(self, enei_arr, **options):
        """Pre-compute Green function table at multiple wavelengths.

        MATLAB: @compgreentablayer/set.m
        Stores NORMALIZED 4D arrays (nr, nz1, nz2, n_enei) per component.
        """
        enei_arr = np.atleast_1d(np.asarray(enei_arr, dtype=float))
        n_enei = len(enei_arr)
        nr = len(self.r)
        nz1 = len(self.z1)
        nz2 = len(self.z2)

        # Grid normalization factors
        d_grid, d3_over_r, d3_over_zmin = self._grid_norm_factors()

        # Determine component names
        r_sample = self.r[:1]
        z1_sample = np.full_like(r_sample, self.z1[0])
        z2_sample = np.full_like(r_sample, self.z2[0])
        result = self.layer.green(enei_arr[0], r_sample, z1_sample, z2_sample)
        names = list(result[0].keys())

        # 4D arrays: (nr, nz1, nz2, n_enei) — normalized
        self._Gsav_multi = {k: np.zeros((nr, nz1, nz2, n_enei), dtype=complex) for k in names}
        self._Frsav_multi = {k: np.zeros((nr, nz1, nz2, n_enei), dtype=complex) for k in names}
        self._Fzsav_multi = {k: np.zeros((nr, nz1, nz2, n_enei), dtype=complex) for k in names}

        for ie, enei in enumerate(enei_arr):
            for iz1 in range(nz1):
                for iz2 in range(nz2):
                    r_vec = self.r
                    z1_vec = np.full_like(r_vec, self.z1[iz1])
                    z2_vec = np.full_like(r_vec, self.z2[iz2])
                    result = self.layer.green(enei, r_vec, z1_vec, z2_vec)
                    for name in names:
                        G_raw = np.asarray(result[0][name], dtype=complex)
                        Fr_raw = np.asarray(result[1][name], dtype=complex)
                        Fz_raw = np.asarray(result[2][name], dtype=complex)
                        # Normalize
                        self._Gsav_multi[name][:, iz1, iz2, ie] = G_raw * d_grid[:, iz1, iz2]
                        self._Frsav_multi[name][:, iz1, iz2, ie] = Fr_raw * d3_over_r[:, iz1, iz2]
                        self._Fzsav_multi[name][:, iz1, iz2, ie] = Fz_raw * d3_over_zmin[:, iz1, iz2]

        self._enei_tab = enei_arr
        return self

    def _interp_wavelength(self, enei):
        """Interpolate 4D multi-wavelength table to 3D at given wavelength.

        Result is stored in _Gsav_comp / _Frsav_comp / _Fzsav_comp dicts,
        still in NORMALIZED form.
        """
        enei_arr = self._enei_tab
        names = list(self._Gsav_multi.keys())
        self._Gsav_comp = {}
        self._Frsav_comp = {}
        self._Fzsav_comp = {}

        if len(enei_arr) == 1:
            # Single wavelength — no interpolation needed
            for name in names:
                self._Gsav_comp[name] = self._Gsav_multi[name][:, :, :, 0]
                self._Frsav_comp[name] = self._Frsav_multi[name][:, :, :, 0]
                self._Fzsav_comp[name] = self._Fzsav_multi[name][:, :, :, 0]
        else:
            idx = np.searchsorted(enei_arr, enei, side='right') - 1
            idx = np.clip(idx, 0, len(enei_arr) - 2)
            frac = (enei - enei_arr[idx]) / (enei_arr[idx + 1] - enei_arr[idx])
            for name in names:
                self._Gsav_comp[name] = (1 - frac) * self._Gsav_multi[name][:, :, :, idx] + frac * self._Gsav_multi[name][:, :, :, idx + 1]
                self._Frsav_comp[name] = (1 - frac) * self._Frsav_multi[name][:, :, :, idx] + frac * self._Frsav_multi[name][:, :, :, idx + 1]
                self._Fzsav_comp[name] = (1 - frac) * self._Fzsav_multi[name][:, :, :, idx] + frac * self._Fzsav_multi[name][:, :, :, idx + 1]

        self._enei_comp = enei

    # ------------------------------------------------------------------
    # eval / eval_components — public API
    # ------------------------------------------------------------------

    def eval(self,
            enei: float,
            r: np.ndarray,
            z1: np.ndarray,
            z2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate reflected Green function, returning summed totals.

        Returns (G, Fr, Fz) as plain arrays (sum of all reflection components).
        For per-component results, use eval_components().
        """
        G_dict, Fr_dict, Fz_dict = self.eval_components(enei, r, z1, z2)

        G = self._sum_components(G_dict)
        Fr = self._sum_components(Fr_dict)
        Fz = self._sum_components(Fz_dict)

        self.G = G
        self.Fr = Fr
        self.Fz = Fz
        self.enei = enei

        return G, Fr, Fz

    def eval_components(self,
            enei: float,
            r: np.ndarray,
            z1: np.ndarray,
            z2: np.ndarray) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Evaluate reflected Green function preserving per-component structure.

        Returns (G_dict, Fr_dict, Fz_dict) where each is a dict keyed by
        reflection names ('p', 'ss', 'hh', 'sh', 'hs').
        """
        if self.r is not None:
            return self._interp_components(enei, r, z1, z2)
        else:
            return self._compute_components(enei, r, z1, z2)

    # ------------------------------------------------------------------
    # Direct computation (no tabulation)
    # ------------------------------------------------------------------

    def _compute_components(self,
            enei: float,
            r: np.ndarray,
            z1: np.ndarray,
            z2: np.ndarray) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:

        result = self.layer.green(enei, r, z1, z2)
        G_dict = {k: np.asarray(v, dtype=complex) for k, v in result[0].items()}
        Fr_dict = {k: np.asarray(v, dtype=complex) for k, v in result[1].items()}
        Fz_dict = {k: np.asarray(v, dtype=complex) for k, v in result[2].items()}
        self._pos = result[3]
        return G_dict, Fr_dict, Fz_dict

    # ------------------------------------------------------------------
    # Tabulation + interpolation (with norm/denorm smoothing)
    # ------------------------------------------------------------------

    def _ensure_tab(self, enei):
        """Ensure per-component table is computed/interpolated for this enei."""
        if hasattr(self, '_Gsav_multi') and self._Gsav_multi is not None:
            # Multi-wavelength path
            if self._enei_comp is None or not np.isclose(self._enei_comp, enei):
                self._interp_wavelength(enei)
        elif self._Gsav_comp is None or (
                self._enei_comp is not None and not np.isclose(self._enei_comp, enei)):
            self._compute_tab(enei)

    def _interp_components(self,
            enei: float,
            r: np.ndarray,
            z1: np.ndarray,
            z2: np.ndarray) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Interpolate per-component table with norm/denorm smoothing.

        MATLAB: @greentablayer/interp.m -> interp2.m / interp3.m
        """
        shape = np.asarray(r).shape

        self._ensure_tab(enei)

        # Query points
        r_q = np.clip(np.asarray(r, dtype=float).ravel(), self.r[0], self.r[-1])
        # MATLAB interp2.m:17 / interp3.m:17: [z1, z2] = round(obj.layer, z1, z2)
        # Push points within zmin of an interface outward by zmin so they
        # fall inside the tabulation domain. Critical for near-surface
        # accuracy (demospecret13). Without this, near-layer queries
        # extrapolate below the grid start (0.999 * zmin) and lose precision.
        z1_arr = np.asarray(z1, dtype=float).ravel()
        z2_arr = np.asarray(z2, dtype=float).ravel()
        try:
            z1_rounded, z2_rounded = self.layer.round_z(z1_arr, z2_arr)
            z1_q = np.asarray(z1_rounded, dtype=float).ravel()
            z2_q = np.asarray(z2_rounded, dtype=float).ravel()
        except (AttributeError, TypeError):
            z1_q = z1_arr
            z2_q = z2_arr

        names = list(self._Gsav_comp.keys())
        G_dict = {}
        Fr_dict = {}
        Fz_dict = {}

        if len(self.z2) == 1:
            # --------------------------------------------------------
            # 2D interpolation in uppermost/lowermost layer
            # MATLAB: interp2.m
            # --------------------------------------------------------
            z2_ref = self.z2[0]
            mindist_z2, _ = self.layer.mindist(z2_q)
            if z2_ref >= self.layer.z[0]:
                z1_eff = np.clip(z1_q + mindist_z2, self.z1[0], self.z1[-1])
            else:
                z1_eff = np.clip(z1_q - mindist_z2, self.z1[0], self.z1[-1])

            points_2d = np.column_stack([r_q, z1_eff])
            grid_2d = (self.r, self.z1)

            # Denormalization factors at query points
            inv_d, r_over_d3, zmin_over_d3 = self._query_denorm_factors_2d(r_q, z1_eff)

            # Batched interpolation: stack all (component x field) slabs and
            # locate each query cell once.  Each name contributes 3 slabs
            # (G, Fr, Fz); the cell index only depends on (r, z1) so doing
            # it once for the whole stack removes the per-slab redundancy.
            slabs = []
            for name in names:
                slabs.append(self._Gsav_comp[name][:, :, 0])
                slabs.append(self._Frsav_comp[name][:, :, 0])
                slabs.append(self._Fzsav_comp[name][:, :, 0])
            vals = self._interp_complex_batch(grid_2d, slabs, points_2d)

            for ci, name in enumerate(names):
                G_n = vals[3 * ci + 0]
                Fr_n = vals[3 * ci + 1]
                Fz_n = vals[3 * ci + 2]
                # Denormalize
                G_dict[name] = (G_n * inv_d).reshape(shape)
                Fr_dict[name] = (Fr_n * r_over_d3).reshape(shape)
                Fz_dict[name] = (Fz_n * zmin_over_d3).reshape(shape)
        else:
            # --------------------------------------------------------
            # 3D interpolation
            # MATLAB: interp3.m
            # --------------------------------------------------------
            z1_q = np.clip(z1_q, self.z1[0], self.z1[-1])
            z2_q = np.clip(z2_q, self.z2[0], self.z2[-1])
            points = np.column_stack([r_q, z1_q, z2_q])
            grid = (self.r, self.z1, self.z2)

            # Denormalization factors at query points
            inv_d, r_over_d3, zmin_over_d3 = self._query_denorm_factors(r_q, z1_q, z2_q)

            slabs = []
            for name in names:
                slabs.append(self._Gsav_comp[name])
                slabs.append(self._Frsav_comp[name])
                slabs.append(self._Fzsav_comp[name])
            vals = self._interp_complex_batch(grid, slabs, points)

            for ci, name in enumerate(names):
                G_n = vals[3 * ci + 0]
                Fr_n = vals[3 * ci + 1]
                Fz_n = vals[3 * ci + 2]
                # Denormalize
                G_dict[name] = (G_n * inv_d).reshape(shape)
                Fr_dict[name] = (Fr_n * r_over_d3).reshape(shape)
                Fz_dict[name] = (Fz_n * zmin_over_d3).reshape(shape)

        return G_dict, Fr_dict, Fz_dict

    @staticmethod
    def _interp_complex(grid, data, points):
        """Interpolate complex array on a regular grid (split real/imag).

        Dispatches to the numba multilinear kernel when numba is importable
        (it is numerically identical to RegularGridInterpolator's linear
        method) and falls back to scipy otherwise.  The numba path is the
        default whenever numba is present because the scipy path rebuilds a
        fresh RegularGridInterpolator for every component/field and is
        ~20-50x slower on the O(n^2) substrate query grids that dominate
        BEMRetLayer.init() — set ``MNPBEM_NUMBA=0`` to force the scipy
        reference path.
        """
        from ._numba_layer import trilinear_complex, _numba_layer_preferred
        if _numba_layer_preferred():
            return trilinear_complex(grid, data, points)
        val_r = RegularGridInterpolator(
            grid, data.real, method='linear',
            bounds_error=False, fill_value=None)(points)
        val_i = RegularGridInterpolator(
            grid, data.imag, method='linear',
            bounds_error=False, fill_value=None)(points)
        return val_r + 1j * val_i

    @staticmethod
    def _interp_complex_batch(grid, slabs, points):
        """Interpolate a list of complex grids at common query points.

        Stacks ``slabs`` into a single (K, *grid_shape) array so the numba
        kernel locates each query point's cell once and reuses the weights
        across all K slabs (the 5 reflection components x {G, Fr, Fz} of
        the substrate Green function).  Returns a (K, n) array.  Falls back
        to per-slab scipy interpolation when numba is unavailable so the
        reference path stays bit-identical.
        """
        from ._numba_layer import (trilinear_complex_batch,
                                    _numba_layer_preferred)
        data_stack = np.ascontiguousarray(np.stack(slabs, axis=0),
                                          dtype=complex)
        if _numba_layer_preferred():
            return trilinear_complex_batch(grid, data_stack, points)
        out = np.empty((data_stack.shape[0], points.shape[0]), dtype=complex)
        for kk in range(data_stack.shape[0]):
            d = data_stack[kk]
            val_r = RegularGridInterpolator(
                grid, d.real, method='linear',
                bounds_error=False, fill_value=None)(points)
            val_i = RegularGridInterpolator(
                grid, d.imag, method='linear',
                bounds_error=False, fill_value=None)(points)
            out[kk] = val_r + 1j * val_i
        return out

    def _compute_tab(self,
            enei: float) -> None:
        """Tabulate Green function for all grid points, storing NORMALIZED
        per-component values.

        MATLAB: @greentablayer/eval.m (with 'new' key) + norm.m
        """
        nr = len(self.r)
        nz1 = len(self.z1)
        nz2 = len(self.z2)

        # Grid normalization factors
        d_grid, d3_over_r, d3_over_zmin = self._grid_norm_factors()

        # Determine component names from a sample call
        r_sample = self.r[:1]
        z1_sample = np.full_like(r_sample, self.z1[0])
        z2_sample = np.full_like(r_sample, self.z2[0])
        result = self.layer.green(enei, r_sample, z1_sample, z2_sample)
        names = list(result[0].keys())

        self._Gsav_comp = {k: np.zeros((nr, nz1, nz2), dtype=complex) for k in names}
        self._Frsav_comp = {k: np.zeros((nr, nz1, nz2), dtype=complex) for k in names}
        self._Fzsav_comp = {k: np.zeros((nr, nz1, nz2), dtype=complex) for k in names}

        for iz1 in range(nz1):
            for iz2 in range(nz2):
                r_vec = self.r
                z1_vec = np.full_like(r_vec, self.z1[iz1])
                z2_vec = np.full_like(r_vec, self.z2[iz2])
                result = self.layer.green(enei, r_vec, z1_vec, z2_vec)
                for name in names:
                    G_raw = np.asarray(result[0][name], dtype=complex)
                    Fr_raw = np.asarray(result[1][name], dtype=complex)
                    Fz_raw = np.asarray(result[2][name], dtype=complex)
                    # Store normalized values
                    self._Gsav_comp[name][:, iz1, iz2] = G_raw * d_grid[:, iz1, iz2]
                    self._Frsav_comp[name][:, iz1, iz2] = Fr_raw * d3_over_r[:, iz1, iz2]
                    self._Fzsav_comp[name][:, iz1, iz2] = Fz_raw * d3_over_zmin[:, iz1, iz2]

        self._enei_comp = enei

    # ------------------------------------------------------------------
    # Legacy _interp / _compute (kept for backward compat of eval())
    # ------------------------------------------------------------------
    # eval() now delegates to eval_components() + _sum_components(),
    # so these are no longer needed as separate paths.

    # ------------------------------------------------------------------
    # Grid setup
    # ------------------------------------------------------------------

    def setup_grid(self,
            r: np.ndarray,
            z1: np.ndarray,
            z2: np.ndarray) -> None:

        self.r = np.asarray(r, dtype=float)
        self.z1 = np.asarray(z1, dtype=float)
        self.z2 = np.asarray(z2, dtype=float)
        # Invalidate caches
        self._Gsav = None
        self._Frsav = None
        self._Fzsav = None
        self._Gsav_comp = None
        self._Frsav_comp = None
        self._Fzsav_comp = None
        self.enei = None
        self._enei_comp = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sum_components(d):
        """Sum all reflection-component arrays stored in a dict.

        layer.green() returns dicts keyed by reflection names
        ('p', 'ss', 'hs', 'sh', 'hh').  This helper adds them together
        to yield the total reflected Green function as a single array.

        If *d* is already an ndarray (not a dict) it is returned as-is.
        """
        if isinstance(d, dict):
            total = None
            for v in d.values():
                if total is None:
                    total = np.array(v, dtype=complex)
                else:
                    total = total + np.array(v, dtype=complex)
            return total if total is not None else np.zeros(0, dtype=complex)
        return np.asarray(d, dtype=complex)

    def norm(self) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        # MATLAB: @greentablayer/norm.m
        # Multiply Green function with distance-dependent normalization factors.
        assert self.G is not None

        # Tabulated radii and minimal distances from pos
        r = self._pos['r']
        zmin = self._pos['zmin']
        # Distance
        d = np.sqrt(r ** 2 + zmin ** 2)

        G_out = {}
        Fr_out = {}
        Fz_out = {}
        for name in self.G.keys():
            G_out[name] = self.G[name] * d
            Fr_out[name] = self.Fr[name] * d ** 3 / r
            Fz_out[name] = self.Fz[name] * d ** 3 / zmin

        return G_out, Fr_out, Fz_out

    def inside(self,
            r: np.ndarray,
            z1: np.ndarray,
            z2: Optional[np.ndarray] = None) -> np.ndarray:
        # MATLAB: @greentablayer/inside.m
        layer = self.layer

        r = np.asarray(r, dtype=float)
        z1 = np.asarray(z1, dtype=float)

        # Round radii and z-values
        r = np.maximum(layer.rmin, r)
        if z2 is not None:
            z2 = np.asarray(z2, dtype=float)
            z1, z2 = layer.round_z(z1, z2)
        else:
            z1, = layer.round_z(z1)

        def fun(x: np.ndarray, limits: np.ndarray) -> np.ndarray:
            return (x >= np.min(limits)) & (x <= np.max(limits))

        # Uppermost or lowermost layer (single z2 value in table)
        if np.atleast_1d(self.z2).size == 1:
            ind1, _ = layer.indlayer(z1)
            ind2, _ = layer.indlayer(z2)

            # Find z-values in uppermost or lowermost layer
            in1 = (ind1 == ind2) & (ind1 == 1)
            in2 = (ind1 == ind2) & (ind1 == layer.n + 1)
            result = in1 | in2

            if np.any(in1):
                mindist_z2, _ = layer.mindist(z2[in1])
                result[in1] = fun(r[in1], self.r) & fun(z1[in1] + mindist_z2, self.z1)
            if np.any(in2):
                mindist_z2, _ = layer.mindist(z2[in2])
                result[in2] = fun(r[in2], self.r) & fun(z1[in2] - mindist_z2, self.z1)
        else:
            result = fun(r, self.r) & fun(z1, self.z1) & fun(z2, self.z2)

        return result

    def ismember(self,
            layer: Any,
            enei: Optional[np.ndarray] = None) -> bool:
        # MATLAB: @greentablayer/ismember.m
        # Check if precomputed table is compatible with given layer and enei.

        # enei not set
        if self.enei is None and self._enei_comp is None:
            if not hasattr(self, '_enei_tab') or self._enei_tab is None:
                return False

        # Check wavelength range
        if hasattr(self, '_enei_tab') and self._enei_tab is not None:
            enei_tab = np.atleast_1d(self._enei_tab)
        elif self._enei_comp is not None:
            enei_tab = np.atleast_1d(self._enei_comp)
        elif self.enei is not None:
            enei_tab = np.atleast_1d(self.enei)
        else:
            return False

        if enei is not None:
            enei = np.atleast_1d(enei)
            if np.min(enei) < np.min(enei_tab) or np.max(enei) > np.max(enei_tab):
                return False

        # Check layer structure compatibility
        if layer.n != self.layer.n:
            return False
        if not np.allclose(layer.z, self.layer.z):
            return False

        # Evaluate dielectric functions and compare
        for eps_new, eps_old in zip(layer.eps, self.layer.eps):
            for e in enei_tab:
                val_new = eps_new(e)
                val_old = eps_old(e)
                # eps functions return (eps, k) tuples
                if isinstance(val_new, tuple):
                    val_new = val_new[0]
                if isinstance(val_old, tuple):
                    val_old = val_old[0]
                if abs(val_new - val_old) > 1e-8:
                    return False

        return True

    def parset(self,
            enei_arr: np.ndarray,
            **options: Any) -> 'GreenTabLayer':
        """Pre-compute Green function table for multiple wavelengths.

        MATLAB: @greentablayer/parset.m
        Stores NORMALIZED 4D per-component arrays.
        """
        enei_arr = np.atleast_1d(np.asarray(enei_arr, dtype=float))
        n_enei = len(enei_arr)
        nr = len(self.r)
        nz1 = len(self.z1)
        nz2 = len(self.z2)

        # Grid normalization factors
        d_grid, d3_over_r, d3_over_zmin = self._grid_norm_factors()

        pos_saved = None

        for ien in range(n_enei):
            # Determine component names from a sample call on the first iteration
            if ien == 0:
                r_sample = self.r[:1]
                z1_sample = np.full_like(r_sample, self.z1[0])
                z2_sample = np.full_like(r_sample, self.z2[0])
                result_sample = self.layer.green(enei_arr[0], r_sample, z1_sample, z2_sample)
                names = list(result_sample[0].keys())

                siz = (n_enei, nr, nz1, nz2)
                Gsav = {k: np.zeros(siz, dtype=complex) for k in names}
                Frsav = {k: np.zeros(siz, dtype=complex) for k in names}
                Fzsav = {k: np.zeros(siz, dtype=complex) for k in names}

            for iz1 in range(nz1):
                for iz2 in range(nz2):
                    r_vec = self.r
                    z1_vec = np.full_like(r_vec, self.z1[iz1])
                    z2_vec = np.full_like(r_vec, self.z2[iz2])
                    result = self.layer.green(enei_arr[ien], r_vec, z1_vec, z2_vec)
                    if pos_saved is None:
                        pos_saved = result[3]
                    for name in names:
                        G_raw = np.asarray(result[0][name], dtype=complex)
                        Fr_raw = np.asarray(result[1][name], dtype=complex)
                        Fz_raw = np.asarray(result[2][name], dtype=complex)
                        # Store normalized values
                        Gsav[name][ien, :, iz1, iz2] = G_raw * d_grid[:, iz1, iz2]
                        Frsav[name][ien, :, iz1, iz2] = Fr_raw * d3_over_r[:, iz1, iz2]
                        Fzsav[name][ien, :, iz1, iz2] = Fz_raw * d3_over_zmin[:, iz1, iz2]

        # Store results — shape is [n_enei, nr, nz1, nz2] per component
        # Convert to (nr, nz1, nz2, n_enei) to match set() convention
        self._Gsav_multi = {k: np.moveaxis(Gsav[k], 0, -1) for k in names}
        self._Frsav_multi = {k: np.moveaxis(Frsav[k], 0, -1) for k in names}
        self._Fzsav_multi = {k: np.moveaxis(Fzsav[k], 0, -1) for k in names}

        self.enei = enei_arr
        self._pos = pos_saved
        self._enei_tab = enei_arr

        return self

    def __repr__(self) -> str:
        r_info = 'nr={}'.format(len(self.r)) if self.r is not None else 'no table'
        return 'GreenTabLayer({})'.format(r_info)
