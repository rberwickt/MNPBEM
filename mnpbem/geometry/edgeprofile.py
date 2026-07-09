import os
import sys

import numpy as np
from typing import Optional, Tuple, List, Union
from scipy.interpolate import CubicSpline

from ..utils.matlab_compat import mlinspace, mcos, msin


class EdgeProfile(object):

    def __init__(self,
            *args,
            e: float = 0.4,
            dz: float = 0.15,
            mode: str = '00',
            min_z: Optional[float] = None,
            max_z: Optional[float] = None,
            center: Optional[float] = None):
        # MATLAB: @edgeprofile/edgeprofile.m + @edgeprofile/init.m
        self.pos = None  # (z, d) values for edge profile -- stored as Nx2 array
        self.z = None  # z-values for extruding polygon

        if len(args) == 0:
            return

        # Explicit constructor with pos and z arrays
        first_arg = np.asarray(args[0])
        if first_arg.size != 1:
            # Called as EdgeProfile(pos, z, ...)
            self.pos = np.asarray(args[0], dtype = float)
            self.z = np.asarray(args[1], dtype = float)
        else:
            # Called as EdgeProfile(height) or EdgeProfile(height, nz)
            height = float(first_arg)
            nz = 7  # default
            if len(args) >= 2:
                nz = int(args[1])

            if mode == '11':
                # Sharp edges on both ends
                self.pos = np.array([
                    [np.nan, 0.0],
                    [0.0, -0.5 * height],
                    [0.0, 0.5 * height],
                    [np.nan, 0.0]])
                self.z = mlinspace(-0.5, 0.5, nz) * height
            else:
                # Supercircle edge profile
                pows = lambda x: np.sign(x) * np.abs(x) ** e

                # Angles
                phi = mlinspace(-1, 1, 51).reshape(-1) * np.pi / 2.0

                x = pows(mcos(phi))
                z_vals = pows(msin(phi))

                # Find index closest to (1 - dz)
                ind = np.argmin(np.abs(z_vals - (1.0 - dz)))

                # Build edge profile
                self.pos = 0.5 * height * np.column_stack([x - x[ind], z_vals])

                # Representative values along z
                z_lin = mlinspace(-1, 1, nz)
                self.z = self.pos[ind, 1] * np.abs(z_lin) ** e * np.sign(z_lin)

                # Indices for different regions
                ind2 = (self.pos[:, 1] > 0) & (self.pos[:, 0] >= 0)
                ind3 = (self.pos[:, 1] == 0)
                ind4 = (self.pos[:, 1] < 0) & (self.pos[:, 0] >= 0)
                ind5 = (self.pos[:, 1] < 0) & (self.pos[:, 0] < 0)

                # Sharp upper edge
                if mode[0] != '0':
                    dz_shift = 0.5 * height - np.max(self.pos[ind2, 1])
                    if mode[0] == '1':
                        self.pos[ind2, 0] = np.max(self.pos[ind2, 0])
                    self.pos[ind2, 1] = self.pos[ind2, 1] + dz_shift
                    keep = ind2 | ind3 | ind4 | ind5
                    nan_row = np.array([[np.nan, 0.0]])
                    kept = self.pos[keep]
                    result = np.empty((kept.shape[0] + 1, 2), dtype = float)
                    result[:kept.shape[0]] = kept
                    result[kept.shape[0]:] = nan_row
                    self.pos = result
                    self.z[self.z > 0] = self.z[self.z > 0] + dz_shift

                # Recompute indices after potential modification
                ind1 = (self.pos[:, 1] > 0) & (self.pos[:, 0] < 0)
                ind2 = (self.pos[:, 1] > 0) & (self.pos[:, 0] >= 0)
                ind3 = (self.pos[:, 1] == 0)
                ind4 = (self.pos[:, 1] < 0) & (self.pos[:, 0] >= 0)

                # Sharp lower edge
                if mode[1] != '0':
                    dz_shift = 0.5 * height + np.min(self.pos[ind4, 1])
                    if mode[1] == '1':
                        self.pos[ind4, 0] = np.max(self.pos[ind4, 0])
                    self.pos[ind4, 1] = self.pos[ind4, 1] - dz_shift
                    keep = ind1 | ind2 | ind3 | ind4
                    nan_row = np.array([[np.nan, 0.0]])
                    kept = self.pos[keep]
                    result = np.empty((1 + kept.shape[0], 2), dtype = float)
                    result[0:1] = nan_row
                    result[1:] = kept
                    self.pos = result
                    self.z[self.z < 0] = self.z[self.z < 0] - dz_shift

        # Handle shift arguments
        dz_final = 0.0
        if max_z is not None:
            dz_final = max_z - np.nanmax(self.pos[:, 1])
        elif min_z is not None:
            dz_final = min_z - np.nanmin(self.pos[:, 1])
        elif center is not None:
            dz_final = center

        if dz_final != 0.0:
            self.pos[:, 1] = self.pos[:, 1] + dz_final
            self.z = self.z + dz_final

    @property
    def dmin(self) -> float:
        # MATLAB: subsref.m -- min d-value (column 0)
        return np.nanmin(self.pos[:, 0])

    @property
    def dmax(self) -> float:
        # MATLAB: subsref.m -- max d-value (column 0)
        return np.nanmax(self.pos[:, 0])

    @property
    def zmin(self) -> float:
        # MATLAB: subsref.m -- min z-value (column 1)
        return np.nanmin(self.pos[:, 1])

    @property
    def zmax(self) -> float:
        # MATLAB: subsref.m -- max z-value (column 1)
        return np.nanmax(self.pos[:, 1])

    def hshift(self, z: np.ndarray) -> np.ndarray:
        # MATLAB: @edgeprofile/hshift.m
        # Displace nodes at edges in horizontal direction
        if self.pos is None:
            return z

        z = np.asarray(z, dtype = float)

        # Filter out NaN entries
        valid = ~np.isnan(self.pos[:, 0]) & ~np.isnan(self.pos[:, 1])
        pos_valid = self.pos[valid]

        # Spline interpolation: given z-values, return d-values
        cs = CubicSpline(pos_valid[:, 1], pos_valid[:, 0])
        return cs(z)

    def vshift(self, z: float, d: np.ndarray) -> np.ndarray:
        # MATLAB: @edgeprofile/vshift.m
        # Displace nodes at edges in vertical direction
        if self.pos is None:
            return 0.0

        z = float(z)
        d = np.asarray(d, dtype = float)

        z_max = np.nanmax(self.pos[:, 1])
        z_min = np.nanmin(self.pos[:, 1])

        assert z == z_min or z == z_max, '[error] z must be zmin or zmax'

        if z == z_max:
            # Upper edge
            if np.isnan(self.pos[-1, 0]):
                return 0.0
            pos = self.pos[self._upper_indices()]
            # sort by d-values (column 0) for CubicSpline
            sort_idx = np.argsort(pos[:, 0])
            pos = pos[sort_idx]
            cs = CubicSpline(pos[:, 0], pos[:, 1])
            d_clamped = np.maximum(np.min(pos[:, 0]), -np.abs(d))
            return cs(d_clamped) - np.max(pos[:, 1])
        else:
            # Lower edge
            if np.isnan(self.pos[0, 0]):
                return 0.0
            pos = self.pos[self._lower_indices()]
            # sort by d-values (column 0) for CubicSpline
            sort_idx = np.argsort(pos[:, 0])
            pos = pos[sort_idx]
            cs = CubicSpline(pos[:, 0], pos[:, 1])
            d_clamped = np.maximum(np.min(pos[:, 0]), -np.abs(d))
            return cs(d_clamped) - np.min(pos[:, 1])

    def _upper_indices(self) -> np.ndarray:
        # MATLAB: upper() in vshift.m
        # Indices for upper edge: where dx < 0 at the end of the profile
        dx = np.diff(self.pos[:, 0])
        ind = dx < 0
        # Extend to same length as pos
        ind_full = np.empty(len(self.pos), dtype = bool)
        ind_full[:-1] = ind
        ind_full[-1] = ind[-1]
        # Find last element where ind differs from ind[-1], zero out everything before
        last_diff = -1
        for i in range(len(ind_full)):
            if ind_full[i] != ind_full[-1]:
                last_diff = i
        if last_diff >= 0:
            ind_full[:last_diff + 1] = False
        return ind_full

    def _lower_indices(self) -> np.ndarray:
        # MATLAB: lower() in vshift.m
        # Indices for lower edge: where dx > 0 at the start of the profile
        dx = np.diff(self.pos[:, 0])
        ind = dx > 0
        # Find first element where ind differs from ind[0], zero out everything after
        first_diff = len(ind)
        for i in range(len(ind)):
            if ind[i] != ind[0]:
                first_diff = i
                break
        ind_full = np.zeros(len(self.pos), dtype = bool)
        ind_full[:first_diff] = ind[:first_diff] if first_diff <= len(ind) else ind
        return ind_full
