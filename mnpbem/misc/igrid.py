"""
Interpolation grids for 2D and 3D.

MATLAB: @igrid2/, @igrid3/
"""

import numpy as np
from typing import Callable, Tuple


class IGrid2(object):
    """
    2D grid for bilinear interpolation.

    MATLAB: @igrid2

    Parameters
    ----------
    x : ndarray
        x-values of grid
    y : ndarray
        y-values of grid

    Methods
    -------
    finterp(x, y) -> Callable
    fderiv(x, y, direction) -> Callable
    """

    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = np.asarray(x).ravel()
        self.y = np.asarray(y).ravel()

    @property
    def size(self) -> Tuple[int, int]:
        return (len(self.x), len(self.y))

    @property
    def numel(self) -> int:
        return len(self.x) * len(self.y)

    def finterp(self, x: np.ndarray, y: np.ndarray) -> Callable:
        """
        MATLAB: @igrid2/finterp.m

        Interpolation function for points (x, y).
        Returns a callable that takes tabulated values V and returns
        interpolated values at (x, y).
        """
        x_shape = np.asarray(x).shape
        x_flat = np.asarray(x).ravel()
        y_flat = np.asarray(y).ravel()

        # index to grid positions
        ix = np.searchsorted(self.x, x_flat, side = 'right') - 1
        iy = np.searchsorted(self.y, y_flat, side = 'right') - 1

        # handle boundary cases
        ix = np.clip(ix, 0, len(self.x) - 2)
        iy = np.clip(iy, 0, len(self.y) - 2)

        # bin coordinates
        xx = (x_flat - self.x[ix]) / (self.x[ix + 1] - self.x[ix])
        yy = (y_flat - self.y[iy]) / (self.y[iy + 1] - self.y[iy])

        # size of interpolation array
        nx = len(self.x)
        ny = len(self.y)

        # interpolation indices (row-major flattening)
        ind = np.column_stack([
            (ix + 0) * ny + (iy + 0),
            (ix + 1) * ny + (iy + 0),
            (ix + 0) * ny + (iy + 1),
            (ix + 1) * ny + (iy + 1)])

        # interpolation weights
        w = np.column_stack([
            (1 - xx) * (1 - yy),
            xx * (1 - yy),
            (1 - xx) * yy,
            xx * yy])

        def interp_fn(v: np.ndarray) -> np.ndarray:
            v_flat = v.ravel()
            result = np.sum(w * v_flat[ind], axis = 1)
            return result.reshape(x_shape)

        return interp_fn

    def fderiv(self, x: np.ndarray, y: np.ndarray,
            direction: int) -> Callable:
        """
        MATLAB: @igrid2/fderiv.m

        Derivative function for points (x, y).
        direction: 1 for x-derivative, 2 for y-derivative.
        """
        x_shape = np.asarray(x).shape
        x_flat = np.asarray(x).ravel()
        y_flat = np.asarray(y).ravel()

        # index to grid positions
        ix = np.searchsorted(self.x, x_flat, side = 'right') - 1
        iy = np.searchsorted(self.y, y_flat, side = 'right') - 1

        ix = np.clip(ix, 0, len(self.x) - 2)
        iy = np.clip(iy, 0, len(self.y) - 2)

        # bin sizes and bin coordinates
        hx = self.x[ix + 1] - self.x[ix]
        hy = self.y[iy + 1] - self.y[iy]
        xx = (x_flat - self.x[ix]) / hx
        yy = (y_flat - self.y[iy]) / hy

        # size of interpolation array
        ny = len(self.y)

        # interpolation indices
        ind = np.column_stack([
            (ix + 0) * ny + (iy + 0),
            (ix + 1) * ny + (iy + 0),
            (ix + 0) * ny + (iy + 1),
            (ix + 1) * ny + (iy + 1)])

        # derivative of interpolation weights
        if direction == 1:
            inv_hx = 1.0 / hx
            w = np.column_stack([
                -(1 - yy) * inv_hx,
                (1 - yy) * inv_hx,
                -yy * inv_hx,
                yy * inv_hx])
        elif direction == 2:
            inv_hy = 1.0 / hy
            w = np.column_stack([
                -(1 - xx) * inv_hy,
                -xx * inv_hy,
                (1 - xx) * inv_hy,
                xx * inv_hy])
        else:
            raise ValueError('[error] Invalid <direction>!')

        def deriv_fn(v: np.ndarray) -> np.ndarray:
            v_flat = v.ravel()
            result = np.sum(w * v_flat[ind], axis = 1)
            return result.reshape(x_shape)

        return deriv_fn

    def __call__(self, x: np.ndarray, y: np.ndarray,
            v: np.ndarray) -> np.ndarray:
        """Perform interpolation for array V at points (x, y)."""
        return self.finterp(x, y)(v)

    def __repr__(self) -> str:
        return 'IGrid2(nx={}, ny={})'.format(len(self.x), len(self.y))


class IGrid3(object):
    """
    3D grid for trilinear interpolation.

    MATLAB: @igrid3

    Parameters
    ----------
    x : ndarray
        x-values of grid
    y : ndarray
        y-values of grid
    z : ndarray
        z-values of grid

    Methods
    -------
    finterp(x, y, z) -> Callable
    fderiv(x, y, z, direction) -> Callable
    """

    def __init__(self, x: np.ndarray, y: np.ndarray,
            z: np.ndarray) -> None:
        self.x = np.asarray(x).ravel()
        self.y = np.asarray(y).ravel()
        self.z = np.asarray(z).ravel()

    @property
    def size(self) -> Tuple[int, int, int]:
        return (len(self.x), len(self.y), len(self.z))

    @property
    def numel(self) -> int:
        return len(self.x) * len(self.y) * len(self.z)

    def finterp(self, x: np.ndarray, y: np.ndarray,
            z: np.ndarray) -> Callable:
        """
        MATLAB: @igrid3/finterp.m

        Interpolation function for points (x, y, z).
        Returns a callable that takes tabulated values V and returns
        interpolated values at (x, y, z).
        """
        x_shape = np.asarray(x).shape
        x_flat = np.asarray(x).ravel()
        y_flat = np.asarray(y).ravel()
        z_flat = np.asarray(z).ravel()

        # index to grid positions
        ix = np.searchsorted(self.x, x_flat, side = 'right') - 1
        iy = np.searchsorted(self.y, y_flat, side = 'right') - 1
        iz = np.searchsorted(self.z, z_flat, side = 'right') - 1

        ix = np.clip(ix, 0, len(self.x) - 2)
        iy = np.clip(iy, 0, len(self.y) - 2)
        iz = np.clip(iz, 0, len(self.z) - 2)

        # bin coordinates
        xx = (x_flat - self.x[ix]) / (self.x[ix + 1] - self.x[ix])
        yy = (y_flat - self.y[iy]) / (self.y[iy + 1] - self.y[iy])
        zz = (z_flat - self.z[iz]) / (self.z[iz + 1] - self.z[iz])

        # size for index computation
        ny = len(self.y)
        nz = len(self.z)

        # interpolation indices (row-major)
        def _idx(di: int, dj: int, dk: int) -> np.ndarray:
            return (ix + di) * ny * nz + (iy + dj) * nz + (iz + dk)

        ind = np.column_stack([
            _idx(0, 0, 0), _idx(1, 0, 0),
            _idx(0, 1, 0), _idx(1, 1, 0),
            _idx(0, 0, 1), _idx(1, 0, 1),
            _idx(0, 1, 1), _idx(1, 1, 1)])

        # interpolation weights
        w = np.column_stack([
            (1 - xx) * (1 - yy) * (1 - zz),
            xx * (1 - yy) * (1 - zz),
            (1 - xx) * yy * (1 - zz),
            xx * yy * (1 - zz),
            (1 - xx) * (1 - yy) * zz,
            xx * (1 - yy) * zz,
            (1 - xx) * yy * zz,
            xx * yy * zz])

        def interp_fn(v: np.ndarray) -> np.ndarray:
            v_flat = v.ravel()
            result = np.sum(w * v_flat[ind], axis = 1)
            return result.reshape(x_shape)

        return interp_fn

    def fderiv(self, x: np.ndarray, y: np.ndarray, z: np.ndarray,
            direction: int) -> Callable:
        """
        MATLAB: @igrid3/fderiv.m

        Derivative function for points (x, y, z).
        direction: 1 for x, 2 for y, 3 for z.
        """
        x_shape = np.asarray(x).shape
        x_flat = np.asarray(x).ravel()
        y_flat = np.asarray(y).ravel()
        z_flat = np.asarray(z).ravel()

        ix = np.searchsorted(self.x, x_flat, side = 'right') - 1
        iy = np.searchsorted(self.y, y_flat, side = 'right') - 1
        iz = np.searchsorted(self.z, z_flat, side = 'right') - 1

        ix = np.clip(ix, 0, len(self.x) - 2)
        iy = np.clip(iy, 0, len(self.y) - 2)
        iz = np.clip(iz, 0, len(self.z) - 2)

        hx = self.x[ix + 1] - self.x[ix]
        hy = self.y[iy + 1] - self.y[iy]
        hz = self.z[iz + 1] - self.z[iz]
        xx = (x_flat - self.x[ix]) / hx
        yy = (y_flat - self.y[iy]) / hy
        zz = (z_flat - self.z[iz]) / hz

        ny = len(self.y)
        nz = len(self.z)

        def _idx(di: int, dj: int, dk: int) -> np.ndarray:
            return (ix + di) * ny * nz + (iy + dj) * nz + (iz + dk)

        ind = np.column_stack([
            _idx(0, 0, 0), _idx(1, 0, 0),
            _idx(0, 1, 0), _idx(1, 1, 0),
            _idx(0, 0, 1), _idx(1, 0, 1),
            _idx(0, 1, 1), _idx(1, 1, 1)])

        if direction == 1:
            inv_hx = 1.0 / hx
            w = np.column_stack([
                -inv_hx * (1 - yy) * (1 - zz),
                inv_hx * (1 - yy) * (1 - zz),
                -inv_hx * yy * (1 - zz),
                inv_hx * yy * (1 - zz),
                -inv_hx * (1 - yy) * zz,
                inv_hx * (1 - yy) * zz,
                -inv_hx * yy * zz,
                inv_hx * yy * zz])
        elif direction == 2:
            inv_hy = 1.0 / hy
            w = np.column_stack([
                (1 - xx) * (-inv_hy) * (1 - zz),
                xx * (-inv_hy) * (1 - zz),
                (1 - xx) * inv_hy * (1 - zz),
                xx * inv_hy * (1 - zz),
                (1 - xx) * (-inv_hy) * zz,
                xx * (-inv_hy) * zz,
                (1 - xx) * inv_hy * zz,
                xx * inv_hy * zz])
        elif direction == 3:
            inv_hz = 1.0 / hz
            w = np.column_stack([
                (1 - xx) * (1 - yy) * (-inv_hz),
                xx * (1 - yy) * (-inv_hz),
                (1 - xx) * yy * (-inv_hz),
                xx * yy * (-inv_hz),
                (1 - xx) * (1 - yy) * inv_hz,
                xx * (1 - yy) * inv_hz,
                (1 - xx) * yy * inv_hz,
                xx * yy * inv_hz])
        else:
            raise ValueError('[error] Invalid <direction>!')

        def deriv_fn(v: np.ndarray) -> np.ndarray:
            v_flat = v.ravel()
            result = np.sum(w * v_flat[ind], axis = 1)
            return result.reshape(x_shape)

        return deriv_fn

    def __call__(self, x: np.ndarray, y: np.ndarray,
            z: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Perform interpolation for array V at points (x, y, z)."""
        return self.finterp(x, y, z)(v)

    def __repr__(self) -> str:
        return 'IGrid3(nx={}, ny={}, nz={})'.format(
            len(self.x), len(self.y), len(self.z))
