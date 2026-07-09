from __future__ import annotations

import copy

import numpy as np
from typing import Optional, List, Tuple, Any, Union

from ..utils.matlab_compat import matan2, msqrt, macos


class Point(object):
    """Single collection of points in space.

    MATLAB: Particles/@point/point.m

    Parameters
    ----------
    pos : ndarray, shape (n, 3)
        Coordinates of points
    nvec : ndarray, shape (n, 3), optional
        Normal vectors at positions (default: zeros)
    area : ndarray, shape (n,), optional
        Areas associated with points (default: ones)
    """

    def __init__(self,
            pos: np.ndarray,
            nvec: Optional[np.ndarray] = None,
            area: Optional[np.ndarray] = None) -> None:

        self.pos = np.atleast_2d(np.asarray(pos, dtype = np.float64))
        n = self.pos.shape[0]

        if nvec is not None:
            self.nvec = np.atleast_2d(np.asarray(nvec, dtype = np.float64))
        else:
            self.nvec = np.zeros((n, 3), dtype = np.float64)

        if area is not None:
            self.area = np.asarray(area, dtype = np.float64).ravel()
        else:
            self.area = np.ones(n, dtype = np.float64)

    @property
    def n(self) -> int:
        return self.pos.shape[0]

    @property
    def nfaces(self) -> int:
        return self.pos.shape[0]

    def select(self,
            index: Optional[np.ndarray] = None,
            carfun: Optional[Any] = None,
            polfun: Optional[Any] = None,
            sphfun: Optional[Any] = None) -> 'Point':
        if index is not None:
            idx = np.asarray(index)
            return Point(self.pos[idx], self.nvec[idx], self.area[idx])

        if carfun is not None:
            mask = carfun(self.pos[:, 0], self.pos[:, 1], self.pos[:, 2])
            mask = np.asarray(mask, dtype = bool)
            return Point(self.pos[mask], self.nvec[mask], self.area[mask])

        if polfun is not None:
            x, y, z = self.pos[:, 0], self.pos[:, 1], self.pos[:, 2]
            phi = matan2(y, x)
            r = msqrt(x ** 2 + y ** 2)
            mask = polfun(phi, r, z)
            mask = np.asarray(mask, dtype = bool)
            return Point(self.pos[mask], self.nvec[mask], self.area[mask])

        if sphfun is not None:
            x, y, z = self.pos[:, 0], self.pos[:, 1], self.pos[:, 2]
            r = msqrt(x ** 2 + y ** 2 + z ** 2)
            phi = matan2(y, x)
            theta = macos(np.clip(z / np.maximum(r, 1e-30), -1, 1))
            mask = sphfun(phi, theta, r)
            mask = np.asarray(mask, dtype = bool)
            return Point(self.pos[mask], self.nvec[mask], self.area[mask])

        return Point(self.pos.copy(), self.nvec.copy(), self.area.copy())

    def __add__(self, other: 'Point') -> 'Point':
        total = self.n + other.n
        pos = np.empty((total, 3), dtype = np.float64)
        nvec = np.empty((total, 3), dtype = np.float64)
        area = np.empty(total, dtype = np.float64)
        pos[:self.n] = self.pos
        pos[self.n:] = other.pos
        nvec[:self.n] = self.nvec
        nvec[self.n:] = other.nvec
        area[:self.n] = self.area
        area[self.n:] = other.area
        return Point(pos, nvec, area)

    def __repr__(self) -> str:
        return 'Point(n={})'.format(self.n)


class ComPoint(object):
    """Compound of points in a dielectric environment.

    MATLAB: Particles/@compoint/compoint.m

    Groups a set of positions into the dielectric media defined by a
    ComParticle object. The resulting ComPoint can then be used by
    CompGreen classes, dipole excitations, and field evaluations.

    Parameters
    ----------
    p : ComParticle
        ComParticle defining the dielectric environment
    pos : ndarray, shape (npos, 3)
        Positions of points
    mindist : float or ndarray, optional
        Minimum distance of points to particle boundary (default: 0)
    medium : int or list of int, optional
        Only keep points in selected media (1-indexed)
    layer : object, optional
        Layer structure for substrate simulations

    Attributes
    ----------
    eps : list
        Dielectric functions (from ComParticle)
    p : list of Point
        Point groups, one per medium
    inout : ndarray
        Medium index for each group (1-indexed, matching eps)
    mask : list of int
        Active group indices
    pos : ndarray, shape (n, 3)
        All active positions
    n : int
        Total number of active points
    nvec : ndarray, shape (n, 3)
        Normal vectors (zeros for bare points)
    area : ndarray, shape (n,)
        Areas (ones for bare points)
    """

    def __init__(self,
            p_or_eps: Any,
            pos_or_points: Any,
            inout_or_medium: Any = None,
            mindist: Union[float, np.ndarray, None] = None,
            medium: Optional[Any] = None,
            layer: Optional[Any] = None) -> None:

        # Detect construction mode:
        # Mode A: ComPoint(comparticle, pos_array, ...)
        # Mode B: ComPoint(eps_list, point_list, inout_array) -- direct construction
        if isinstance(pos_or_points, list) and len(pos_or_points) > 0 and isinstance(pos_or_points[0], Point):
            self._init_direct(p_or_eps, pos_or_points, inout_or_medium)
        elif isinstance(pos_or_points, list) and len(pos_or_points) > 0 and isinstance(pos_or_points[0], np.ndarray):
            self._init_from_pos_list(p_or_eps, pos_or_points, inout_or_medium)
        else:
            self._init_from_comparticle(p_or_eps, pos_or_points, mindist, medium, layer)

    def _init_direct(self,
            eps: List[Any],
            points: List[Point],
            inout: Any) -> None:
        self.eps = eps
        self.p = points
        self.inout = np.atleast_1d(np.asarray(inout, dtype = int))
        self._mask = list(range(len(self.p)))
        self._npos = sum(pt.n for pt in self.p)
        self._ind = None
        self._update_pc()

    def _init_from_pos_list(self,
            eps: List[Any],
            pos_list: List[np.ndarray],
            inout: Any) -> None:
        self.eps = eps
        self.p = [Point(pos) for pos in pos_list]
        self.inout = np.atleast_1d(np.asarray(inout, dtype = int))
        self._mask = list(range(len(self.p)))
        self._npos = sum(pt.n for pt in self.p)
        self._ind = None
        self._update_pc()

    def _init_from_comparticle(self,
            comparticle: Any,
            pos: np.ndarray,
            mindist: Union[float, np.ndarray, None] = None,
            medium: Optional[Any] = None,
            layer: Optional[Any] = None) -> None:

        pos = np.atleast_2d(np.asarray(pos, dtype = np.float64)).copy()
        self._npos = pos.shape[0]
        self.eps = comparticle.eps
        self._pin = comparticle
        self.layer = layer

        if mindist is None:
            mindist_val = 0.0
        else:
            mindist_val = mindist

        # Determine which medium each point is in by checking
        # distance to nearest particle boundary
        inout_per_point = self._classify_points(comparticle, pos, mindist_val)

        # Layer-aware shift: points sitting exactly on a layer interface are
        # nudged into the upper layer by 1e-8 so they have a well-defined
        # medium index. Mirrors MATLAB @compoint/init.m L47-L51 (only points
        # whose inout matches a layer.ind entry are eligible).
        if layer is not None and hasattr(layer, 'ind') and hasattr(layer, 'mindist'):
            layer_inds = np.atleast_1d(np.asarray(layer.ind, dtype=int)).ravel()
            eligible = np.isin(inout_per_point, layer_inds) & (inout_per_point > 0)
            if np.any(eligible):
                z_eligible = pos[eligible, 2]
                zmin_vals, _ = layer.mindist(z_eligible)
                on_interface = np.asarray(zmin_vals).ravel() < 1e-10
                if np.any(on_interface):
                    eligible_idx = np.where(eligible)[0]
                    shift_idx = eligible_idx[on_interface]
                    pos[shift_idx, 2] = pos[shift_idx, 2] + 1e-8
                    # Re-assign inout from the new z via indlayer -> layer.ind
                    eligible_z = pos[eligible_idx, 2]
                    new_layer_pos, _ = layer.indlayer(eligible_z)
                    new_inout = layer_inds[np.asarray(new_layer_pos).ravel() - 1]
                    inout_per_point[eligible_idx] = new_inout

        # Group points by medium
        unique_media = np.unique(inout_per_point[inout_per_point > 0])

        points = []
        inout_list = []
        ind_list = []

        for med in unique_media:
            idx = np.where(inout_per_point == med)[0]
            if len(idx) > 0:
                points.append(Point(pos[idx]))
                inout_list.append(med)
                ind_list.append(idx)

        self.p = points
        self.inout = np.array(inout_list, dtype = int) if len(inout_list) > 0 else np.array([], dtype = int)
        self._ind = ind_list

        # Apply medium mask
        if medium is not None:
            if np.isscalar(medium):
                medium = [medium]
            self._mask = []
            for i, io in enumerate(self.inout):
                if io in medium:
                    self._mask.append(i)
        else:
            self._mask = list(range(len(self.p)))

        self._update_pc()

    def _classify_points(self,
            comparticle: Any,
            pos: np.ndarray,
            mindist: Union[float, np.ndarray]) -> np.ndarray:
        npts = pos.shape[0]
        inout_per_point = np.ones(npts, dtype = int)

        if not hasattr(comparticle, 'p') or len(comparticle.p) == 0:
            return inout_per_point

        # For each point, find signed distance to nearest face
        for ip, particle in enumerate(comparticle.p):
            if not hasattr(particle, 'pos') or particle.pos.shape[0] == 0:
                continue

            # Compute distance from each point to nearest face centroid
            face_pos = particle.pos  # (nfaces, 3)
            for j in range(npts):
                diff = face_pos - pos[j]
                dists = msqrt(np.sum(diff ** 2, axis = 1))
                imin = np.argmin(dists)
                r = dists[imin]

                if r < np.abs(mindist) if np.isscalar(mindist) else np.abs(mindist[ip]):
                    # Too close: mark as invalid
                    inout_per_point[j] = 0
                    continue

                # Determine inside/outside using normal vector
                if hasattr(particle, 'nvec') and particle.nvec.shape[0] > 0:
                    nvec = particle.nvec[imin]
                    direction = pos[j] - face_pos[imin]
                    sign = np.dot(direction, nvec)
                    if sign >= 0:
                        # outside
                        inout_val = int(comparticle.inout[ip, 1])
                    else:
                        # inside
                        inout_val = int(comparticle.inout[ip, 0])
                    inout_per_point[j] = inout_val

        return inout_per_point

    def _update_pc(self) -> None:
        active = [self.p[i] for i in self._mask]
        if len(active) == 0:
            self._pc = Point(np.zeros((0, 3)))
        elif len(active) == 1:
            self._pc = active[0]
        else:
            result = active[0]
            for pt in active[1:]:
                result = result + pt
            self._pc = result

    # -- compound-compatible properties --

    @property
    def pos(self) -> np.ndarray:
        return self._pc.pos

    @property
    def n(self) -> int:
        return self._pc.n

    @property
    def nvec(self) -> np.ndarray:
        return self._pc.nvec

    @property
    def area(self) -> np.ndarray:
        return self._pc.area

    @property
    def nfaces(self) -> int:
        return self._pc.n

    @property
    def index(self) -> List[np.ndarray]:
        idx_list = []
        offset = 0
        for i in self._mask:
            npts = self.p[i].n
            idx_list.append(np.arange(offset, offset + npts, dtype = int))
            offset += npts
        return idx_list

    @property
    def np(self) -> int:
        return len(self._mask)

    def eps1(self,
            enei: float) -> np.ndarray:
        eps_vals = np.zeros(self.n, dtype = complex)
        offset = 0
        for i in self._mask:
            npts = self.p[i].n
            eps_idx = int(self.inout[i]) - 1  # 1-indexed to 0-indexed
            eps_func = self.eps[eps_idx]
            result = eps_func(enei)
            if isinstance(result, tuple):
                eps_val = result[0]
            else:
                eps_val = result
            eps_val = complex(np.asarray(eps_val).flat[0])
            eps_vals[offset:offset + npts] = eps_val
            offset += npts
        return eps_vals

    def eps2(self,
            enei: float) -> np.ndarray:
        """Outside dielectric constants at given wavelength.

        For ComPoint, observation points sit in a single medium
        (no inside/outside distinction), so eps2 == eps1.
        """
        return self.eps1(enei)

    def index_func(self,
            particle_indices: Any) -> np.ndarray:
        """Point indices for given group indices (1-indexed).

        Mirrors ComParticle.index_func: returns the indices into
        the concatenated pos array for the requested point groups.
        """
        if np.isscalar(particle_indices):
            particle_indices = [particle_indices]

        point_indices: List[int] = []
        offset = 0
        for i, pt in enumerate(self.p):
            if (i + 1) in particle_indices:
                point_indices.extend(range(offset, offset + pt.n))
            offset += pt.n
        return np.array(point_indices, dtype = int)

    def closedparticle(self,
            ind: int) -> Tuple[None, None, None]:
        return None, None, None

    def flip(self,
            direction: Union[int, List[int]]) -> 'ComPoint':
        obj = copy.deepcopy(self)
        if np.isscalar(direction):
            direction = [direction]
        for d in direction:
            # d is 1-indexed (1=x, 2=y, 3=z) following MATLAB convention
            col = d - 1
            for pt in obj.p:
                pt.pos[:, col] = -pt.pos[:, col]
        obj._update_pc()
        return obj

    def select(self,
            index: Optional[np.ndarray] = None,
            carfun: Optional[Any] = None,
            polfun: Optional[Any] = None,
            sphfun: Optional[Any] = None) -> 'ComPoint':
        obj = copy.deepcopy(self)

        if index is not None:
            # Global index selection
            idx = np.asarray(index)
            # Build global-to-group mapping
            ipt = []
            local_idx = []
            for i in range(len(obj.p)):
                for j in range(obj.p[i].n):
                    ipt.append(i)
                    local_idx.append(j)
            ipt = np.array(ipt)
            local_idx = np.array(local_idx)

            sel_ipt = ipt[idx]
            sel_local = local_idx[idx]

            for i in range(len(obj.p)):
                mask = sel_ipt == i
                if np.any(mask):
                    obj.p[i] = obj.p[i].select(index = sel_local[mask])
                else:
                    obj.p[i] = Point(np.zeros((0, 3)))
        else:
            kwargs = {}
            if carfun is not None:
                kwargs['carfun'] = carfun
            elif polfun is not None:
                kwargs['polfun'] = polfun
            elif sphfun is not None:
                kwargs['sphfun'] = sphfun
            obj.p = [pt.select(**kwargs) for pt in obj.p]

        # Remove empty groups
        nonempty = [i for i, pt in enumerate(obj.p) if pt.n > 0]
        obj.p = [obj.p[i] for i in nonempty]
        obj.inout = obj.inout[nonempty] if len(nonempty) > 0 else np.array([], dtype = int)
        obj._mask = list(range(len(obj.p)))
        obj._update_pc()
        return obj

    def __call__(self,
            valpt: np.ndarray,
            valdef: float = float('nan')) -> np.ndarray:
        siz = valpt.shape
        dtype = valpt.dtype if hasattr(valpt, 'dtype') else np.float64
        if np.isnan(valdef):
            val = np.full((self._npos,) + siz[1:], np.nan, dtype=dtype)
        else:
            val = np.full((self._npos,) + siz[1:], valdef, dtype=dtype)

        offset = 0
        for i in self._mask:
            npts = self.p[i].n
            if self._ind is not None:
                orig_idx = self._ind[i]
            else:
                orig_idx = np.arange(offset, offset + npts)
            val[orig_idx] = valpt[offset:offset + npts]
            offset += npts

        return val

    def __repr__(self) -> str:
        return 'ComPoint(n={}, groups={})'.format(self.n, len(self._mask))
