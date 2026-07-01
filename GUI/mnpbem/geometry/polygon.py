import numpy as np
from typing import Tuple, Optional, Union, List, Any, Dict
from matplotlib.path import Path

from ..utils.matlab_compat import mlinspace, mcos, msin, matan2, macos, msqrt


class Polygon(object):

    def __init__(self,
            n_or_verts: Union[int, np.ndarray],
            mode: str = 'size',
            dir: int = 1,
            sym: Optional[str] = None,
            size: Optional[Union[float, np.ndarray, List[float], Tuple[float, float]]] = None):

        # MATLAB @polygon/polygon.m + private/init.m
        if isinstance(n_or_verts, (int, np.integer)):
            n = int(n_or_verts)
            phi = np.arange(n) / n * 2 * np.pi + np.pi / n
            self.pos = np.column_stack([mcos(phi), msin(phi)])
        else:
            self.pos = np.asarray(n_or_verts, dtype = float).copy()
            if self.pos.ndim == 1:
                self.pos = self.pos.reshape(-1, 2)

        self.dir = dir
        self.sym = sym

        if size is not None:
            size_arr = np.atleast_1d(np.asarray(size, dtype = float))
            if len(size_arr) == 1:
                size_arr = np.array([size_arr[0], size_arr[0]])

            x_range = np.max(self.pos[:, 0]) - np.min(self.pos[:, 0])
            y_range = np.max(self.pos[:, 1]) - np.min(self.pos[:, 1])

            if x_range > 0:
                self.pos[:, 0] = size_arr[0] / x_range * self.pos[:, 0]
            if y_range > 0:
                self.pos[:, 1] = size_arr[-1] / y_range * self.pos[:, 1]

        if self.sym is not None:
            self._apply_symmetry(self.sym)

    def __repr__(self) -> str:
        return 'Polygon(n_verts={}, dir={}, sym={})'.format(
            self.pos.shape[0], self.dir, self.sym)

    @property
    def size_(self) -> np.ndarray:
        x_range = np.max(self.pos[:, 0]) - np.min(self.pos[:, 0])
        y_range = np.max(self.pos[:, 1]) - np.min(self.pos[:, 1])
        return np.array([x_range, y_range])

    @property
    def n_verts(self) -> int:
        return self.pos.shape[0]

    def close(self) -> 'Polygon':
        # MATLAB @polygon/close.m
        if self.sym is None or self.sym != 'xy':
            return self

        self.sort_()
        pos = self.pos

        if (not np.all(np.abs(pos[-1, :]) < 1e-6) and
                abs(np.prod(pos[0, :])) < 1e-6 and
                abs(np.prod(pos[-1, :])) < 1e-6):
            self.pos = np.vstack([pos, np.array([[0.0, 0.0]])])

        return self

    def round_(self,
            rad: Optional[float] = None,
            nrad: int = 5,
            edge: Optional[np.ndarray] = None) -> 'Polygon':

        # MATLAB @polygon/round.m
        pos = self.pos.copy()
        n = pos.shape[0]

        if rad is None:
            rad = 0.1 * np.max(np.abs(pos))

        if edge is None:
            edge = np.arange(n)

        def _norm_vec(v: np.ndarray) -> np.ndarray:
            lengths = msqrt(np.sum(v ** 2, axis = 1, keepdims = True))
            lengths[lengths < np.finfo(float).eps] = 1.0
            return v / lengths

        # edge direction vectors
        next_pos = np.roll(pos, -1, axis = 0)
        vec = _norm_vec(next_pos - pos)
        prev_vec = np.roll(vec, 1, axis = 0)
        dir_vec = _norm_vec(vec - prev_vec)

        # half angle between consecutive edges
        dot_prod = np.sum(prev_vec * vec, axis = 1)
        dot_prod = np.clip(dot_prod, -1.0, 1.0)
        beta = macos(dot_prod) / 2.0

        # center of rounding circles
        cos_beta = mcos(beta)
        cos_beta[cos_beta < np.finfo(float).eps] = 1.0
        zero = pos + rad * dir_vec / cos_beta[:, np.newaxis]

        # check which centers are inside polygon
        poly_path = Path(np.vstack([pos, pos[0]]))
        sgn = np.where(poly_path.contains_points(zero), 1, -1)

        # build new positions
        new_pos_list = []
        for i in range(n):
            if i not in edge:
                new_pos_list.append(pos[i:i + 1])
            else:
                if abs(beta[i]) < 1e-3:
                    angles = np.array([0.0])
                else:
                    angles = beta[i] * mlinspace(-1, 1, nrad) * sgn[i]

                for phi in angles:
                    c = mcos(phi)
                    s = msin(phi)
                    rot_mat = np.array([[c, s], [-s, c]])
                    pt = zero[i] - rad * dir_vec[i] @ rot_mat
                    new_pos_list.append(pt.reshape(1, 2))

        self.pos = np.vstack(new_pos_list)
        return self

    def rot(self, angle: float) -> 'Polygon':
        # MATLAB @polygon/rot.m - rotate by angle in degrees
        angle_rad = angle / 180.0 * np.pi
        c = mcos(angle_rad)
        s = msin(angle_rad)
        rot_mat = np.array([[c, s], [-s, c]])
        self.pos = self.pos @ rot_mat
        return self

    def scale(self, factor: Union[float, np.ndarray, List[float], Tuple[float, float]]) -> 'Polygon':
        # MATLAB @polygon/scale.m
        factor_arr = np.atleast_1d(np.asarray(factor, dtype = float))
        if len(factor_arr) == 1:
            factor_arr = np.array([factor_arr[0], factor_arr[0]])
        self.pos = self.pos * factor_arr
        return self

    def shift(self, vec: Union[np.ndarray, List[float], Tuple[float, float]]) -> 'Polygon':
        # MATLAB @polygon/shift.m
        vec_arr = np.asarray(vec, dtype = float)
        self.pos = self.pos + vec_arr
        return self

    def shiftbnd(self,
            dist: float,
            return_dist: bool = False) -> Union['Polygon', Tuple['Polygon', np.ndarray]]:
        # MATLAB @polygon/shiftbnd.m - shift boundary along normals
        nvec = self.compute_normals()
        nvec = np.sign(dist) * nvec

        x = self.pos[:, 0]
        y = self.pos[:, 1]
        nx = nvec[:, 0]
        ny = nvec[:, 1]

        n = len(x)
        inner_prod = nx[:, np.newaxis] * ny[np.newaxis, :] - ny[:, np.newaxis] * nx[np.newaxis, :]
        inner_prod[np.abs(inner_prod) < 1e-10] = 0

        # distance to crossing points
        lam = np.full((n, n), 1e10)
        for i in range(n):
            for j in range(n):
                if abs(inner_prod[i, j]) > 0:
                    lam[i, j] = (x[j] * ny[j] - y[j] * nx[j] - (x[i] * ny[j] - y[i] * nx[j])) / inner_prod[i, j]

        # discard negative and self-crossings
        lam[(np.isnan(lam)) | (lam < 0) | (lam * lam.T < 0)] = 1e10

        # take max of symmetric pairs
        a = np.maximum(lam, lam.T)
        mask = lam != 1e10
        lam[mask] = a[mask]

        lam_min = np.min(0.8 * lam, axis = 1)
        lam_min[lam_min > abs(dist)] = abs(dist)

        self.pos = self.pos + np.column_stack([lam_min * nx, lam_min * ny])

        if return_dist:
            distp = np.sign(dist) * lam_min
            return self, distp
        return self

    def midpoints(self, mode: str = 'add') -> 'Polygon':
        # MATLAB @polygon/midpoints.m - add midpoints for smooth polygon
        from scipy.interpolate import CubicSpline

        if mode == 'same':
            # positions already include midpoints, smooth only
            pos_closed = np.empty((len(self.pos[::2]) + 1, 2))
            pos_closed[:len(self.pos[::2])] = self.pos[::2]
            pos_closed[-1] = self.pos[0]
        else:
            pos_closed = np.empty((self.pos.shape[0] + 1, 2))
            pos_closed[:self.pos.shape[0]] = self.pos
            pos_closed[-1] = self.pos[0]

        n = pos_closed.shape[0] - 1

        # arc length of polygon segments
        diffs = pos_closed[1:] - pos_closed[:-1]
        seg_len = msqrt(np.sum(diffs ** 2, axis = 1))

        # cumulative arc length
        x = np.empty(n + 1)
        x[0] = 0.0
        x[1:] = np.cumsum(seg_len)

        # midpoint arc length values
        xi = 0.5 * (x[:-1] + x[1:])

        # spline interpolation for each coordinate
        cs_x = CubicSpline(x, pos_closed[:, 0])
        cs_y = CubicSpline(x, pos_closed[:, 1])
        posi = np.column_stack([cs_x(xi), cs_y(xi)])

        # interleave original and interpolated positions
        new_pos = np.empty((2 * n, 2))
        new_pos[0::2] = pos_closed[:-1]
        new_pos[1::2] = posi

        self.pos = new_pos
        return self

    def flip(self, axis: int = 0) -> 'Polygon':
        # MATLAB @polygon/flip.m - flip along axis (0=x, 1=y)
        self.pos[:, axis] = -self.pos[:, axis]
        return self

    def compute_normals(self) -> np.ndarray:
        # MATLAB @polygon/norm.m
        pos = self.pos

        def _unit(v: np.ndarray) -> np.ndarray:
            lengths = msqrt(np.sum(v ** 2, axis = 1, keepdims = True))
            lengths[lengths < np.finfo(float).eps] = 1.0
            return v / lengths

        # edge vectors
        next_pos = np.roll(pos, -1, axis = 0)
        vec = next_pos - pos

        # outward normals (perpendicular to edge)
        nvec = np.column_stack([-vec[:, 1], vec[:, 0]])
        nvec = _unit(nvec)

        # interpolate to vertex positions
        nvec_prev = np.roll(nvec, 1, axis = 0)
        nvec = (nvec + nvec_prev) / 2
        nvec = _unit(nvec)

        # check direction
        posp = pos + 1e-6 * nvec
        poly_path = Path(np.vstack([pos, pos[0]]))
        inside = poly_path.contains_points(posp)

        if self.dir == 1:
            nvec[inside] = -nvec[inside]
        else:
            nvec[~inside] = -nvec[~inside]

        # fix normals at symmetry points
        if self.sym is not None:
            if self.sym in ('x', 'xy'):
                nvec[np.abs(pos[:, 0]) < 1e-10, 0] = 0
            if self.sym in ('y', 'xy'):
                nvec[np.abs(pos[:, 1]) < 1e-10, 1] = 0
            nvec = _unit(nvec)

        return nvec

    def dist(self, pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # MATLAB @polygon/dist.m - distance from points to polygon
        pts = np.asarray(pts, dtype = float)
        if pts.ndim == 1:
            pts = pts.reshape(1, -1)

        npts = pts.shape[0]
        dmin = np.full(npts, 1e10)
        imin = np.zeros(npts, dtype = int)

        pos = self.pos
        xa = pos[:, 0]
        ya = pos[:, 1]
        xb = np.roll(xa, -1)
        yb = np.roll(ya, -1)

        for j in range(npts):
            x = pts[j, 0]
            y = pts[j, 1]

            lam = ((xb - xa) * (x - xa) + (yb - ya) * (y - ya)) / \
                  ((xb - xa) ** 2 + (yb - ya) ** 2 + np.finfo(float).eps)
            lam = np.clip(lam, 0, 1)

            d = msqrt((xa + lam * (xb - xa) - x) ** 2 +
                        (ya + lam * (yb - ya) - y) ** 2)
            idx = np.argmin(d)
            if d[idx] < dmin[j]:
                dmin[j] = d[idx]
                imin[j] = idx

        return dmin, imin

    def sort_(self) -> 'Polygon':
        # MATLAB @polygon/sort.m
        if self.sym is None:
            return self

        ind = []
        if self.sym in ('x', 'xy'):
            idx_x = np.where(np.abs(self.pos[:, 0]) < 1e-6)[0]
            ind.extend(idx_x.tolist())
        if self.sym in ('y', 'xy'):
            idx_y = np.where(np.abs(self.pos[:, 1]) < 1e-6)[0]
            ind = list(set(ind + idx_y.tolist()))

        ind.sort()

        if len(ind) >= 2 and ind[0] != 0:
            shift = self.pos.shape[0] - ind[-1]
            self.pos = np.roll(self.pos, shift, axis = 0)

        return self

    def _apply_symmetry(self, sym: str) -> None:
        # MATLAB @polygon/symmetry.m - reduce to irreducible part
        if sym is None:
            return

        def _inside(pos_check: np.ndarray, sym_key: str) -> np.ndarray:
            if sym_key == 'x':
                return pos_check[:, 0] >= 0
            elif sym_key == 'y':
                return pos_check[:, 1] >= 0
            elif sym_key == 'xy':
                return (pos_check[:, 0] >= 0) & (pos_check[:, 1] >= 0)
            return np.ones(pos_check.shape[0], dtype = bool)

        def _intersect(posa: np.ndarray, posb: np.ndarray, sym_key: str) -> np.ndarray:
            xa, ya = posa[0], posa[1]
            xb, yb = posb[0], posb[1]

            if sym_key == 'x':
                px = 0.0
                py = ya - xa * (yb - ya) / (xb - xa + np.finfo(float).eps)
                return np.array([px, py])
            elif sym_key == 'y':
                px = xa - ya * (xb - xa) / (yb - ya + np.finfo(float).eps)
                py = 0.0
                return np.array([px, py])
            elif sym_key == 'xy':
                if xa * xb <= 0:
                    return _intersect(posa, posb, 'x')
                elif ya * yb <= 0:
                    return _intersect(posa, posb, 'y')
                return np.array([0.0, 0.0])
            return posa

        issame = lambda a, b: np.all(np.abs(a - b) < 1e-8)

        # close polygon and round near-zero
        pos = np.vstack([self.pos, self.pos[0]])
        pos[np.abs(pos[:, 0]) < 1e-8, 0] = 0
        pos[np.abs(pos[:, 1]) < 1e-8, 1] = 0

        in_mask = _inside(pos, sym)
        first = -1
        for i in range(len(in_mask)):
            if in_mask[i]:
                first = i
                break

        if first < 0:
            return

        if first == 0:
            sympos = [pos[first].copy()]
        else:
            sympos = [_intersect(pos[first - 1], pos[first], sym)]

        for i in range(first + 1, pos.shape[0]):
            if in_mask[i - 1] and in_mask[i]:
                sympos.append(pos[i].copy())
            elif in_mask[i - 1] != in_mask[i]:
                posi = _intersect(pos[i - 1], pos[i], sym)
                if not issame(posi, sympos[-1]):
                    sympos.append(posi)
                if sym == 'xy' and not in_mask[i]:
                    sympos.append(np.array([0.0, 0.0]))
                if in_mask[i] and not issame(posi, pos[i]):
                    sympos.append(pos[i].copy())

        sympos = np.array(sympos)

        # remove duplicate first/last
        if len(sympos) > 1 and issame(sympos[0], sympos[-1]):
            sympos = sympos[:-1]

        self.pos = sympos
        self.sym = sym
        self.sort_()

        # remove origin for xy-symmetry
        if sym == 'xy' and len(self.pos) > 0 and np.all(self.pos[-1] == 0):
            if not np.all(np.any(self.pos[:-1] == 0, axis = 0)):
                self.pos = self.pos[:-1]

    def get_full_polygon(self) -> np.ndarray:
        # reconstruct full polygon from irreducible part
        if self.sym is None:
            return self.pos.copy()

        pos = self.pos.copy()

        # remove origin if present
        if self.sym == 'xy' and len(pos) > 0 and np.all(pos[-1] == 0):
            pos = pos[:-1]

        if self.sym in ('x', 'xy'):
            if np.any(pos[:, 0] == 0):
                flipped = pos[1:-1][::-1] * np.array([-1, 1])
                total = pos.shape[0] + flipped.shape[0]
                full = np.empty((total, 2))
                full[:pos.shape[0]] = pos
                full[pos.shape[0]:] = flipped
                pos = full

        if self.sym in ('y', 'xy'):
            if np.any(pos[:, 1] == 0):
                flipped = pos[1:-1][::-1] * np.array([1, -1])
                total = pos.shape[0] + flipped.shape[0]
                full = np.empty((total, 2))
                full[:pos.shape[0]] = pos
                full[pos.shape[0]:] = flipped
                pos = full

        return pos

    def _union_parts(self, *others: 'Polygon') -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
        # Internal helper: concatenate polygons and return upos, unet plus
        # one edge-index array per input polygon (one "face" per loop).
        all_polys = [self] + list(others)
        upos = np.empty((0, 2))
        unet = np.empty((0, 2), dtype = int)
        face_list: List[np.ndarray] = []

        for poly in all_polys:
            pos = poly.pos
            n = pos.shape[0]
            # edges: 0-1, 1-2, ..., (n-2)-(n-1), (n-1)-0
            net = np.column_stack([np.arange(n), np.roll(np.arange(n), -1)])

            net_shifted = net + upos.shape[0]

            edge_start = unet.shape[0]
            total_net = unet.shape[0] + net_shifted.shape[0]
            unet_new = np.empty((total_net, 2), dtype = int)
            unet_new[:unet.shape[0]] = unet
            unet_new[unet.shape[0]:] = net_shifted
            unet = unet_new

            total_pos = upos.shape[0] + pos.shape[0]
            upos_new = np.empty((total_pos, 2))
            upos_new[:upos.shape[0]] = upos
            upos_new[upos.shape[0]:] = pos
            upos = upos_new

            face_list.append(np.arange(edge_start, edge_start + n, dtype = int))

        return upos, unet, face_list

    def union(self, *others: 'Polygon') -> Tuple[np.ndarray, np.ndarray]:
        # MATLAB @polygon/union.m - combine polygons for mesh2d.
        # Returns (upos, unet). For hole-aware meshing use
        # :meth:`union_faces` instead, which also returns a list of
        # edge-index arrays (one per input polygon) suitable for passing as
        # ``face=...`` to :func:`mesh2d`.
        upos, unet, _ = self._union_parts(*others)
        return upos, unet

    def union_faces(self, *others: 'Polygon') -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
        # Same as :meth:`union` but also returns ``face_list``: a list where
        # ``face_list[k]`` is the array of edge indices belonging to loop k.
        # The caller can pass this straight to ``mesh2d(..., face=face_list)``
        # to mesh an outer polygon with holes (MATLAB plate() semantics).
        return self._union_parts(*others)

    def polymesh2d(self,
            *others: 'Polygon',
            hdata: Optional[Dict[str, Any]] = None,
            options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, np.ndarray]:

        # MATLAB @polygon/polymesh2d.m
        # When called with extra Polygon arguments, treats them as additional
        # loops (typically holes inside self). Each polygon becomes one face
        # in the mesh2d call; mesh2d's face-classification auto-detects
        # outer / hole topology.
        from .mesh2d import mesh2d

        if options is None:
            options = {'output': False}
        elif 'output' not in options:
            options['output'] = False

        pos, cnet, face_list = self.union_faces(*others)

        verts, faces = mesh2d(pos, cnet, hdata = hdata, options = options, face = face_list)
        return verts, faces

    def copy(self) -> 'Polygon':
        new_poly = Polygon.__new__(Polygon)
        new_poly.pos = self.pos.copy()
        new_poly.dir = self.dir
        new_poly.sym = self.sym
        return new_poly

    def norm(self) -> np.ndarray:
        # MATLAB @polygon/norm.m - alias for compute_normals
        return self.compute_normals()

    def plot(self,
            line_spec: str = 'b',
            nvec: bool = False,
            scale: float = 1.0,
            ax: Optional[Any] = None) -> Any:
        # MATLAB @polygon/plot.m - plot polygon (and optional normals)
        import matplotlib.pyplot as plt

        if ax is None:
            ax = plt.gca()

        pos = self.pos
        n = pos.shape[0]

        closed = np.empty((n + 1, 2))
        closed[:n] = pos
        closed[n] = pos[0]
        ax.plot(closed[:, 0], closed[:, 1], line_spec)

        if nvec:
            nv = self.compute_normals()
            ax.quiver(pos[:, 0], pos[:, 1], nv[:, 0], nv[:, 1],
                    angles = 'xy', scale_units = 'xy', scale = 1.0 / scale)

        ax.set_aspect('equal', adjustable = 'datalim')
        return ax

    def symmetry(self, sym: Optional[str] = None) -> Tuple['Polygon', 'Polygon']:
        # MATLAB @polygon/symmetry.m
        # Returns (obj, full): irreducible part and symmetrized full polygon.
        full = self.copy()
        if sym is None or sym == '':
            return self, full

        new_poly = self.copy()
        new_poly.sym = None
        new_poly._apply_symmetry(sym)

        # full polygon: mirror irreducible part along symmetry axes
        full = new_poly.copy()
        full.sym = None

        if sym == 'xy' and len(full.pos) > 0 and np.all(full.pos[-1] == 0):
            full.pos = full.pos[:-1]

        if sym in ('x', 'xy') and np.any(full.pos[:, 0] == 0):
            mid = full.pos[1:-1]
            flipped = mid[::-1] * np.array([-1.0, 1.0])
            total = full.pos.shape[0] + flipped.shape[0]
            out = np.empty((total, 2))
            out[:full.pos.shape[0]] = full.pos
            out[full.pos.shape[0]:] = flipped
            full.pos = out

        if sym in ('y', 'xy') and np.any(full.pos[:, 1] == 0):
            mid = full.pos[1:-1]
            flipped = mid[::-1] * np.array([1.0, -1.0])
            total = full.pos.shape[0] + flipped.shape[0]
            out = np.empty((total, 2))
            out[:full.pos.shape[0]] = full.pos
            out[full.pos.shape[0]:] = flipped
            full.pos = out

        return new_poly, full

    def interp1(self, pos: np.ndarray) -> 'Polygon':
        """
        Make new polygon through given positions using interpolation.

        MATLAB: @polygon/interp1.m

        Finds all points in `pos` that lie on the polygon boundary
        (within tolerance 1e-6) and creates a new polygon from those points,
        ordered along the boundary.

        Parameters
        ----------
        pos : ndarray, shape (n, 2)
            Candidate positions (e.g. mesh vertices)

        Returns
        -------
        self : Polygon
            Modified polygon with positions from `pos` on boundary
        """
        pos = np.asarray(pos, dtype = float)
        if pos.ndim == 1:
            pos = pos.reshape(1, -1)

        # Find points that are on the polygon boundary (distance < 1e-6)
        d, inst = self.dist(pos)

        on_boundary = np.abs(d) < 1e-6
        ipos = np.where(on_boundary)[0]
        inst_sel = inst[on_boundary]

        if len(ipos) == 0:
            return self

        # Compute parameter along polygon boundary for ordering
        # For each boundary point, compute its position along the polygon
        # as segment_index + fraction_along_segment
        poly_pos = self.pos
        n_seg = len(poly_pos)
        xa, ya = poly_pos[:, 0], poly_pos[:, 1]
        xb = np.roll(xa, -1)
        yb = np.roll(ya, -1)

        param = np.empty(len(ipos))
        for k in range(len(ipos)):
            seg = inst_sel[k]
            px, py = pos[ipos[k], 0], pos[ipos[k], 1]
            # Fraction along segment
            dx = xb[seg] - xa[seg]
            dy = yb[seg] - ya[seg]
            seg_len2 = dx * dx + dy * dy
            if seg_len2 > 0:
                frac = ((px - xa[seg]) * dx + (py - ya[seg]) * dy) / seg_len2
                frac = np.clip(frac, 0.0, 1.0)
            else:
                frac = 0.0
            param[k] = seg + frac

        # Sort by parameter along boundary
        sort_idx = np.argsort(param)
        ipos = ipos[sort_idx]

        # Create new polygon from these positions
        self.pos = pos[ipos].copy()
        return self
