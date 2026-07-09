import numpy as np
from typing import Tuple, Optional, Union, List, Any
from ..utils.matlab_compat import mround
from .polygon import Polygon
from .edgeprofile import EdgeProfile
from .particle import Particle
from .mesh_generators import fvgrid, _add_midpoints_flat


class Polygon3(object):

    # MATLAB: @polygon3/polygon3.m + @polygon3/init.m

    def __init__(self,
            poly: Polygon,
            z: float,
            edge: Optional[EdgeProfile] = None,
            refun: Optional[Any] = None):

        self.poly = poly.copy()
        self.z = z
        self.edge = edge if edge is not None else EdgeProfile()
        self._refun = refun

    def __repr__(self) -> str:
        return 'Polygon3(z={}, poly={}, edge={})'.format(self.z, self.poly, self.edge)

    def set(self, **kwargs) -> 'Polygon3':
        # MATLAB: @polygon3/set.m
        for key, val in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, val)
            elif hasattr(self.poly, key):
                setattr(self.poly, key, val)
        return self

    def flip(self, axis: int) -> 'Polygon3':
        # MATLAB: @polygon3/flip.m
        self.poly = self.poly.flip(axis)
        return self

    def shift(self, vec: np.ndarray) -> 'Polygon3':
        # MATLAB: @polygon3/shift.m
        vec = np.asarray(vec, dtype = float)
        self.poly = self.poly.shift(vec[:2])
        self.z = self.z + vec[2]
        return self

    def shiftbnd(self, dist: float) -> 'Polygon3':
        # MATLAB: @polygon3/shiftbnd.m
        self.poly = self.poly.shiftbnd(dist)
        self.edge = EdgeProfile()
        return self

    def copy(self) -> 'Polygon3':
        import copy
        return copy.deepcopy(self)

    def plate(self,
            dir: int = 1,
            edge: Optional[EdgeProfile] = None,
            hdata: Optional[dict] = None,
            options: Optional[dict] = None,
            refun: Optional[Any] = None,
            sym: Optional[str] = None) -> Tuple[Particle, 'Polygon3']:
        # MATLAB: @polygon3/plate.m -- single Polygon3 entry point.
        # Delegates to plate_from_list so the single- and multi-polygon paths
        # share one implementation (MATLAB plate.m always takes an obj array).
        p, obj_list = Polygon3.plate_from_list(
            [self], dir = dir, edge = edge,
            hdata = hdata, options = options, refun = refun, sym = sym)
        return p, obj_list[0]

    @staticmethod
    def plate_from_list(
            obj_list: List['Polygon3'],
            dir: int = 1,
            edge: Optional[EdgeProfile] = None,
            hdata: Optional[dict] = None,
            options: Optional[dict] = None,
            refun: Optional[Any] = None,
            sym: Optional[str] = None) -> Tuple[Particle, List['Polygon3']]:
        # MATLAB: @polygon3/plate.m (full porting, including multi-polygon /
        # plate-with-hole path used by e.g. demoeelsret7 (outer square +
        # inner triangle) and demospecstat6 (inner ring + outer square).

        if len(obj_list) == 0:
            raise ValueError('[error] plate_from_list: empty obj_list')

        # Work on local copies so we do not mutate caller state.
        obj_list = [o.copy() for o in obj_list]

        # MATLAB L20: assert identical z across all entries.
        z_vals = np.array([o.z for o in obj_list])
        z_round = mround(z_vals, 8)
        assert np.all(z_round == z_round[0]), '[error] plate: z-values must match'
        z_plate = float(z_round[0])

        # MATLAB L24-25: override edge profile if passed in.
        if edge is not None:
            for o in obj_list:
                o.edge = edge

        if options is None:
            options = {'output': False}
        if hdata is None:
            hdata = {}
        else:
            hdata = dict(hdata)

        # MATLAB L27-30: gather per-obj refine functions.
        fun_list = []
        poly_list_for_refun: List[Polygon] = []
        for o in obj_list:
            if o._refun is not None:
                fun_list.append(o._refun)
                poly_list_for_refun.append(o.poly)
        global_refun = refun

        # MATLAB L36-38: install combined refine function if any present.
        if len(fun_list) > 0 or global_refun is not None:
            _fl = list(fun_list)
            _polys = list(poly_list_for_refun)
            _gf = global_refun
            _z = z_plate
            _poly0 = obj_list[0].poly

            def _plate_refun(x, y, *args,
                             _fl = _fl, _polys = _polys, _gf = _gf,
                             _z = _z, _poly0 = _poly0):
                x = np.atleast_1d(x).ravel()
                y = np.atleast_1d(y).ravel()
                pos = np.column_stack([x, y, np.full_like(x, _z)])
                h_vals = None
                for fun, poly_i in zip(_fl, _polys):
                    d_i, _ = poly_i.dist(np.column_stack([x, y]))
                    hi = np.asarray(fun(pos, np.asarray(d_i).ravel())).ravel()
                    h_vals = hi if h_vals is None else np.minimum(h_vals, hi)
                if _gf is not None:
                    d0, _ = _poly0.dist(np.column_stack([x, y]))
                    hg = np.asarray(_gf(pos, np.asarray(d0).ravel())).ravel()
                    h_vals = hg if h_vals is None else np.minimum(h_vals, hg)
                return h_vals

            hdata['fun'] = _plate_refun

        # MATLAB L43-45: gather per-obj polygons, symmetrize+close combined.
        polys_for_mesh: List[Polygon] = [o.poly.copy() for o in obj_list]
        if sym is not None:
            for pm in polys_for_mesh:
                pm._apply_symmetry(sym)
            polys_for_mesh = [pm.close() for pm in polys_for_mesh]

        # MATLAB L47-48: polymesh2d(poly1, hdata, options).
        if len(polys_for_mesh) == 1:
            verts_2d, faces_2d = polys_for_mesh[0].polymesh2d(
                hdata = hdata, options = options)
        else:
            verts_2d, faces_2d = polys_for_mesh[0].polymesh2d(
                *polys_for_mesh[1:], hdata = hdata, options = options)

        # Build initial particle at the plate z-level.
        verts_3d = np.empty((verts_2d.shape[0], 3))
        verts_3d[:, :2] = verts_2d
        verts_3d[:, 2] = z_plate

        # mesh2d returns tri faces; pad to 4 columns with NaN (quad marker).
        if faces_2d.shape[1] == 3:
            faces_padded = np.full((faces_2d.shape[0], 4), np.nan)
            faces_padded[:, :3] = faces_2d
        else:
            faces_padded = faces_2d

        p = Particle(verts_3d, faces_padded)
        p = _add_midpoints_flat(p)

        # MATLAB L60-62: flip to respect requested normal direction.
        nvec_sum = np.sum(p.nvec[:, 2])
        if np.sign(nvec_sum) != dir:
            p = p.flipfaces()

        # MATLAB L65-67: per-polygon dmin tracking for vshift.
        dmin = np.full(p.verts2.shape[0], np.inf)

        # MATLAB L70-90: per-polygon boundary smoothing + edge vshift.
        result_objs: List['Polygon3'] = []
        for i, o in enumerate(obj_list):
            # MATLAB L51-53: enrich each input polygon with new boundary
            # vertices and return the full (symmetry-expanded) polygon.
            enriched = o.poly.copy().interp1(verts_2d)
            if sym is not None:
                enriched._apply_symmetry(sym)
                enriched.pos = enriched.get_full_polygon()
                enriched.sym = None
            result_o = o.copy()
            result_o.poly = enriched.copy()
            result_objs.append(result_o)

            # MATLAB L72-73: enrich against verts2 (mid-edges included).
            poly_i = o.poly.copy().interp1(p.verts2[:, :2])
            if sym is not None:
                poly_i._apply_symmetry(sym)
                poly_i.pos = poly_i.get_full_polygon()
                poly_i.sym = None

            # MATLAB L75-77: locate mesh verts2 rows on poly_i (col-major
            # ordering matches MATLAB find()).
            if poly_i.pos.shape[0] > 0:
                pp = poly_i.pos
                vv = p.verts2[:, :2]
                eqx = (vv[:, 0:1] == pp[:, 0][None, :])
                eqy = (vv[:, 1:2] == pp[:, 1][None, :])
                mask = eqx & eqy
                rows: List[int] = []
                cols: List[int] = []
                for c in range(mask.shape[1]):
                    rr = np.where(mask[:, c])[0]
                    for r in rr:
                        rows.append(int(r))
                        cols.append(int(c))
                row = np.asarray(rows, dtype = int)
                col = np.asarray(cols, dtype = int)
            else:
                row = np.empty(0, dtype = int)
                col = np.empty(0, dtype = int)

            # MATLAB L81-82: smooth boundary with midpoints('same') and
            # overwrite the x,y of the matching verts2 rows.
            if poly_i.pos.shape[0] >= 4 and poly_i.pos.shape[0] % 2 == 0 \
                    and len(row) > 0:
                poly_smooth = poly_i.copy().midpoints(mode = 'same')
                p.verts2[row, 0:2] = poly_smooth.pos[col, :]
                poly_for_dist = poly_smooth
            else:
                poly_for_dist = poly_i

            # MATLAB L84-89: distance to polygon, update dmin, apply vshift
            # at boundary vertices whose distance improved. MATLAB plate.m
            # L85 uses the smoothed polygon for dist, not the original.
            d, _ = poly_for_dist.dist(p.verts2[:, :2])
            d = np.asarray(d).ravel()
            ind = d < dmin
            dmin[ind] = d[ind]

            if o.edge is not None and o.edge.pos is not None:
                v = o.edge.vshift(o.z, d[ind])
                if np.isscalar(v):
                    if v != 0.0:
                        p.verts2[ind, 2] = z_plate + v
                else:
                    v = np.asarray(v).ravel()
                    p.verts2[ind, 2] = z_plate + v

        # MATLAB L95: final curved particle from updated verts2/faces2.
        p_final = Particle(p.verts2.copy(), p.faces2.copy())

        return p_final, result_objs

    def vribbon(self,
            z: Optional[np.ndarray] = None,
            edge: Optional[EdgeProfile] = None,
            sym: Optional[str] = None) -> Tuple[Particle, 'Polygon3', 'Polygon3']:
        # MATLAB: @polygon3/vribbon.m
        if edge is not None:
            self.edge = edge

        if z is None:
            if self.edge is not None and self.edge.z is not None:
                z = self.edge.z.copy()
            else:
                assert False, '[error] z-values required for vribbon'

        # edge profile horizontal shift function
        def hshift_fun(z_vals: np.ndarray) -> np.ndarray:
            return self.edge.hshift(z_vals)

        p, up, lo = self._ribbon_v(z, hshift_fun, sym = sym)
        return p, up, lo

    def _ribbon_v(self,
            z: np.ndarray,
            hshift_fun: Any,
            sym: Optional[str] = None) -> Tuple[Particle, 'Polygon3', 'Polygon3']:
        # MATLAB: ribbon() subfunction in vribbon.m

        # smoothened polygon with midpoints
        poly_smooth = self.poly.copy().midpoints()

        # MATLAB: handle symmetry -- reduce after midpoints
        if sym is not None:
            poly_smooth._apply_symmetry(sym)
            # remove origin for xy-symmetry
            if sym == 'xy' and len(poly_smooth.pos) > 0 and np.all(poly_smooth.pos[-1] == 0):
                poly_smooth.pos = poly_smooth.pos[:-1]

        pos = poly_smooth.pos
        nvec = poly_smooth.compute_normals()

        # MATLAB: close contour only if no symmetry or first/last not on axes
        should_close = True
        if sym is not None:
            products = np.abs(np.prod(pos[[0, -1]], axis = 1))
            if np.all(products < 1e-6):
                should_close = False

        if should_close:
            pos_closed = np.empty((pos.shape[0] + 1, 2))
            pos_closed[:pos.shape[0]] = pos
            pos_closed[pos.shape[0]] = pos[0]

            nvec_closed = np.empty((nvec.shape[0] + 1, 2))
            nvec_closed[:nvec.shape[0]] = nvec
            nvec_closed[nvec.shape[0]] = nvec[0]
        else:
            pos_closed = pos
            nvec_closed = nvec

        # extend z-values for midpoints (interleave with averages)
        n_z = len(z)
        z_ext = np.empty(2 * n_z - 1)
        z_ext[0::2] = z
        z_ext[1::2] = 0.5 * (z[:-1] + z[1:])

        # create grid indices: odd positions along polygon, odd along z
        poly_indices = np.arange(0, pos_closed.shape[0], 2)  # 0, 2, 4, ...
        z_indices = np.arange(0, len(z_ext), 2)  # 0, 2, 4, ...

        u, faces = fvgrid(poly_indices.astype(float), z_indices.astype(float))

        # u[:, 0] -> polygon position index, u[:, 1] -> z index
        u_int = u.astype(int)

        # build 3D vertices
        n_verts = u_int.shape[0]
        verts = np.empty((n_verts, 3))
        verts[:, 0] = pos_closed[u_int[:, 0], 0]
        verts[:, 1] = pos_closed[u_int[:, 0], 1]
        verts[:, 2] = z_ext[u_int[:, 1]]

        # apply horizontal shift from edge profile
        shift_vals = hshift_fun(verts[:, 2])
        verts[:, 0] = verts[:, 0] + shift_vals * nvec_closed[u_int[:, 0], 0]
        verts[:, 1] = verts[:, 1] + shift_vals * nvec_closed[u_int[:, 0], 1]

        # create particle
        p = Particle(verts, faces)

        # check normal direction: should point outward
        # find point closest to first polygon point
        dx = p.pos[:, 0] - pos[0, 0]
        dy = p.pos[:, 1] - pos[0, 1]
        ind = np.argmin(dx ** 2 + dy ** 2)

        # reference normal direction
        ref_vec = np.array([nvec[0, 0], nvec[0, 1], 0.0])
        if np.dot(ref_vec, p.nvec[ind]) < 0:
            p = p.flipfaces()

        # boundary polygons for upper and lower
        def _shifted_poly(z_val: float) -> Polygon:
            shift_val = hshift_fun(np.array([z_val]))[0]
            shifted = self.poly.copy().shiftbnd(shift_val)
            return shifted

        up = self.copy()
        up.poly = _shifted_poly(np.max(z))
        up.z = np.max(z)

        lo = self.copy()
        lo.poly = _shifted_poly(np.min(z))
        lo.z = np.min(z)

        return p, up, lo

    def hribbon(self,
            d: np.ndarray,
            dir: int = 1) -> Tuple[Particle, 'Polygon3', 'Polygon3']:
        # MATLAB: @polygon3/hribbon.m
        p, inner, outer = self._ribbon_h(d, dir)
        return p, inner, outer

    def _ribbon_h(self,
            d: np.ndarray,
            dir: int) -> Tuple[Particle, 'Polygon3', 'Polygon3']:
        # MATLAB: ribbon() subfunction in hribbon.m

        # smoothened polygon with midpoints
        poly_smooth = self.poly.copy().midpoints()

        pos = poly_smooth.pos
        nvec = poly_smooth.compute_normals()

        # close contour: append first point
        pos_closed = np.empty((pos.shape[0] + 1, 2))
        pos_closed[:pos.shape[0]] = pos
        pos_closed[pos.shape[0]] = pos[0]

        nvec_closed = np.empty((nvec.shape[0] + 1, 2))
        nvec_closed[:nvec.shape[0]] = nvec
        nvec_closed[nvec.shape[0]] = nvec[0]

        # extend d-values for midpoints
        d = np.asarray(d, dtype = float)
        n_d = len(d)
        d_ext = np.empty(2 * n_d - 1)
        d_ext[0::2] = d
        d_ext[1::2] = 0.5 * (d[:-1] + d[1:])

        # grid indices
        poly_indices = np.arange(0, pos_closed.shape[0], 2)
        d_indices = np.arange(0, len(d_ext), 2)

        u, faces = fvgrid(poly_indices.astype(float), d_indices.astype(float))
        u_int = u.astype(int)

        # compute displaced positions for each d-value
        n_pos = pos_closed.shape[0]
        n_d_ext = len(d_ext)
        x = np.zeros((n_pos, n_d_ext))
        y = np.zeros((n_pos, n_d_ext))

        for i in range(n_d_ext):
            # shift boundary by d_ext[i]
            poly_temp = poly_smooth.copy()
            _, distp = poly_temp.shiftbnd(d_ext[i], return_dist = True)

            # handle closed polygons: need (n_pos) values including closing point
            if len(distp) != n_pos:
                dist_full = np.empty(n_pos)
                dist_full[:len(distp)] = distp
                dist_full[len(distp):] = distp[0]
            else:
                dist_full = distp

            x[:, i] = pos_closed[:, 0] + dist_full * nvec_closed[:, 0]
            y[:, i] = pos_closed[:, 1] + dist_full * nvec_closed[:, 1]

        # assemble vertices from grid
        n_verts = u_int.shape[0]
        verts_x = np.empty(n_verts)
        verts_y = np.empty(n_verts)
        for k in range(n_verts):
            verts_x[k] = x[u_int[k, 0], u_int[k, 1]]
            verts_y[k] = y[u_int[k, 0], u_int[k, 1]]

        verts = np.empty((n_verts, 3))
        verts[:, 0] = verts_x
        verts[:, 1] = verts_y
        verts[:, 2] = self.z

        # create particle
        p = Particle(verts, faces)

        # check normal direction
        nvec_sum = np.sum(p.nvec[:, 2])
        if np.sign(nvec_sum) != dir:
            p = p.flipfaces()

        # inner/outer boundary polygons
        inner = self.copy()
        inner.poly = self.poly.copy().shiftbnd(np.min(d))
        inner.z = self.z

        outer = self.copy()
        outer.poly = self.poly.copy().shiftbnd(np.max(d))
        outer.z = self.z

        return p, inner, outer
