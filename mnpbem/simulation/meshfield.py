"""
Near-field computation on a mesh grid.

MATLAB: Simulation/misc/@meshfield/

Given a BEM solution (surface charges/currents), computes electric and
magnetic fields at arbitrary grid points around the particle.

Matches MATLAB MNPBEM @meshfield implementation.
"""

import numpy as np
from ..geometry.compoint import ComPoint
from ..greenfun.compgreen_stat import CompGreenStat, CompStruct
from . import meshfield_fmm as _mf_fmm


class MeshField(object):
    """
    Near-field computation on a mesh grid.

    MATLAB: @meshfield

    Given a BEM solution (surface charges/currents), computes electric and
    magnetic fields at arbitrary grid points around the particle.

    Parameters
    ----------
    p : ComParticle
        Composite particle defining the dielectric environment
    x, y : ndarray
        1D or 2D grid coordinates
    z : ndarray or float, optional
        z-coordinate (default: 0)
    nmax : int, optional
        Work off calculation in portions of nmax (for memory)
    mindist : float, optional
        Minimum distance of grid points to particle boundary
    sim : str, optional
        'stat' for quasistatic (default), 'ret' for retarded
    **options : dict
        Additional arguments passed to ComPoint and Green function

    Attributes
    ----------
    x, y, z : ndarray
        Grid coordinates (same shape)
    p : ComParticle
        Composite particle
    pt : ComPoint
        Point object for grid positions
    g : CompGreenStat or CompGreenRet
        Green function connecting grid points to particle
    nmax : int or None
        Portion size for batched computation

    Examples
    --------
    >>> from mnpbem import trisphere, EpsConst, ComParticle
    >>> from mnpbem.simulation import MeshField
    >>>
    >>> eps = [EpsConst(1.0), EpsConst(-10 + 1j)]
    >>> p = ComParticle(eps, [trisphere(144, 10)], [[2, 1]])
    >>> x, z = np.meshgrid(np.linspace(-20, 20, 41), np.linspace(-20, 20, 41))
    >>> mf = MeshField(p, x, 0, z)
    >>> e, h = mf.field(sig)
    """

    def __init__(self, p, x, y, z=None, nmax=None, mindist=None,
                 sim='stat', **options):
        # Handle z default
        if z is None:
            z = 0.0

        # Convert to arrays
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        z = np.asarray(z, dtype=np.float64)

        # Expand dimensions if needed (MATLAB: expand function)
        x, y, z = self._expand(x, y, z)

        # Layer structure (substrate). When present, the field is evaluated
        # with the layer-aware Green function (direct + reflected), matching
        # MATLAB @meshfield/init.m where greenfunction() dispatches to
        # compgreenretlayer. The associated tabulated reflected Green function
        # may be supplied via 'greentab'.
        self._layer = options.pop('layer', None)
        self._greentab = options.pop('greentab', None)

        # Save positions and particle
        self.x = x
        self.y = y
        self.z = z
        self.p = p
        self.nmax = nmax
        self._sim = sim
        self._options = options

        # Make ComPoint object
        # Flatten grid positions to (N, 3) array
        pos = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
        self.pt = ComPoint(p, pos, mindist=mindist, layer=self._layer)

        if nmax is None:
            # Precompute Green function
            self.g = self._make_green(self.pt, p, sim, **options)
        else:
            self.g = None

    @staticmethod
    def _expand(x, y, z):
        """
        Expand dimensions to match shapes.

        MATLAB: @meshfield/init.m -> expand()

        If x, y are 2D and z is scalar, broadcast z.
        If x, y are 2D and z is 1D, create 3D arrays.
        """
        # Check if all shapes match
        shapes = []
        for arr in [x, y, z]:
            if arr.ndim == 0:
                shapes.append((1,))
            else:
                shapes.append(arr.shape)

        unique_shapes = set(shapes)
        if len(unique_shapes) == 1:
            # All same shape, nothing to do
            return x, y, z

        # Determine which arrays match
        if x.shape == y.shape:
            # x and y match, expand z
            if z.ndim == 0 or z.size == 1:
                z = np.zeros_like(x) + float(z.ravel()[0])
            else:
                # 3D expansion
                siz1 = x.shape
                n2 = z.size
                x = np.tile(x.reshape(siz1 + (1,)), (1,) * len(siz1) + (n2,))
                y = np.tile(y.reshape(siz1 + (1,)), (1,) * len(siz1) + (n2,))
                z = np.tile(z.reshape((1,) * len(siz1) + (n2,)), siz1 + (1,))
        elif x.shape == z.shape:
            # x and z match, expand y
            if y.ndim == 0 or y.size == 1:
                y = np.zeros_like(x) + float(y.ravel()[0])
            else:
                siz1 = x.shape
                n2 = y.size
                x = np.tile(x.reshape(siz1 + (1,)), (1,) * len(siz1) + (n2,))
                z = np.tile(z.reshape(siz1 + (1,)), (1,) * len(siz1) + (n2,))
                y = np.tile(y.reshape((1,) * len(siz1) + (n2,)), siz1 + (1,))
        elif y.shape == z.shape:
            # y and z match, expand x
            if x.ndim == 0 or x.size == 1:
                x = np.zeros_like(y) + float(x.ravel()[0])
            else:
                siz1 = y.shape
                n2 = x.size
                y = np.tile(y.reshape(siz1 + (1,)), (1,) * len(siz1) + (n2,))
                z = np.tile(z.reshape(siz1 + (1,)), (1,) * len(siz1) + (n2,))
                x = np.tile(x.reshape((1,) * len(siz1) + (n2,)), siz1 + (1,))
        else:
            # Try scalar expansion for each
            if x.ndim == 0 or x.size == 1:
                target = y if y.size > 1 else z
                x = np.zeros_like(target) + float(x.ravel()[0])
            if y.ndim == 0 or y.size == 1:
                target = x if x.size > 1 else z
                y = np.zeros_like(target) + float(y.ravel()[0])
            if z.ndim == 0 or z.size == 1:
                target = x if x.size > 1 else y
                z = np.zeros_like(target) + float(z.ravel()[0])

        return x, y, z

    def _make_green(self, pt, p, sim='stat', **options):
        """
        Create Green function between grid points and particle.

        MATLAB: greenfunction(pt, p, op)

        Parameters
        ----------
        pt : ComPoint
            Grid point object
        p : ComParticle
            Particle
        sim : str
            'stat' or 'ret'

        Returns
        -------
        g : CompGreenStat / CompGreenRet / CompGreenStatLayer / CompGreenRetLayer
            Green function object. Layer-aware variants are returned when a
            layer structure was supplied to the constructor.
        """
        layer = getattr(self, '_layer', None)
        greentab = getattr(self, '_greentab', None)

        if sim == 'stat':
            if layer is not None:
                from ..greenfun.compgreen_stat_layer import CompGreenStatLayer
                return CompGreenStatLayer(pt, p, layer, **options)
            return CompGreenStat(pt, p, **options)
        else:
            if layer is not None:
                from ..greenfun.compgreen_ret_layer import CompGreenRetLayer
                opts = dict(options)
                if greentab is not None:
                    # Mirror BEMRetLayer: pass the underlying GreenTabLayer.
                    gt = greentab
                    if hasattr(gt, 'tab'):
                        opts['greentab_obj'] = gt.tab
                    elif hasattr(gt, 'r'):
                        opts['greentab_obj'] = gt
                return CompGreenRetLayer(pt, p, layer, **opts)
            from ..greenfun.compgreen_ret import CompGreenRet
            return CompGreenRet(pt, p, **options)

    def field(self, sig, inout=2, fmm=False, fmm_eps=1e-12):
        """
        Compute electromagnetic fields at grid points.

        MATLAB: @meshfield/field.m, field1.m, field2.m

        Parameters
        ----------
        sig : CompStruct
            Surface charges (and currents for retarded) from BEM solver,
            or a CompStruct with pre-computed field 'e'
        inout : int, optional
            Fields inside (1) or outside (2, default) of particle surface
        fmm : bool, optional
            Use FMM (fmm3dpy) for free-space evaluation. Only valid for
            ret simulation, single-region, no layer/mirror Green function.
        fmm_eps : float, optional
            FMM precision (default 1e-12)

        Returns
        -------
        e : ndarray
            Electric field, shape matching grid + (3,)
        h : ndarray or None
            Magnetic field (None for quasistatic)
        """
        if fmm and self._fmm_eligible(sig, inout):
            return self._field_fmm(sig, inout, fmm_eps)

        if self.nmax is None:
            return self._field1(sig, inout)
        else:
            return self._field2(sig, inout)

    def _fmm_eligible(self, sig, inout):
        if not _mf_fmm.fmm_available():
            return False
        if self._sim != 'ret':
            return False
        if hasattr(sig, 'val') and 'e' in sig.val:
            return False
        cls_name = type(self.g).__name__ if self.g is not None else ''
        if cls_name not in ('CompGreenRet',):
            return False
        if inout == 1:
            return hasattr(sig, 'sig1') and hasattr(sig, 'h1')
        if inout == 2:
            ok2 = hasattr(sig, 'sig2') and hasattr(sig, 'h2')
            ok1 = hasattr(sig, 'sig1') and hasattr(sig, 'h1')
            return ok2 or ok1
        return False

    def _field_fmm(self, sig, inout, fmm_eps):
        enei = sig.enei
        k_vac = 2 * np.pi / enei

        if inout == 1:
            sig_scalar_raw = sig.sig1
            h_raw = sig.h1
        else:
            sig_scalar_raw = getattr(sig, 'sig2', None)
            h_raw = getattr(sig, 'h2', None)
            if sig_scalar_raw is None or h_raw is None:
                sig_scalar_raw = sig.sig1
                h_raw = sig.h1

        n_regions_p1 = len(self.g.con)
        region_idx = min(inout - 1, n_regions_p1 - 1)
        con_row = self.g.con[region_idx]
        medium_one_based = None
        for con_block in con_row:
            if con_block is not None and isinstance(con_block, np.ndarray) and con_block.size > 0:
                vals = con_block[con_block > 0]
                if vals.size > 0:
                    medium_one_based = int(vals.flat[0])
                    break
        if medium_one_based is None:
            medium_one_based = inout
        medium_idx = medium_one_based - 1

        eps_func = self.p.eps[medium_idx]
        _, k_wave = eps_func(enei)
        zk = complex(k_wave)

        src_pos = self.p.pos
        src_area = self.p.area

        sig_scalar = sig_scalar_raw
        if isinstance(sig_scalar, np.ndarray) and sig_scalar.ndim > 1:
            sig_scalar = sig_scalar[..., 0]

        h_xyz = h_raw
        if isinstance(h_xyz, np.ndarray) and h_xyz.ndim == 3 and h_xyz.shape[-1] == 1:
            h_xyz = h_xyz[..., 0]

        tgt_pos = self.pt.pos

        e_flat, h_flat = _mf_fmm.eval_freespace_field(
            zk, complex(k_vac), src_pos, src_area, tgt_pos,
            sig_scalar, h_xyz, eps = fmm_eps)

        e_full = self.pt(e_flat)
        h_full = self.pt(h_flat)

        e = self._reshape_field(e_full)
        h = self._reshape_field(h_full)
        return e, h

    def _field1(self, sig, inout=2):
        """
        Compute fields with precomputed Green function.

        MATLAB: @meshfield/field1.m
        """
        # Check if sig already has field 'e'
        if hasattr(sig, 'val') and 'e' in sig.val:
            f = sig
        else:
            # Compute electromagnetic fields using Green function
            f = self.g.field(sig, inout)

        # Electric field
        e = self.pt(f.e)

        # Reshape to grid shape
        e = self._reshape_field(e)

        # Magnetic field
        h = None
        if hasattr(f, 'val') and 'h' in f.val:
            h = self.pt(f.h)
            h = self._reshape_field(h)

        return e, h

    def _field2(self, sig, inout=2):
        """
        Compute fields in portions (for large grids).

        MATLAB: @meshfield/field2.m
        """
        # Check if sig already has field 'e'
        if hasattr(sig, 'val') and 'e' in sig.val:
            f = sig
            e = self.pt(f.e)
            e = self._reshape_field(e)
            h = None
            if hasattr(f, 'val') and 'h' in f.val:
                h = self.pt(f.h)
                h = self._reshape_field(h)
            return e, h

        # Work off calculation in portions of nmax
        npts = self.pt.n
        boundaries = list(range(0, npts, self.nmax))
        if boundaries[-1] != npts:
            boundaries.append(npts)

        e_all = None
        h_all = None

        for i in range(1, len(boundaries)):
            start = boundaries[i - 1]
            end = boundaries[i]
            ind = np.arange(start, end)

            # Select subset of points
            pt_sub = self.pt.select(index=ind)

            # Create Green function for subset
            g_sub = self._make_green(pt_sub, self.p, self._sim, **self._options)

            # Compute field for subset
            f_sub = g_sub.field(sig, inout)

            # Allocate and store
            if e_all is None:
                e_shape = (npts,) + f_sub.e.shape[1:]
                e_all = np.zeros(e_shape, dtype=f_sub.e.dtype)
                if hasattr(f_sub, 'val') and 'h' in f_sub.val:
                    h_shape = (npts,) + f_sub.h.shape[1:]
                    h_all = np.zeros(h_shape, dtype=f_sub.h.dtype)

            e_all[start:end] = f_sub.e
            if h_all is not None and hasattr(f_sub, 'val') and 'h' in f_sub.val:
                h_all[start:end] = f_sub.h

        # Apply ComPoint mapping and reshape
        e = self.pt(e_all)
        e = self._reshape_field(e)

        h = None
        if h_all is not None:
            h = self.pt(h_all)
            h = self._reshape_field(h)

        return e, h

    def potential(self, sig, inout=2):
        """
        Compute scalar potential at grid points.

        Parameters
        ----------
        sig : CompStruct
            Surface charges from BEM solver
        inout : int, optional
            Potentials inside (1) or outside (2, default) of particle

        Returns
        -------
        phi : ndarray
            Scalar potential, shape matching grid
        """
        pot = self.g.potential(sig, inout)

        # Get the appropriate potential field
        if inout == 1:
            phi = self.pt(pot.phi1)
        else:
            phi = self.pt(pot.phi2)

        # Reshape to grid shape
        phi = self._reshape_scalar(phi)

        return phi

    def _reshape_field(self, e):
        """
        Reshape field array to match grid shape.

        MATLAB: field1.m lines 24-33

        Parameters
        ----------
        e : ndarray, shape (npts_total, 3, ...)
            Field values at all grid points

        Returns
        -------
        e : ndarray
            Reshaped field
        """
        siz = e.shape
        if self.x.ndim == 1 or self.x.shape[1] == 1 if self.x.ndim == 2 else False:
            # Column vector shape
            new_shape = (self.x.shape[0],) + siz[1:]
        else:
            new_shape = self.x.shape + siz[1:]
        return e.reshape(new_shape)

    def _reshape_scalar(self, phi):
        """Reshape scalar array to match grid shape."""
        if self.x.ndim == 1 or (self.x.ndim == 2 and self.x.shape[1] == 1):
            new_shape = (self.x.shape[0],)
        else:
            new_shape = self.x.shape
        return phi.reshape(new_shape)

    @property
    def pos(self):
        """Point positions."""
        return self.pt.pos

    def __call__(self, sig, inout=2, fmm=False, fmm_eps=1e-12):
        """
        Compute fields (callable interface).

        MATLAB: @meshfield/subsref.m (parentheses access)

        Parameters
        ----------
        sig : CompStruct
            Surface charges or pre-computed fields
        inout : int, optional
            Inside (1) or outside (2, default)
        fmm : bool, optional
            Use FMM for free-space ret evaluation
        fmm_eps : float, optional
            FMM precision

        Returns
        -------
        e : ndarray
            Electric field
        h : ndarray or None
            Magnetic field
        """
        return self.field(sig, inout, fmm = fmm, fmm_eps = fmm_eps)

    def __repr__(self):
        return 'MeshField(x={}, y={}, z={}, p={})'.format(
            self.x.shape, self.y.shape, self.z.shape, self.p)
