"""
Base class for electron energy loss spectroscopy (EELS) simulations.

The electron beam is assumed to propagate along z with velocity vel.
EELSBase provides the base functions for EELSStat and EELSRet.

Reference:
    Garcia de Abajo et al., PRB 65, 115418 (2002), RMP 82, 209 (2010).
    MATLAB MNPBEM Simulation/misc/@eelsbase
"""

import numpy as np
from scipy.special import kv as besselk
from typing import Optional, Tuple

from ..misc import pdist2, lglnodes
from ..utils.matlab_compat import msqrt, mlog


class EELSBase(object):
    """
    Base class for EELS simulations.

    The electron beam propagates along z with velocity vel (in units of c).
    This class provides beam path computation, free-space potentials, enclosure
    checks, and intersection-point bookkeeping shared by EELSStat and EELSRet.

    Parameters
    ----------
    p : ComParticle
        Particle object for EELS simulation
    impact : ndarray, shape (nimp, 2)
        Impact parameters (x, y) of electron beams
    width : float
        Width of electron beam for potential smearing
    vel : float
        Electron velocity in units of speed of light
    cutoff : float, optional
        Distance for integration refinement (default: 10 * width)
    phiout : float, optional
        Half aperture collection angle of spectrometer (default: 1e-2)

    Attributes
    ----------
    p : ComParticle
        Particle object
    impact : ndarray
        Impact parameters
    width : float
        Beam width
    vel : float
        Electron velocity / c
    phiout : float
        Half aperture collection angle

    Notes
    -----
    MATLAB: Simulation/misc/@eelsbase/
    """

    def __init__(self,
            p: object,
            impact: np.ndarray,
            width: float,
            vel: float,
            cutoff: Optional[float] = None,
            phiout: float = 1e-2,
            **options) -> None:
        # MATLAB: eelsbase.m constructor -> init.m
        self.phiout = phiout
        self._init(p, impact, width, vel, cutoff = cutoff, phiout = phiout, **options)

    def _init(self,
            p: object,
            impact: np.ndarray,
            width: float,
            vel: float,
            cutoff: Optional[float] = None,
            phiout: float = 1e-2,
            **options) -> None:
        """
        Initialize electron beam excitation.

        MATLAB: @eelsbase/private/init.m

        Parameters
        ----------
        p : ComParticle
            Particle object
        impact : ndarray, shape (nimp, 2)
            Impact parameters
        width : float
            Beam width
        vel : float
            Electron velocity / c
        cutoff : float, optional
            Distance for integration refinement
        phiout : float, optional
            Spectrometer half aperture angle
        """
        # MATLAB: init.m lines 17-21
        if cutoff is None:
            cutoff = 10 * width
        if phiout is not None:
            self.phiout = phiout

        # MATLAB: init.m line 22
        #   if isfield(op, 'refine'), p = set(p, 'quad', quadface(op)); end
        # The 'eels.refine' bemoption (e.g. demoeelsret7/8) subdivides the
        # unit-triangle integration to improve near-beam boundary integration
        # accuracy. Must be applied to every Particle making up the ComParticle
        # since _quad() walks the composite through quad_integration().
        refine = options.get('refine', None)
        if refine is None:
            # Support MATLAB-style nested option: options = {'eels': {'refine': 2}}
            eels_opts = options.get('eels', None)
            if isinstance(eels_opts, dict):
                refine = eels_opts.get('refine', None)
        if refine is not None and refine > 1:
            from ..utils.quadface import QuadFace as _QuadFace
            rule = options.get('rule', 18)
            npol = options.get('npol', (7, 5))
            new_quad = _QuadFace(rule = rule, npol = npol, refine = int(refine))
            # Replace quad on the composite pc and on each sub-particle so that
            # Particle.quad_integration() (used by _quad) sees the refined rule.
            if hasattr(p, 'pc') and p.pc is not None:
                p.pc.quad = new_quad
            if hasattr(p, 'p') and isinstance(p.p, list):
                for sub in p.p:
                    sub.quad = new_quad
            else:
                # Single particle
                p.quad = new_quad

        # Save input
        # MATLAB: init.m line 25
        impact = np.asarray(impact, dtype = np.float64)
        self.p = p
        self.impact = impact.copy()
        self.width = width
        self.vel = vel

        # ---- auxiliary quantities ----
        # MATLAB: init.m lines 28-31
        rad = self._enclosure(p)
        eta = 1e-6

        # MATLAB: init.m lines 33-39
        n = impact.shape[0]
        dist = pdist2(p.pos[:, 0:2], impact)

        # ---- crossing points between electron beam and surface elements ----
        # MATLAB: init.m lines 42-71
        row_arr, col_arr = np.where(dist <= rad)
        cross = np.zeros((p.n, n), dtype = bool)

        for i in np.unique(row_arr):
            face = p.faces[i, :]
            face = face[~np.isnan(face)].astype(int)
            xv = p.verts[face, 0]
            yv = p.verts[face, 1]

            ind = np.where(row_arr == i)[0]
            x = impact[col_arr[ind], 0]
            y = impact[col_arr[ind], 1]

            # Check if points are inside polygon
            in_mask, on_mask = self._inpolygon(x, y, xv, yv)
            cross[i, col_arr[ind[in_mask]]] = True

            # Move points on boundary into polygon
            if np.any(on_mask):
                j = col_arr[ind[on_mask]]
                impact[j, 0] = (1 - eta) * impact[j, 0] + eta * p.pos[i, 0]
                impact[j, 1] = (1 - eta) * impact[j, 1] + eta * p.pos[i, 1]

        # Update impact after boundary adjustments
        self.impact = impact

        # ---- electron beam trajectories inside particles ----
        # MATLAB: init.m lines 74-81
        inout = np.zeros((p.n, 2), dtype = int)
        _index = p.index_func if callable(getattr(p, 'index_func', None)) else p.index
        for i in range(p.np):
            ind = _index(i + 1) if callable(_index) else list(range(p.n))
            inout[ind, :] = np.tile(p.inout[i, :], (len(ind), 1))

        # MATLAB: init.m lines 83-88
        i1, j1 = np.where(cross)
        z_cross = (p.pos[i1, 2]
                   + (p.nvec[i1, 0] * (p.pos[i1, 0] - impact[j1, 0])
                      + p.nvec[i1, 1] * (p.pos[i1, 1] - impact[j1, 1]))
                   / p.nvec[i1, 2])

        # MATLAB: init.m lines 90-91
        zcut_list = []
        indimp_list = []
        indmat_list = []

        # MATLAB: init.m lines 94-113
        for j2 in range(impact.shape[0]):
            i2 = np.where(j1 == j2)[0]
            if len(i2) == 0:
                continue

            sort_idx = np.argsort(z_cross[i2])
            zz = z_cross[i2[sort_idx]]
            face_idx = i1[i2[sort_idx]]

            mat = inout[face_idx, :].copy()
            neg_nvec = p.nvec[face_idx, 2] < 0
            mat[neg_nvec, 0], mat[neg_nvec, 1] = mat[neg_nvec, 1].copy(), mat[neg_nvec, 0].copy()

            if len(zz) > 1:
                for k in range(len(zz) - 1):
                    zcut_list.append([zz[k], zz[k + 1]])
                    indimp_list.append(j2)
                    indmat_list.append(mat[k + 1, 0])

        # MATLAB: init.m lines 115-118
        if len(zcut_list) > 0:
            zcut = np.array(zcut_list)
            indimp = np.array(indimp_list)
            indmat = np.array(indmat_list)

            # Keep only trajectories outside of background medium (indmat != 1)
            # MATLAB uses 1-indexed materials; material 1 = background
            mask = indmat != 1
            self._z = zcut[mask, :]
            self._indimp = indimp[mask]
            self._indmat = indmat[mask]
        else:
            self._z = np.empty((0, 2))
            self._indimp = np.empty(0, dtype = int)
            self._indmat = np.empty(0, dtype = int)

        # ---- face elements with integration refinement ----
        # MATLAB: init.m line 122
        dmin = self._distmin(p, impact, cutoff)
        self._indquad = (~np.isnan(dmin)) | cross

    def path(self, medium: Optional[int] = None) -> np.ndarray:
        """
        Path length of electron beam propagating in different media.

        MATLAB: @eelsbase/path.m

        Parameters
        ----------
        medium : int, optional
            Select given medium (1-indexed). If None, return all media.

        Returns
        -------
        p : ndarray
            Path lengths. Shape (n_media, n_impact) or (n_impact,) if medium given.
        """
        # MATLAB: path.m lines 12-19
        n_media = len(self.p.eps)
        n_impact = self.impact.shape[0]
        siz = (n_media, n_impact)

        if len(self._indmat) == 0:
            result = np.zeros(siz)
        else:
            result = np.zeros(siz)
            lengths = self._z[:, 1] - self._z[:, 0]
            for k in range(len(self._indmat)):
                # MATLAB uses 1-indexed materials
                mat_idx = self._indmat[k] - 1 if self._indmat[k] > 0 else self._indmat[k]
                result[mat_idx, self._indimp[k]] += lengths[k]

        if medium is not None:
            # MATLAB 1-indexed
            return result[medium - 1, :]
        return result

    def full(self,
            a: np.ndarray,
            ind: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Expand array for electron beams inside particle to all impact parameters.

        MATLAB: @eelsbase/full.m

        Parameters
        ----------
        a : ndarray, shape (n, n_inside)
            Array for electron beams inside of particle
        ind : ndarray, optional
            Index to selected electron beams

        Returns
        -------
        a_full : ndarray, shape (n, n_impact)
            Array for all impact parameters
        """
        # MATLAB: full.m lines 13-26
        if ind is None:
            ind = np.arange(a.shape[1])

        n_rows = a.shape[0]
        n_impact = self.impact.shape[0]
        siz = (n_rows, n_impact)

        if len(ind) > 0:
            result = np.zeros(siz, dtype = a.dtype)
            for j_idx in range(len(ind)):
                imp_idx = self._indimp[ind[j_idx]]
                result[:, imp_idx] += a[:, ind[j_idx]]
            return result
        else:
            return np.zeros(siz, dtype = a.dtype)

    def potinfty(self,
            q: float,
            gamma: float = 1.0,
            mask: Optional[np.ndarray] = None,
            medium: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Potential for infinite electron beam in medium.

        MATLAB: @eelsbase/potinfty.m

        Parameters
        ----------
        q : float
            Wavenumber
        gamma : float, optional
            Lorentz contraction factor (default: 1)
        mask : ndarray, optional
            Compute potential only for selected particle objects
        medium : int, optional
            Compute potential only for beam inside given medium

        Returns
        -------
        phi : ndarray, shape (p.n, n_impact)
            Scalar potential
        phip : ndarray, shape (p.n, n_impact)
            Surface derivative of scalar potential
        """
        p = self.p

        # MATLAB: potinfty.m lines 18-27
        if mask is None:
            ind1 = np.arange(p.n)
        else:
            ind1 = getattr(p, "index_func", lambda x: p.index(x))(mask)

        if medium is None:
            ind2 = np.arange(self.impact.shape[0])
        else:
            ind2 = self._indimp[self._indmat == medium]

        # ---- potential at collocation points ----
        # MATLAB: potinfty.m lines 37-48
        pos1 = p.pos[ind1, :]
        nvec = p.nvec[ind1, :]
        pos2 = self.impact[ind2, :]

        # Relative coordinates
        x = pos1[:, 0:1] - pos2[:, 0:1].T  # (n1, n2)
        y = pos1[:, 1:2] - pos2[:, 1:2].T
        z = np.tile(pos1[:, 2:3], (1, pos2.shape[0]))

        r = msqrt(x ** 2 + y ** 2)
        rr = msqrt(r ** 2 + self.width ** 2)

        # MATLAB: potinfty.m lines 50-62
        phi = np.zeros((p.n, self.impact.shape[0]), dtype = complex)
        phip = np.zeros((p.n, self.impact.shape[0]), dtype = complex)

        K0 = besselk(0, q * rr / gamma)
        K1 = besselk(1, q * rr / gamma)

        exp_iqz = np.exp(1j * q * z)
        phi[np.ix_(ind1, ind2)] = -2 / self.vel * exp_iqz * K0
        phip[np.ix_(ind1, ind2)] = (-2 / self.vel * exp_iqz * q
            * (1j * K0 * nvec[:, 2:3]
               - K1 / gamma * (x * nvec[:, 0:1] + y * nvec[:, 1:2]) / rr))

        # ---- refined integration over boundary elements ----
        # MATLAB: potinfty.m lines 65-98
        quad_faces = np.where(np.any(self._indquad, axis = 1))[0]
        ind1_refine = np.intersect1d(quad_faces, ind1)

        for i1 in ind1_refine:
            pos1_q, w = self._quad(p, i1)
            w = w / p.area[i1]
            nvec_i = p.nvec[i1, :]

            i2 = np.where(self._indquad[i1, :])[0]
            i2 = np.intersect1d(i2, ind2)
            if len(i2) == 0:
                continue

            pos2_q = self.impact[i2, :]

            x_q = pos1_q[:, 0:1] - pos2_q[:, 0:1].T
            y_q = pos1_q[:, 1:2] - pos2_q[:, 1:2].T
            z_q = np.tile(pos1_q[:, 2:3], (1, pos2_q.shape[0]))

            r_q = msqrt(x_q ** 2 + y_q ** 2)
            rr_q = msqrt(r_q ** 2 + self.width ** 2)

            K0_q = besselk(0, q * rr_q / gamma)
            K1_q = besselk(1, q * rr_q / gamma)
            exp_iqz_q = np.exp(1j * q * z_q)

            phi[i1, i2] = -2 / self.vel * w @ (exp_iqz_q * K0_q)
            phip[i1, i2] = (-2 / self.vel * w @ (exp_iqz_q * q
                * (1j * K0_q * nvec_i[2]
                   - K1_q / gamma * (x_q * nvec_i[0] + y_q * nvec_i[1]) / rr_q)))

        return phi, phip

    def potinside(self,
            q: float,
            k: float,
            mask: Optional[np.ndarray] = None,
            medium: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Potential for electron beam inside of media.

        MATLAB: @eelsbase/potinside.m

        Parameters
        ----------
        q : float
            Wavenumber of electron beam
        k : float
            Wavenumber of light
        mask : ndarray, optional
            Compute potential only for selected particle objects
        medium : int, optional
            Compute potential only for electron beam inside given medium

        Returns
        -------
        phi : ndarray, shape (p.n, n_z)
            Scalar potential
        phip : ndarray, shape (p.n, n_z)
            Surface derivative of scalar potential
        """
        p = self.p

        # MATLAB: potinside.m lines 19-30
        if mask is None:
            ind1 = np.arange(p.n)
        else:
            ind1 = getattr(p, "index_func", lambda x: p.index(x))(mask)

        if medium is None:
            ind2 = np.arange(self._z.shape[0])
        else:
            ind2 = np.where(self._indmat == medium)[0]

        if len(ind2) == 0:
            phi = np.zeros((p.n, self._z.shape[0]), dtype = complex)
            phip = np.zeros((p.n, self._z.shape[0]), dtype = complex)
            return phi, phip

        # ---- potential at collocation points ----
        # MATLAB: potinside.m lines 33-57
        phi = np.zeros((p.n, self._z.shape[0]), dtype = complex)
        phip = np.zeros((p.n, self._z.shape[0]), dtype = complex)

        pos1 = p.pos[ind1, :]
        nvec = p.nvec[ind1, :]
        pos2 = self.impact[self._indimp[ind2], :]

        x = pos1[:, 0:1] - pos2[:, 0:1].T
        y = pos1[:, 1:2] - pos2[:, 1:2].T
        z = np.tile(pos1[:, 2:3], (1, pos2.shape[0]))

        r = msqrt(x ** 2 + y ** 2)
        rr = msqrt(r ** 2 + self.width ** 2)

        # Potential from charged wire
        I, Ir, Iz = self._potwire(rr, z, q, k, self._z[ind2, 0], self._z[ind2, 1])

        phi[np.ix_(ind1, ind2)] = -1 / self.vel * I
        phip[np.ix_(ind1, ind2)] = (-1 / self.vel
            * (Iz * nvec[:, 2:3]
               + Ir * (x * nvec[:, 0:1] + y * nvec[:, 1:2]) / rr))

        # ---- refined integration over boundary elements ----
        # MATLAB: potinside.m lines 60-95
        quad_faces = np.where(np.any(self._indquad, axis = 1))[0]
        ind1_refine = np.intersect1d(quad_faces, ind1)

        for i1 in ind1_refine:
            pos1_q, w = self._quad(p, i1)
            w = w / p.area[i1]
            nvec_i = p.nvec[i1, :]

            # Impact parameters for refinement
            indimp_mapped = self._indquad[i1, self._indimp]
            i2_q = np.intersect1d(np.where(indimp_mapped)[0], ind2)
            if len(i2_q) == 0:
                continue

            pos2_q = self.impact[self._indimp[i2_q], :]

            x_q = pos1_q[:, 0:1] - pos2_q[:, 0:1].T
            y_q = pos1_q[:, 1:2] - pos2_q[:, 1:2].T
            z_q = np.tile(pos1_q[:, 2:3], (1, pos2_q.shape[0]))

            r_q = msqrt(x_q ** 2 + y_q ** 2)
            rr_q = msqrt(r_q ** 2 + self.width ** 2)

            I_q, Ir_q, Iz_q = self._potwire(
                rr_q, z_q, q, k, self._z[i2_q, 0], self._z[i2_q, 1])

            phi[i1, i2_q] = -1 / self.vel * w @ I_q
            phip[i1, i2_q] = (-1 / self.vel * w
                @ (Iz_q * nvec_i[2]
                   + Ir_q * (x_q * nvec_i[0] + y_q * nvec_i[1]) / rr_q))

        return phi, phip

    # ---- static methods ----

    @staticmethod
    def ene2vel(ene: float) -> float:
        """
        Convert kinetic electron energy in eV to velocity in units of c.

        MATLAB: eelsbase.ene2vel

        Parameters
        ----------
        ene : float
            Electron energy in eV

        Returns
        -------
        vel : float
            Electron velocity in units of speed of light
        """
        return msqrt(1 - 1.0 / (1 + ene / 0.51e6) ** 2)

    # ---- private helper methods ----

    @staticmethod
    def _enclosure(p: object) -> float:
        """
        Maximal radius between centroids and boundary vertices (2D projection).

        MATLAB: @eelsbase/private/enclosure.m

        Parameters
        ----------
        p : particle object

        Returns
        -------
        rad : float
            Maximal 2D enclosure radius
        """
        # MATLAB: enclosure.m
        x = np.zeros((p.n, 4))
        y = np.zeros((p.n, 4))

        faces = p.faces
        if faces.shape[1] >= 4:
            ind_tri = np.where(np.isnan(faces[:, 3]))[0]
            ind_quad = np.where(~np.isnan(faces[:, 3]))[0]
        else:
            ind_tri = np.arange(p.n)
            ind_quad = np.array([], dtype = int)

        # Triangles
        if len(ind_tri) > 0:
            for k in range(3):
                vidx = faces[ind_tri, k].astype(int)
                x[ind_tri, k] = p.pos[ind_tri, 0] - p.verts[vidx, 0]
                y[ind_tri, k] = p.pos[ind_tri, 1] - p.verts[vidx, 1]

        # Quadrilaterals
        if len(ind_quad) > 0:
            for k in range(4):
                vidx = faces[ind_quad, k].astype(int)
                x[ind_quad, k] = p.pos[ind_quad, 0] - p.verts[vidx, 0]
                y[ind_quad, k] = p.pos[ind_quad, 1] - p.verts[vidx, 1]

        rad_arr = msqrt(x ** 2 + y ** 2)
        return np.max(rad_arr)

    @staticmethod
    def _distmin(p: object,
            pos: np.ndarray,
            cutoff: float) -> np.ndarray:
        """
        Minimum distance in 2D between particle faces and positions.

        MATLAB: @eelsbase/private/distmin.m

        Parameters
        ----------
        p : particle object
        pos : ndarray, shape (npos, 2)
            Positions
        cutoff : float
            Consider only values smaller than cutoff

        Returns
        -------
        dmin : ndarray, shape (p.n, npos)
            Minimum distance (NaN where > cutoff)
        """
        from ..misc import nettable

        npos = pos.shape[0]
        faces = p.faces

        # MATLAB: distmin.m lines 17-19
        net, inet = nettable(faces)
        net_sorted = np.sort(net, axis = 1)
        _, unique_idx, i2 = np.unique(
            net_sorted, axis = 0, return_index = True, return_inverse = True)
        net_unique = net_sorted[unique_idx]

        # MATLAB: distmin.m lines 21-29
        xv = np.tile(p.verts[net_unique[:, 0], 0:1], (1, npos))
        yv = np.tile(p.verts[net_unique[:, 0], 1:2], (1, npos))
        dx = np.tile(p.verts[net_unique[:, 1], 0:1], (1, npos)) - xv
        dy = np.tile(p.verts[net_unique[:, 1], 1:2], (1, npos)) - yv
        x = np.tile(pos[:, 0:1].T, (net_unique.shape[0], 1))
        y = np.tile(pos[:, 1:2].T, (net_unique.shape[0], 1))

        # MATLAB: distmin.m lines 31-38
        lam = (dx * (x - xv) + dy * (y - yv)) / np.maximum(dx ** 2 + dy ** 2, np.finfo(float).eps)
        lam = np.clip(lam, 0, 1)
        dnet = msqrt((x - xv - lam * dx) ** 2 + (y - yv - lam * dy) ** 2)

        # MATLAB: distmin.m lines 40-49
        dmin = np.full((p.n, npos), np.nan)
        dnet_full = dnet[i2, :]

        for i in range(npos):
            ind_close = dnet_full[:, i] < cutoff
            if not np.any(ind_close):
                continue
            face_indices = inet[ind_close]
            values = dnet_full[ind_close, i]
            for fi, vi in zip(face_indices, values):
                if np.isnan(dmin[fi, i]) or vi < dmin[fi, i]:
                    dmin[fi, i] = vi

        return dmin

    @staticmethod
    def _inpolygon(x: np.ndarray,
            y: np.ndarray,
            xv: np.ndarray,
            yv: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Check if points are inside/on polygon (like MATLAB inpolygon).

        Parameters
        ----------
        x, y : ndarray
            Test points
        xv, yv : ndarray
            Polygon vertices

        Returns
        -------
        in_mask : ndarray of bool
            Points strictly inside or on boundary
        on_mask : ndarray of bool
            Points on boundary
        """
        from matplotlib.path import Path

        x = np.atleast_1d(x)
        y = np.atleast_1d(y)
        xv = np.atleast_1d(xv)
        yv = np.atleast_1d(yv)

        # Close polygon
        verts = np.column_stack([
            np.append(xv, xv[0]),
            np.append(yv, yv[0])
        ])
        path = Path(verts)
        points = np.column_stack([x, y])

        in_mask = path.contains_points(points)

        # Detect points on boundary
        in_expanded = path.contains_points(points, radius = 1e-10)
        on_mask = in_expanded & ~in_mask

        # Explicit vertex check: points coincident with polygon vertices
        # can be missed by matplotlib Path (impact=0 grazing vertex case).
        vertex_coords = np.column_stack([xv, yv])
        if vertex_coords.size > 0:
            from scipy.spatial.distance import cdist
            dist_to_vertices = cdist(points, vertex_coords)
            min_dist = np.min(dist_to_vertices, axis = 1)
            on_vertex = min_dist < 1e-10
            on_mask = on_mask | on_vertex

        # Combine: in_mask includes points on boundary
        in_mask = in_mask | on_mask

        return in_mask, on_mask

    @staticmethod
    def _potwire(r: np.ndarray,
            z: np.ndarray,
            q: float,
            k: float,
            z0: np.ndarray,
            z1: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Potential for charged wire.

        Integrates exp(i q zz) / sqrt(r^2 + (zz - z)^2) from z0 to z1
        using Gauss-Legendre quadrature.

        MATLAB: Simulation/misc/potwire.m

        Parameters
        ----------
        r : ndarray
            Distance normal to electron beam
        z : ndarray
            Distance along electron beam
        q : float
            Wavenumber of electron beam
        k : float
            Wavenumber of light
        z0 : ndarray
            Beginning of wire segments
        z1 : ndarray
            End of wire segments

        Returns
        -------
        phi : ndarray
            Potential
        phir : ndarray
            Derivative wrt r
        phiz : ndarray
            Derivative wrt z
        """
        # MATLAB: potwire.m lines 25-29
        z0_arr = np.tile(np.reshape(z0, (1, -1)), (r.shape[0], 1)) - z
        z1_arr = np.tile(np.reshape(z1, (1, -1)), (r.shape[0], 1)) - z

        v0 = mlog(z0_arr + msqrt(r ** 2 + z0_arr ** 2))
        v1 = mlog(z1_arr + msqrt(r ** 2 + z1_arr ** 2))

        # MATLAB: potwire.m lines 31-34
        phi = np.zeros_like(r, dtype = complex)
        phir = np.zeros_like(r, dtype = complex)
        phiz = np.zeros_like(r, dtype = complex)

        x_gl, w_gl = lglnodes(10)

        # MATLAB: potwire.m lines 37-49
        for i in range(len(x_gl)):
            v = 0.5 * ((1 - x_gl[i]) * v0 + (1 + x_gl[i]) * v1)
            u = 0.5 * (np.exp(v) - r ** 2 * np.exp(-v))
            rr = msqrt(r ** 2 + u ** 2)
            fac = np.exp(1j * (q * u + k * rr))

            phi += w_gl[i] * fac
            phir += w_gl[i] * r * (1j * k / rr - 1.0 / rr ** 2) * fac
            phiz -= w_gl[i] * u * (1j * k / rr - 1.0 / rr ** 2) * fac

        # MATLAB: potwire.m lines 51-53
        prefactor = 0.5 * (v1 - v0) * np.exp(1j * q * z)
        phi = prefactor * phi
        phir = prefactor * phir
        phiz = prefactor * phiz

        return phi, phir, phiz

    @staticmethod
    def _quad(p: object, face_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get quadrature positions and weights for a single boundary element.

        MATLAB: [pos, w] = quad(p, i1)

        Parameters
        ----------
        p : particle object
        face_idx : int
            Face index

        Returns
        -------
        pos : ndarray, shape (nquad, 3)
            Quadrature positions
        w : ndarray, shape (nquad,)
            Quadrature weights (sum to p.area[face_idx])
        """
        # Prefer full Gauss-Legendre quadrature via quad_integration which
        # matches MATLAB Particle/quad.m. Returns (pos, w_sparse, iface).
        if hasattr(p, 'quad_integration') and callable(p.quad_integration):
            pos, w_sparse, _ = p.quad_integration(np.array([face_idx]))
            # w_sparse has shape (1, nquad); convert to 1-D array summing to area
            w = np.asarray(w_sparse.toarray()).ravel()
            # Only keep quadrature points with non-zero weights for this face
            mask = w != 0
            if not np.all(mask):
                pos = pos[mask, :]
                w = w[mask]
            return pos, w
        # Fallback: single point at centroid. Weight must equal face area so
        # subsequent normalization w/area yields unit weight.
        return p.pos[face_idx:face_idx + 1, :], np.array([p.area[face_idx]])

    def __repr__(self) -> str:
        return 'EELSBase(n_impact={}, vel={:.4f}, width={})'.format(
            self.impact.shape[0], self.vel, self.width)

    def __str__(self) -> str:
        return ('EELS Base:\n'
                '  Impact parameters: {}\n'
                '  Velocity: {:.4f} c\n'
                '  Width: {}\n'
                '  Phiout: {}').format(
            self.impact.shape[0], self.vel, self.width, self.phiout)
