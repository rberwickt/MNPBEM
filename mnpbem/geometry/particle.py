"""
Particle class for discretized surfaces.

Matches MATLAB MNPBEM @particle implementation exactly.
"""

import numpy as np
from scipy.linalg import expm
from scipy.sparse import csr_matrix, diags
from ..utils.quadface import QuadFace as QuadFaceNew
from ..utils.matlab_compat import mcos, msin, matan2, msqrt, masin, macos, mfloor, mceil, mround
from ..geometry.shape_functions import TriangleShape, QuadShape


# Keep old QuadFace for backward compatibility temporarily
class QuadFace(object):
    """
    Integration rules for triangular/quadrilateral boundary elements.

    Matches MATLAB @quadface class.

    Parameters
    ----------
    rule : int
        Integration rule (default: 3, 7-point Dunavant rule)
    npol : int
        Number of points for polar integration (default: 5)
    """

    def __init__(self, rule=3, npol=5):
        """Initialize quadrature rules."""
        self.npol = npol

        # Standard triangle integration (Dunavant rules)
        # Rule 3: 7-point rule (degree 5)
        if rule == 1:
            # 1-point rule (centroid)
            self.x = np.array([1/3])
            self.y = np.array([1/3])
            self.w = np.array([1.0])
        elif rule == 2:
            # 3-point rule
            self.x = np.array([1/6, 2/3, 1/6])
            self.y = np.array([1/6, 1/6, 2/3])
            self.w = np.array([1/3, 1/3, 1/3])
        else:  # rule == 3 (default)
            # 7-point rule (Dunavant)
            a = 0.470142064105115
            b = 0.059715871789770
            c = 0.101286507323456
            self.x = np.array([1/3, a, 1-2*a, a, b, 1-2*b, b])
            self.y = np.array([1/3, a, a, 1-2*a, b, b, 1-2*b])
            w1 = 0.225
            w2 = 0.132394152788506
            w3 = 0.125939180544827
            self.w = np.array([w1, w2, w2, w2, w3, w3, w3])

        # Polar integration for triangles (radial from centroid)
        self._init_polar_tri(npol)
        # Polar integration for quadrilaterals
        self._init_polar_quad(npol)

    def _lglnodes(self, n):
        """
        Legendre-Gauss-Lobatto nodes for integration.

        MATLAB: Misc/integration/lglnodes.m

        Parameters
        ----------
        n : int
            Number of integration points

        Returns
        -------
        x : ndarray
            Integration points in interval [-1, 1]
        w : ndarray
            Integration weights
        """
        n1 = n + 1
        # Use Chebyshev-Gauss-Lobatto nodes as first guess
        x = mcos(np.pi * np.arange(n1) / n)

        # Legendre Vandermonde Matrix
        p = np.zeros((n1, n1))

        # Newton-Raphson iteration
        xold = 2 * np.ones_like(x)
        while np.max(np.abs(x - xold)) > np.finfo(float).eps:
            xold = x.copy()
            p[:, 0] = 1
            p[:, 1] = x

            for k in range(2, n+1):
                p[:, k] = ((2*k - 1) * x * p[:, k-1] - (k - 1) * p[:, k-2]) / k

            x = xold - (x * p[:, n] - p[:, n-1]) / (n1 * p[:, n])

        w = 2 / (n * n1 * p[:, n]**2)

        return x, w

    def _init_polar_tri(self, npol):
        """
        Initialize polar integration for triangles.

        MATLAB: Misc/integration/@quadface/private/init.m lines 25-53
        """
        # MATLAB: if numel(op.npol) ~= 2, op.npol = [1,1] * op.npol; end
        # Scalar npol → broadcast to [npol, npol] (NOT the [7,5] default).
        if np.isscalar(npol):
            npol = [int(npol), int(npol)]

        # Legendre-Gauss-Lobatto nodes for radius and angle
        x1, w1 = self._lglnodes(npol[0])
        rho = 0.5 * (x1 + 1 + 1e-6)  # Transform to (0, 1]

        x2, w2 = self._lglnodes(npol[1])
        phi = (270 + 60 * x2) / 180 * np.pi  # Angular range for one sector

        # Rotation angle for 3 sectors
        phi0 = 120 / 180 * np.pi

        # Make 2D meshgrid
        rho_grid, phi_grid = np.meshgrid(rho, phi, indexing='ij')
        rho_flat = rho_grid.flatten()
        phi_flat = phi_grid.flatten()

        # Radius scaling for triangle geometry
        rad = 1.0 / np.abs(2 * msin(phi_flat))

        # Create 3 rotated sectors (120° apart)
        phi_all = np.hstack([phi_flat, phi_flat + phi0, phi_flat + 2*phi0])
        rho_all = np.tile(rho_flat, 3)
        rad_all = np.tile(rad, 3)

        # Integration points in triangle
        x = mcos(phi_all) * rho_all * rad_all
        y = msin(phi_all) * rho_all * rad_all

        # Transform to unit triangle coordinates
        # MATLAB: (1 - sqrt(3)*x - y)/3, (1 + sqrt(3)*x - y)/3
        x_tri = (1 - msqrt(3) * x - y) / 3
        y_tri = (1 + msqrt(3) * x - y) / 3

        # Integration weights
        w = np.outer(w1, w2).flatten()  # Tensor product of 1D weights
        w = w * rho_flat * rad**2  # Include Jacobian
        w = np.tile(w, 3)  # Replicate for 3 sectors
        w = w / np.sum(w)  # Normalize to sum to 1

        self.x3 = x_tri
        self.y3 = y_tri
        self.w3 = w

    def _init_polar_quad(self, npol):
        """
        Initialize polar integration for quadrilaterals.

        MATLAB: Misc/integration/@quadface/private/init.m lines 55-80
        """
        # MATLAB: if numel(op.npol) ~= 2, op.npol = [1,1] * op.npol; end
        # Scalar npol → broadcast to [npol, npol] (NOT the [7,5] default).
        if np.isscalar(npol):
            npol = [int(npol), int(npol)]

        # Legendre-Gauss-Lobatto nodes for radius and angle
        x1, w1 = self._lglnodes(npol[0])
        rho = 0.5 * (x1 + 1 + 1e-6)  # Transform to (0, 1]

        x2, w2 = self._lglnodes(npol[1])
        phi = (90 + 45 * x2) / 180 * np.pi  # Angular range for one sector

        # Rotation angle for 4 sectors
        phi0 = np.pi / 2

        # Make 2D meshgrid
        rho_grid, phi_grid = np.meshgrid(rho, phi, indexing='ij')
        rho_flat = rho_grid.flatten()
        phi_flat = phi_grid.flatten()

        # Radius scaling for quadrilateral geometry
        rad = 1.0 / np.abs(msin(phi_flat))

        # Create 4 rotated sectors (90° apart)
        phi_all = np.hstack([phi_flat, phi_flat + phi0,
                             phi_flat + 2*phi0, phi_flat + 3*phi0])
        rho_all = np.tile(rho_flat, 4)
        rad_all = np.tile(rad, 4)

        # Integration points in quadrilateral
        x_quad = mcos(phi_all) * rho_all * rad_all
        y_quad = msin(phi_all) * rho_all * rad_all

        # Integration weights
        w = np.outer(w1, w2).flatten()  # Tensor product of 1D weights
        w = w * rho_flat * rad**2  # Include Jacobian
        w = np.tile(w, 4)  # Replicate for 4 sectors
        w = 4 * w / np.sum(w)  # Normalize to sum to 4

        self.x4 = x_quad
        self.y4 = y_quad
        self.w4 = w

    def _max_radius_triangle(self, center, direction):
        """Find max radius from center to triangle edge."""
        # Triangle vertices: (0,0), (1,0), (0,1)
        vertices = np.array([[0, 0], [1, 0], [0, 1]])

        r_max = 10.0  # Large initial value
        for i in range(3):
            v1 = vertices[i]
            v2 = vertices[(i+1) % 3]
            edge = v2 - v1

            # Ray-edge intersection
            denom = direction[0] * edge[1] - direction[1] * edge[0]
            if abs(denom) > 1e-10:
                t = ((v1[0] - center[0]) * edge[1] - (v1[1] - center[1]) * edge[0]) / denom
                s = ((v1[0] - center[0]) * direction[1] - (v1[1] - center[1]) * direction[0]) / denom
                if t > 0 and 0 <= s <= 1:
                    r_max = min(r_max, t)

        return r_max

    def _max_radius_quad(self, center, direction):
        """Find max radius from center to quad edge ([-1,1]x[-1,1])."""
        r_max = 10.0
        edges = [
            (np.array([-1, -1]), np.array([1, -1])),
            (np.array([1, -1]), np.array([1, 1])),
            (np.array([1, 1]), np.array([-1, 1])),
            (np.array([-1, 1]), np.array([-1, -1])),
        ]

        for v1, v2 in edges:
            edge = v2 - v1
            denom = direction[0] * edge[1] - direction[1] * edge[0]
            if abs(denom) > 1e-10:
                t = ((v1[0] - center[0]) * edge[1] - (v1[1] - center[1]) * edge[0]) / denom
                s = ((v1[0] - center[0]) * direction[1] - (v1[1] - center[1]) * direction[0]) / denom
                if t > 0 and 0 <= s <= 1:
                    r_max = min(r_max, t)

        return r_max


class Particle(object):
    """
    Faces and vertices of discretized particle.

    The particle faces can be either triangles or quadrilaterals, or both.
    Matches MATLAB MNPBEM @particle class exactly.

    Parameters
    ----------
    verts : ndarray, shape (nverts, 3)
        Vertex coordinates [x, y, z]
    faces : ndarray, shape (nfaces, 3) or (nfaces, 4)
        Face connectivity (0-indexed vertex indices)
    interp : str
        'flat' or 'curv' for particle boundaries

    Attributes
    ----------
    verts : ndarray
        Vertices
    faces : ndarray, shape (nfaces, 4)
        Triangle or quadrilateral faces (NaN for 4th vertex if triangle)
    pos : ndarray, shape (nfaces, 3)
        Centroid positions of faces
    vec : list of ndarray
        Basis vectors [vec1, vec2, nvec] (matches MATLAB obj.vec cell array)
        vec[0] : First tangent vector (shape: nfaces, 3)
        vec[1] : Second tangent vector (shape: nfaces, 3)
        vec[2] : Normal vector (shape: nfaces, 3)
    area : ndarray, shape (nfaces,)
        Area of each face
    nvec : property -> vec[2]
        Outward normal vectors (matches MATLAB obj.nvec)
    tvec1 : property -> vec[0]
        First tangent vector (matches MATLAB obj.tvec1)
    tvec2 : property -> vec[1]
        Second tangent vector (matches MATLAB obj.tvec2)
    nverts : property
        Number of vertices
    nfaces : property
        Number of faces

    Examples
    --------
    >>> verts = np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1]])
    >>> faces = np.array([[0,1,2], [0,1,3], [0,2,3], [1,2,3]])
    >>> p = Particle(verts, faces)
    >>> print("Particle: {} vertices, {} faces".format(p.nverts, p.nfaces))
        Centroids of faces
    vec : list
        [tvec1, tvec2, nvec] tangential and normal vectors at centroids
    area : ndarray
        Area of faces
    quad : QuadFace
        Quadrature rules for boundary element integration
    verts2 : ndarray or None
        Additional vertices for curved particle boundary
    faces2 : ndarray or None
        Additional faces for curved particle boundary
    """

    def __init__(self, verts, faces=None, interp='flat', norm='on', **kwargs):
        """
        Initialize particle from vertices and faces.

        MATLAB: obj = particle(verts, faces, op, PropertyPair)
        """
        # Return empty particle if no verts
        if verts is None or len(verts) == 0:
            self.verts = np.array([]).reshape(0, 3)
            self.faces = np.array([]).reshape(0, 4)
            self.pos = np.array([]).reshape(0, 3)
            self.vec = [np.array([]).reshape(0, 3)] * 3
            self.area = np.array([])
            self.verts2 = None
            self.faces2 = None
            self.interp = interp
            self.quad = QuadFace()
            return

        verts = np.asarray(verts, dtype=float)

        # Validate verts shape: must be (N, 3). Bare 1D vector is treated as a
        # degenerate "no-faces" stub but never produces faces.
        if verts.ndim == 1:
            if verts.size != 3 and faces is not None:
                raise ValueError(
                    "Particle: 1D 'verts' is only valid as a single point "
                    "(length 3) or with faces=None; got length {} with faces."
                    .format(verts.size))
            verts = verts.reshape(1, 3) if verts.size == 3 else verts.reshape(0, 3)
        elif verts.ndim != 2 or verts.shape[1] != 3:
            raise ValueError(
                "Particle: 'verts' must have shape (N, 3); got {}.".format(verts.shape))

        # Handle face format
        if faces is None:
            self.verts = verts
            self.faces = np.array([]).reshape(0, 4)
        else:
            faces = np.asarray(faces, dtype=float)
            if faces.ndim != 2:
                raise ValueError(
                    "Particle: 'faces' must be 2D; got {}D.".format(faces.ndim))

            if faces.shape[1] == 3:
                # Only triangular elements - add NaN column
                self.verts = verts
                nan_col = np.full((faces.shape[0], 1), np.nan)
                self.faces = np.hstack([faces, nan_col])
            elif faces.shape[1] == 4:
                # Triangular and/or quadrilateral elements
                self.verts = verts
                self.faces = faces
            else:
                # Intermediate points for curved particle boundary
                self.verts2 = verts
                self.faces2 = faces
                # Extract corner vertices
                corner_faces = faces[:, :4].reshape(-1)
                valid_idx = ~np.isnan(corner_faces)
                unique_verts, inv_idx = np.unique(corner_faces[valid_idx].astype(int), return_inverse=True)
                self.verts = verts[unique_verts]
                # Remap face indices
                new_faces = np.full_like(faces[:, :4], np.nan)
                new_faces.flat[valid_idx] = inv_idx
                self.faces = new_faces

        # Curved boundary data (None for flat)
        if not hasattr(self, 'verts2'):
            self.verts2 = None
            self.faces2 = None

        # Quadrature rules
        rule = kwargs.get('rule', 18)  # Use rule=18 (28 points) matching MATLAB default
        npol = kwargs.get('npol', (7, 5))  # (n_radial, n_angular)
        refine = kwargs.get('refine', None)  # MATLAB bemoptions('refine', N)
        self.quad = QuadFaceNew(rule=rule, npol=npol, refine=refine)

        # Interpolation type
        self.interp = interp

        # Compute geometric properties
        if norm != 'off' and len(self.faces) > 0:
            self._norm()
        else:
            self.pos = np.array([]).reshape(0, 3)
            self.vec = [np.array([]).reshape(0, 3)] * 3
            self.area = np.array([])

    # ==================== Properties (MATLAB subsref) ====================

    @property
    def nvec(self):
        """Normal vectors of surface elements (MATLAB: obj.nvec)."""
        return self.vec[2]

    @nvec.setter
    def nvec(self, value):
        """Set normal vectors."""
        self.vec[2] = value

    @property
    def tvec(self):
        """Tangential vectors (MATLAB: obj.tvec)."""
        return [self.vec[0], self.vec[1]]

    @property
    def tvec1(self):
        """First tangential vector (MATLAB: obj.tvec1)."""
        return self.vec[0]

    @tvec1.setter
    def tvec1(self, value):
        """Set first tangential vector."""
        self.vec[0] = value

    @property
    def tvec2(self):
        """Second tangential vector (MATLAB: obj.tvec2)."""
        return self.vec[1]

    @tvec2.setter
    def tvec2(self, value):
        """Set second tangential vector."""
        self.vec[1] = value

    @property
    def nfaces(self):
        """Number of surface elements (MATLAB: obj.nfaces, obj.n, obj.size)."""
        return self.faces.shape[0]

    @property
    def n(self):
        """Number of surface elements (alias)."""
        return self.nfaces

    @property
    def nverts(self):
        """Number of vertices (MATLAB: obj.nverts)."""
        return self.verts.shape[0]

    # ==================== Geometry computation ====================

    def _norm(self):
        """
        Compute auxiliary information for discretized particle surface.

        MATLAB: obj = norm(obj)
        """
        if self.interp == 'flat':
            self._norm_flat()
        else:
            self._norm_curv()

    def _norm_flat(self):
        """
        Compute centroids, areas, and basis vectors for flat elements.

        MATLAB: norm_flat.m
        """
        n = self.faces.shape[0]
        ind3, ind4 = self.index34()

        # Compute centroids
        self.pos = np.zeros((n, 3))

        if len(ind3) > 0:
            f3 = self.faces[ind3, :3].astype(int)
            self.pos[ind3] = (self.verts[f3[:, 0]] +
                              self.verts[f3[:, 1]] +
                              self.verts[f3[:, 2]]) / 3.0

        if len(ind4) > 0:
            f4 = self.faces[ind4].astype(int)
            self.pos[ind4] = (self.verts[f4[:, 0]] +
                              self.verts[f4[:, 1]] +
                              self.verts[f4[:, 2]] +
                              self.verts[f4[:, 3]]) / 4.0

        # Split into triangles
        tri_faces, ind4_split = self.totriangles()

        # Get triangle vertices
        v1 = self.verts[tri_faces[:, 0].astype(int)]
        v2 = self.verts[tri_faces[:, 1].astype(int)]
        v3 = self.verts[tri_faces[:, 2].astype(int)]

        # Triangle vectors
        vec1 = v1 - v2
        vec2 = v3 - v2

        # Normal vector
        nvec = np.cross(vec1, vec2)

        # Area + norms via explicit sqrt(dot(.,.,2)) to match MATLAB FP order.
        # np.linalg.norm uses BLAS dnrm2 which can differ by 1-2 ULP and flips
        # the larger-area tiebreak for non-planar quads (see Fri-Apr-29 tricube
        # corner-face investigation: 7/3168 quad nvec mismatch).
        area = 0.5 * np.sqrt(np.sum(nvec * nvec, axis=1))

        vec1_norm = np.sqrt(np.sum(vec1 * vec1, axis=1, keepdims=True))
        nvec_norm = np.sqrt(np.sum(nvec * nvec, axis=1, keepdims=True))
        vec1 = vec1 / np.maximum(vec1_norm, 1e-14)
        nvec = nvec / np.maximum(nvec_norm, 1e-14)

        # Orthogonal basis
        vec2 = np.cross(nvec, vec1)

        if len(ind4_split) == 0:
            # Only triangles
            self.area = area
            self.vec = [vec1, vec2, nvec]
        else:
            # Accumulate area for quads (two triangles per quad)
            self.area = np.zeros(n)
            for i in range(len(area)):
                if i < n:
                    self.area[i] = area[i]
                else:
                    # Second triangle of quad
                    orig_idx = ind4_split[i - n, 0]
                    self.area[orig_idx] += area[i]

            # Select vectors from larger triangle
            vec1_out = vec1[:n].copy()
            vec2_out = vec2[:n].copy()
            nvec_out = nvec[:n].copy()

            if len(ind4_split) > 0:
                area1 = area[ind4_split[:, 0]]
                area2 = area[ind4_split[:, 1]]
                larger = area2 > area1

                for i, (idx1, idx2) in enumerate(ind4_split):
                    if larger[i]:
                        vec1_out[idx1] = vec1[idx2]
                        vec2_out[idx1] = vec2[idx2]
                        nvec_out[idx1] = nvec[idx2]

            self.vec = [vec1_out, vec2_out, nvec_out]

    def _norm_curv(self):
        """
        Compute centroids, areas, and basis vectors for curved elements.

        MATLAB: norm_curv.m
        """
        n = self.faces.shape[0]

        # Get area from integration weights
        _, w, _ = self.quad_integration()
        self.area = np.array(w.sum(axis=1)).flatten()

        ind3, ind4 = self.index34()
        # Use totriangles to get the 6-column face layout
        # MATLAB: faces = totriangles(obj)
        # For triangles: [v0, v1, v2, e01, e12, e20]
        # For quads (first tri): [v0, v1, v2, e01, e12, centroid]
        faces = self.totriangles()[0]

        # Allocate arrays
        pos = np.zeros((n, 3))
        vec1 = np.zeros((n, 3))
        vec2 = np.zeros((n, 3))

        # Triangular elements
        if len(ind3) > 0:
            # Shape functions at centroid
            tri = np.array([-1, -1, -1, 4, 4, 4]) / 9
            trix = np.array([1, 0, -1, 4, -4, 0]) / 3
            triy = np.array([0, 1, -1, 4, 0, -4]) / 3

            for i in ind3:
                face_idx = faces[i].astype(int)
                for j in range(6):
                    pos[i] += tri[j] * self.verts2[face_idx[j]]
                    vec1[i] += triy[j] * self.verts2[face_idx[j]]
                    vec2[i] += trix[j] * self.verts2[face_idx[j]]

        # Quadrilateral elements
        # MATLAB: pos(ind4,:) = obj.verts2(faces(ind4, 6), :)
        # faces(ind4, 6) is the centroid (6th column of totriangles output)
        if len(ind4) > 0:
            for i in ind4:
                # Centroid is the 6th element (index 5) of the totriangles output
                centroid_idx = int(faces[i, 5])
                pos[i] = self.verts2[centroid_idx]

                # Derivatives using the 6-column totriangles face layout
                trix = np.array([1, 0, -1, 0, 0, 0])
                triy = np.array([0, -1, -1, 2, 2, -2])

                face_idx = faces[i, :6].astype(int)
                for j in range(6):
                    vec1[i] += triy[j] * self.verts2[face_idx[j]]
                    vec2[i] += trix[j] * self.verts2[face_idx[j]]

        # Normalize
        nvec = np.cross(vec1, vec2)
        nvec_norm = np.linalg.norm(nvec, axis=1, keepdims=True)
        nvec = nvec / np.maximum(nvec_norm, 1e-14)

        vec1_norm = np.linalg.norm(vec1, axis=1, keepdims=True)
        vec1 = vec1 / np.maximum(vec1_norm, 1e-14)

        vec2 = np.cross(nvec, vec1)

        self.pos = pos
        self.vec = [vec1, vec2, nvec]

    # ==================== Index methods ====================

    def index34(self, ind=None):
        """
        Index to triangular and quadrilateral boundary elements.

        MATLAB: [ind3, ind4] = index34(obj, ind)

        Parameters
        ----------
        ind : array_like, optional
            Index to specific boundary elements

        Returns
        -------
        ind3 : ndarray
            Indices of triangular faces
        ind4 : ndarray
            Indices of quadrilateral faces
        """
        if ind is None:
            is_tri = np.isnan(self.faces[:, 3])
            ind3 = np.where(is_tri)[0]
            ind4 = np.where(~is_tri)[0]
        else:
            ind = np.asarray(ind)
            is_tri = np.isnan(self.faces[ind, 3])
            ind3 = np.where(is_tri)[0]
            ind4 = np.where(~is_tri)[0]

        return ind3, ind4

    def totriangles(self, ind=None):
        """
        Split quadrilateral face elements to triangles.

        MATLAB: [faces, ind4] = totriangles(obj, ind)

        Returns
        -------
        faces : ndarray
            Triangle faces
        ind4 : ndarray
            Pointer to split quadrilaterals [original_idx, new_idx]
        """
        if self.interp == 'flat':
            return self._totriangles_flat(ind)
        else:
            return self._totriangles_curv(ind)

    def _totriangles_flat(self, ind=None):
        """Split quads to triangles (flat)."""
        if ind is None:
            ind = np.arange(self.nfaces)
        ind = np.asarray(ind)

        _, ind4 = self.index34(ind)

        # Start with first 3 vertices of each face
        faces = self.faces[ind, :3].copy()

        if len(ind4) > 0:
            # Add second triangles for quads: v3, v4, v1
            quad_faces = np.column_stack([
                self.faces[ind[ind4], 2],
                self.faces[ind[ind4], 3],
                self.faces[ind[ind4], 0]
            ])
            faces = np.vstack([faces, quad_faces])

            # Index mapping: [original_quad_idx, new_triangle_idx]
            ind4_out = np.column_stack([
                ind4,
                len(ind) + np.arange(len(ind4))
            ])
        else:
            ind4_out = np.array([]).reshape(0, 2).astype(int)

        return faces, ind4_out

    def _totriangles_curv(self, ind=None):
        """Split quads to triangles (curved)."""
        if ind is None:
            ind = np.arange(self.nfaces)
        ind = np.asarray(ind)

        ind3, ind4 = self.index34(ind)

        # Allocate output
        faces = np.zeros((len(ind), 6))

        # Triangular elements
        if len(ind3) > 0:
            faces[ind3] = self.faces2[ind[ind3]][:, [0, 1, 2, 4, 5, 6]]

        # Quadrilateral elements
        if len(ind4) > 0:
            # First triangle
            faces[ind4] = self.faces2[ind[ind4]][:, [0, 1, 2, 4, 5, 8]]
            # Second triangle
            second_tri = self.faces2[ind[ind4]][:, [2, 3, 0, 6, 7, 8]]
            faces = np.vstack([faces, second_tri])

            ind4_out = np.column_stack([
                ind4,
                len(ind) + np.arange(len(ind4))
            ])
        else:
            ind4_out = np.array([]).reshape(0, 2).astype(int)

        return faces, ind4_out

    def vertices(self, ind, close=False):
        """
        Vertices of indexed face.

        MATLAB: v = vertices(obj, ind, 'close')

        Parameters
        ----------
        ind : int
            Face index
        close : bool
            If True, close the face indices

        Returns
        -------
        v : ndarray
            Vertices of the face
        """
        face = self.faces[ind]
        face = face[~np.isnan(face)].astype(int)

        if close:
            face = np.append(face, face[0])

        return self.verts[face]

    # ==================== Geometry transformations ====================

    def shift(self, vec):
        """
        Shift (translate) particle.

        MATLAB: obj = shift(obj, vec)

        Parameters
        ----------
        vec : array_like, shape (3,)
            Translation vector

        Returns
        -------
        self : Particle
            Shifted particle
        """
        vec = np.asarray(vec)
        self.verts = self.verts + vec
        if self.verts2 is not None:
            self.verts2 = self.verts2 + vec
        self._norm()
        return self

    def rot(self, angle, dir=None):
        """
        Rotate particle.

        MATLAB: obj = rot(obj, angle, dir)

        Parameters
        ----------
        angle : float
            Rotation angle in degrees
        dir : array_like, shape (3,), optional
            Rotation axis (default: z-axis [0,0,1])

        Returns
        -------
        self : Particle
            Rotated particle
        """
        if dir is None:
            dir = np.array([0, 0, 1])
        dir = np.asarray(dir, dtype=float)
        dir = dir / np.linalg.norm(dir)

        # Convert to radians
        angle_rad = angle * np.pi / 180

        # Rotation generators (skew-symmetric matrices)
        j1 = np.array([[0, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float)
        j2 = np.array([[0, 0, 1], [0, 0, 0], [-1, 0, 0]], dtype=float)
        j3 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]], dtype=float)

        # Rotation matrix via matrix exponential
        R = expm(-angle_rad * (dir[0]*j1 + dir[1]*j2 + dir[2]*j3))

        # Rotate vertices
        self.verts = self.verts @ R
        if self.verts2 is not None:
            self.verts2 = self.verts2 @ R

        self._norm()
        return self

    def scale(self, scale_factor):
        """
        Scale particle coordinates.

        MATLAB: obj = scale(obj, scale)

        Parameters
        ----------
        scale_factor : float or array_like
            Scaling factor (scalar or vector for each axis)

        Returns
        -------
        self : Particle
            Scaled particle
        """
        scale_factor = np.asarray(scale_factor)
        self.verts = self.verts * scale_factor
        if self.verts2 is not None:
            self.verts2 = self.verts2 * scale_factor
        self._norm()
        return self

    def flip(self, dir=0):
        """
        Flip particle along given direction (returns new particle).

        MATLAB: obj = flip(obj, dir)

        Parameters
        ----------
        dir : int or list
            Direction to flip (0=x, 1=y, 2=z). Default: 0

        Returns
        -------
        new : Particle
            Flipped particle (original is not modified)
        """
        import copy
        new = copy.deepcopy(self)
        dirs = [dir] if isinstance(dir, (int, np.integer)) else dir
        for d in dirs:
            new.verts[:, d] = -new.verts[:, d]
            if new.verts2 is not None:
                new.verts2[:, d] = -new.verts2[:, d]
        return new.flipfaces()

    def flipfaces(self):
        """
        Flip orientation of surface elements (returns new particle).

        MATLAB: obj = flipfaces(obj)

        Returns
        -------
        new : Particle
            Particle with flipped faces (original is not modified)
        """
        import copy
        new = copy.deepcopy(self)
        ind3, ind4 = new.index34()

        # Flip triangular faces
        if len(ind3) > 0:
            new.faces[ind3, :3] = new.faces[ind3, :3][:, ::-1]

        # Flip quadrilateral faces
        if len(ind4) > 0:
            new.faces[ind4, :4] = new.faces[ind4, :4][:, ::-1]

        # Also flip faces2 if present
        if new.faces2 is not None:
            if len(ind3) > 0:
                # Flip: [v0,v1,v2,m01,m12,m20] -> [v2,v1,v0,m12,m01,m20]
                cols_src = [2, 1, 0, 5, 4, 6]
                cols_dst = [0, 1, 2, 4, 5, 6]
                temp = new.faces2[ind3][:, cols_src].copy()
                new.faces2[np.ix_(ind3, cols_dst)] = temp
            if len(ind4) > 0:
                cols_src = [3, 2, 1, 0, 6, 5, 4, 7]
                cols_dst = [0, 1, 2, 3, 4, 5, 6, 7]
                temp = new.faces2[ind4][:, cols_src].copy()
                new.faces2[np.ix_(ind4, cols_dst)] = temp

        new._norm()
        return new

    # ==================== Selection and merging ====================

    def select(self, index=None, carfun=None, polfun=None, sphfun=None):
        """
        Select parts of discretized particle surface.

        MATLAB: [obj1, obj2] = select(obj, 'PropertyName', PropertyValue)

        Parameters
        ----------
        index : array_like, optional
            Index to selected elements
        carfun : callable, optional
            Function f(x, y, z) returning True for selected elements
        polfun : callable, optional
            Function f(phi, r, z) in cylindrical coordinates
        sphfun : callable, optional
            Function f(phi, theta, r) in spherical coordinates

        Returns
        -------
        obj1 : Particle
            Particle with selected faces
        obj2 : Particle or None
            Complement (if requested)
        """
        x = self.pos[:, 0]
        y = self.pos[:, 1]
        z = self.pos[:, 2]

        if index is not None:
            idx = np.asarray(index)
        elif carfun is not None:
            idx = np.where(carfun(x, y, z))[0]
        elif polfun is not None:
            phi = matan2(y, x)
            r = msqrt(x**2 + y**2)
            idx = np.where(polfun(phi, r, z))[0]
        elif sphfun is not None:
            phi = matan2(y, x)
            r = msqrt(x**2 + y**2 + z**2)
            theta = matan2(msqrt(x**2 + y**2), z)
            idx = np.where(sphfun(phi, np.pi/2 - theta, r))[0]
        else:
            raise ValueError("Must specify index, carfun, polfun, or sphfun")

        obj1 = self._compress(idx)
        obj2 = self._compress(np.setdiff1d(np.arange(self.nfaces), idx))

        return obj1, obj2

    def _compress(self, index):
        """Compress particle and remove unused vertices."""
        if len(index) == 0:
            return Particle(np.array([]).reshape(0, 3), np.array([]).reshape(0, 4))

        faces = self.faces[index].copy()

        # Find unique vertices
        flat_faces = faces[~np.isnan(faces)].astype(int)
        unique_verts = np.unique(flat_faces)

        # Create remapping table
        remap = np.zeros(self.nverts, dtype=int)
        remap[unique_verts] = np.arange(len(unique_verts))

        # Remap faces
        valid_mask = ~np.isnan(faces)
        faces[valid_mask] = remap[faces[valid_mask].astype(int)]

        new_verts = self.verts[unique_verts]

        # Handle curved data
        new_verts2 = None
        new_faces2 = None
        if self.verts2 is not None and self.faces2 is not None:
            faces2 = self.faces2[index].copy()
            flat_faces2 = faces2[~np.isnan(faces2)].astype(int)
            unique_verts2 = np.unique(flat_faces2)
            remap2 = np.zeros(len(self.verts2), dtype=int)
            remap2[unique_verts2] = np.arange(len(unique_verts2))
            valid_mask2 = ~np.isnan(faces2)
            faces2[valid_mask2] = remap2[faces2[valid_mask2].astype(int)]
            new_verts2 = self.verts2[unique_verts2]
            new_faces2 = faces2

        new_particle = Particle(new_verts, faces, interp=self.interp, norm='off')
        new_particle.verts2 = new_verts2
        new_particle.faces2 = new_faces2
        new_particle._norm()

        return new_particle

    def __add__(self, other):
        """Concatenate particles using + operator."""
        return self.vertcat(other)

    def vertcat(self, *others):
        """
        Concatenate particles vertically (returns new particle).

        MATLAB: obj = vertcat(obj1, obj2, obj3, ...)
        MATLAB: obj = [obj1; obj2; obj3; ...]

        Returns
        -------
        new : Particle
            Combined particle surface (originals are not modified)
        """
        new_verts = self.verts.copy()
        new_faces = self.faces.copy()
        new_verts2 = self.verts2.copy() if self.verts2 is not None else None
        new_faces2 = self.faces2.copy() if self.faces2 is not None else None

        for other in others:
            offset = new_verts.shape[0]
            new_faces = np.vstack([new_faces, other.faces + offset])
            new_verts = np.vstack([new_verts, other.verts])

            if new_verts2 is not None and other.verts2 is not None:
                offset2 = new_verts2.shape[0]
                new_faces2 = np.vstack([new_faces2, other.faces2 + offset2])
                new_verts2 = np.vstack([new_verts2, other.verts2])

        new_particle = Particle(new_verts, new_faces, interp=self.interp,
                                norm='off')
        new_particle.verts2 = new_verts2
        new_particle.faces2 = new_faces2
        # Inherit quad rule from source particle (MATLAB vertcat keeps obj.quad).
        # Without this, the concatenated particle defaults to rule=18 / refine=None,
        # losing the higher-order integration set by bemoptions on sub-particles.
        if hasattr(self, 'quad') and self.quad is not None:
            new_particle.quad = self.quad
        new_particle._norm()
        return new_particle

    # ==================== Edge and boundary methods ====================

    def edges(self):
        """
        Find unique edges of particle.

        MATLAB: [net, faces] = edges(obj)

        Returns
        -------
        net : ndarray, shape (nedges, 2)
            List of unique edges (vertex indices)
        edge_faces : ndarray
            Edge indices for each face
        """
        ind3, ind4 = self.index34()
        faces = self.faces

        # Build edge list
        edge_list = []

        # Triangular edges
        if len(ind3) > 0:
            f3 = faces[ind3, :3].astype(int)
            edge_list.extend([
                np.column_stack([f3[:, 0], f3[:, 1]]),
                np.column_stack([f3[:, 1], f3[:, 2]]),
                np.column_stack([f3[:, 2], f3[:, 0]])
            ])

        # Quadrilateral edges
        if len(ind4) > 0:
            f4 = faces[ind4, :4].astype(int)
            edge_list.extend([
                np.column_stack([f4[:, 0], f4[:, 1]]),
                np.column_stack([f4[:, 1], f4[:, 2]]),
                np.column_stack([f4[:, 2], f4[:, 3]]),
                np.column_stack([f4[:, 3], f4[:, 0]])
            ])

        if not edge_list:
            return np.array([]).reshape(0, 2), np.array([])

        all_edges = np.vstack(edge_list)

        # Sort edge vertices and find unique
        sorted_edges = np.sort(all_edges, axis=1)
        net, inv_idx = np.unique(sorted_edges, axis=0, return_inverse=True)

        # Build face edge index
        edge_faces = np.full_like(faces, np.nan)
        offset = 0

        if len(ind3) > 0:
            n3 = len(ind3)
            edge_faces[ind3, 0] = inv_idx[offset:offset + n3]
            edge_faces[ind3, 1] = inv_idx[offset + n3:offset + 2*n3]
            edge_faces[ind3, 2] = inv_idx[offset + 2*n3:offset + 3*n3]
            offset += 3 * n3

        if len(ind4) > 0:
            n4 = len(ind4)
            edge_faces[ind4, 0] = inv_idx[offset:offset + n4]
            edge_faces[ind4, 1] = inv_idx[offset + n4:offset + 2*n4]
            edge_faces[ind4, 2] = inv_idx[offset + 2*n4:offset + 3*n4]
            edge_faces[ind4, 3] = inv_idx[offset + 3*n4:offset + 4*n4]

        return net, edge_faces

    def border(self):
        """
        Find border (single edges) of particle.

        MATLAB: net = border(obj)

        Returns
        -------
        net : ndarray, shape (n_border, 2)
            Border edge list
        """
        ind3, ind4 = self.index34()
        faces = self.faces

        # Build edge list
        edge_list = []

        if len(ind3) > 0:
            f3 = faces[ind3, :3].astype(int)
            edge_list.extend([
                np.column_stack([f3[:, 0], f3[:, 1]]),
                np.column_stack([f3[:, 1], f3[:, 2]]),
                np.column_stack([f3[:, 2], f3[:, 0]])
            ])

        if len(ind4) > 0:
            f4 = faces[ind4, :4].astype(int)
            edge_list.extend([
                np.column_stack([f4[:, 0], f4[:, 1]]),
                np.column_stack([f4[:, 1], f4[:, 2]]),
                np.column_stack([f4[:, 2], f4[:, 3]]),
                np.column_stack([f4[:, 3], f4[:, 0]])
            ])

        if not edge_list:
            return np.array([]).reshape(0, 2)

        all_edges = np.vstack(edge_list)
        sorted_edges = np.sort(all_edges, axis=1)

        # Find edges that appear only once
        _, idx, counts = np.unique(sorted_edges, axis=0,
                                   return_index=True, return_counts=True)
        single_edges = all_edges[idx[counts == 1]]

        return single_edges

    # ==================== Integration methods ====================

    def quad_integration(self, ind=None):
        """
        Quadrature points and weights for boundary element integration.

        MATLAB: [pos, w, iface] = quad(obj, ind)

        Parameters
        ----------
        ind : array_like, optional
            Face indices (default: all faces)

        Returns
        -------
        pos : ndarray
            Integration points
        w : sparse matrix
            Integration weights
        """
        if self.interp == 'flat':
            return self._quad_flat(ind)
        else:
            return self._quad_curv(ind)

    def _quad_flat(self, ind=None):
        """Quadrature for flat elements."""
        if ind is None:
            ind = np.arange(self.nfaces)
        ind = np.asarray(ind)

        # Decompose into triangles
        tri_faces, ind4_split = self._totriangles_flat(ind)

        # Get triangle indices
        ind3 = list(range(len(ind)))
        if len(ind4_split) > 0:
            ind3.extend(ind4_split[:, 0].tolist())

        # Normal vectors and areas
        v1 = self.verts[tri_faces[:, 0].astype(int)]
        v2 = self.verts[tri_faces[:, 1].astype(int)]
        v3 = self.verts[tri_faces[:, 2].astype(int)]

        nvec = np.cross(v2 - v1, v3 - v1)
        area = 0.5 * np.linalg.norm(nvec, axis=1)

        # Integration points and weights
        x, y, w = self.quad.x, self.quad.y, self.quad.w
        m = len(w)
        n_total = m * len(ind3)

        pos = np.zeros((n_total, 3))
        weights = np.zeros(n_total)
        rows = np.zeros(n_total, dtype=int)

        # Shape functions: [x, y, 1-x-y]
        tri_shape = np.column_stack([x, y, 1 - x - y])                    # (m, 3)

        # v1.6.1: vectorise per-triangle Python loop into tensor contractions.
        face_idx = tri_faces[:, :3].astype(int)
        verts_block = self.verts[face_idx]                                # (n_tri, 3, 3)
        pos_block = np.einsum('tc,icd->itd', tri_shape, verts_block)      # (n_tri, m, 3)
        pos = pos_block.reshape(-1, 3)
        w_block = w[np.newaxis, :] * area[:, np.newaxis]                  # (n_tri, m)
        weights = w_block.ravel()
        rows = np.repeat(np.asarray(ind3), m)

        # Create sparse weight matrix
        cols = np.arange(n_total)
        w_sparse = csr_matrix((weights, (rows, cols)), shape=(len(ind), n_total))

        return pos, w_sparse

    def _quad_curv(self, ind=None):
        """Quadrature for curved elements."""
        if ind is None:
            ind = np.arange(self.nfaces)
        ind = np.asarray(ind)

        tri_faces, ind4_split = self._totriangles_curv(ind)

        ind3 = list(range(len(ind)))
        if len(ind4_split) > 0:
            ind3.extend(ind4_split[:, 0].tolist())

        x, y, w = self.quad.x, self.quad.y, self.quad.w
        m = len(w)
        n_total = m * len(ind3)

        pos = np.zeros((n_total, 3))
        weights = np.zeros(n_total)
        rows = np.zeros(n_total, dtype=int)

        # 6-node triangle shape functions
        tri_shape = self._tri6_shape(x, y)
        tri_dx, tri_dy = self._tri6_deriv(x, y)

        # v1.6.1: vectorise per-triangle Python loop into tensor contractions.
        face_idx = tri_faces.astype(int)
        verts_block = self.verts2[face_idx]                               # (n_tri, n_corners, 3)
        pos_block = np.einsum('tc,icd->itd', tri_shape, verts_block)
        posx_block = np.einsum('tc,icd->itd', tri_dx, verts_block)
        posy_block = np.einsum('tc,icd->itd', tri_dy, verts_block)
        nvec_block = np.cross(posx_block, posy_block, axis = 2)
        jac_block = 0.5 * np.linalg.norm(nvec_block, axis = 2)            # (n_tri, m)

        pos = pos_block.reshape(-1, 3)
        weights = (w[np.newaxis, :] * jac_block).ravel()
        rows = np.repeat(np.asarray(ind3), m)

        cols = np.arange(n_total)
        w_sparse = csr_matrix((weights, (rows, cols)), shape=(len(ind), n_total))

        return pos, w_sparse

    def quadpol(self, ind=None):
        """
        Quadrature points and weights for polar integration.

        MATLAB: [pos, weight, row] = quadpol(obj, ind)

        Parameters
        ----------
        ind : array_like, optional
            Face indices

        Returns
        -------
        pos : ndarray
            Integration points
        weight : ndarray
            Integration weights
        row : ndarray
            Face index for each integration point
        """
        if self.interp == 'flat':
            return self._quadpol_flat(ind)
        else:
            return self._quadpol_curv(ind)

    def _quadpol_flat(self, ind=None):
        """Polar quadrature for flat elements.

        v1.6.1: per-face Python loop replaced by tensor contractions.
        For tricube / quad meshes this collapses ``n_face`` Python iterations
        of ``shape @ verts`` (16-byte tensor work) into a single
        ``(n_face, m, n_corners) @ (n_face, n_corners, 3)`` einsum, which
        eliminates the > 95 % construct-time bottleneck observed on dimer
        meshes (``_quadpol_flat`` was 28 s tottime at 1452 faces baseline).
        """
        if ind is None:
            ind = np.arange(self.nfaces)
        ind = np.asarray(ind)

        ind3, ind4 = self.index34(ind)
        q = self.quad

        m3, m4 = len(q.x3), len(q.x4)
        n_total = len(ind3) * m3 + len(ind4) * m4

        pos = np.zeros((n_total, 3))
        weight = np.zeros(n_total)
        row = np.zeros(n_total, dtype = int)

        offset = 0

        # Triangular elements
        if len(ind3) > 0:
            tri_shape = np.column_stack([q.x3, q.y3, 1 - q.x3 - q.y3])  # (m3, 3)
            ind3 = np.asarray(ind3)
            face_idx = self.faces[ind[ind3], :3].astype(int)             # (k3, 3)
            verts_block = self.verts[face_idx]                            # (k3, 3, 3)
            # pos[i,t,:] = tri_shape[t,c] * verts_block[i,c,:]
            pos_block = np.einsum('tc,icd->itd', tri_shape, verts_block)  # (k3, m3, 3)
            pos[offset:offset + len(ind3) * m3] = pos_block.reshape(-1, 3)
            w_block = (q.w3[np.newaxis, :] *
                       self.area[ind[ind3]][:, np.newaxis])               # (k3, m3)
            weight[offset:offset + len(ind3) * m3] = w_block.ravel()
            row[offset:offset + len(ind3) * m3] = np.repeat(ind3, m3)
            offset += len(ind3) * m3

        # Quadrilateral elements
        if len(ind4) > 0:
            quad_shape = self._quad4_shape(q.x4, q.y4)                    # (m4, 4)
            quad_dx, quad_dy = self._quad4_deriv(q.x4, q.y4)              # (m4, 4) each
            ind4 = np.asarray(ind4)
            face_idx = self.faces[ind[ind4], :4].astype(int)              # (k4, 4)
            verts_block = self.verts[face_idx]                            # (k4, 4, 3)
            pos_block = np.einsum('tc,icd->itd', quad_shape, verts_block) # (k4, m4, 3)
            posx_block = np.einsum('tc,icd->itd', quad_dx, verts_block)
            posy_block = np.einsum('tc,icd->itd', quad_dy, verts_block)
            nvec_block = np.cross(posx_block, posy_block, axis = 2)        # (k4, m4, 3)
            jac_block = np.linalg.norm(nvec_block, axis = 2)              # (k4, m4)

            n_block = len(ind4) * m4
            pos[offset:offset + n_block] = pos_block.reshape(-1, 3)
            w_block = q.w4[np.newaxis, :] * jac_block                     # (k4, m4)
            weight[offset:offset + n_block] = w_block.ravel()
            row[offset:offset + n_block] = np.repeat(ind4, m4)
            offset += n_block

        return pos, weight, row

    def _quadpol_curv(self, ind=None):
        """Polar quadrature for curved elements."""
        if ind is None:
            ind = np.arange(self.nfaces)
        ind = np.asarray(ind)

        ind3, ind4 = self.index34(ind)
        q = self.quad
        faces = self.faces2[ind]

        m3, m4 = len(q.x3), len(q.x4)
        n_total = len(ind3) * m3 + len(ind4) * m4

        pos = np.zeros((n_total, 3))
        weight = np.zeros(n_total)
        row = np.zeros(n_total, dtype=int)

        offset = 0

        # Triangular elements
        if len(ind3) > 0:
            tri_shape = self._tri6_shape(q.x3, q.y3)
            tri_dx, tri_dy = self._tri6_deriv(q.x3, q.y3)

            for i in ind3:
                it = slice(offset, offset + m3)
                face_idx = faces[i, [0, 1, 2, 4, 5, 6]].astype(int)

                pos[it] = tri_shape @ self.verts2[face_idx]
                posx = tri_dx @ self.verts2[face_idx]
                posy = tri_dy @ self.verts2[face_idx]

                nvec = np.cross(posx, posy)
                jac = 0.5 * np.linalg.norm(nvec, axis=1)

                weight[it] = q.w3 * jac
                row[it] = i
                offset += m3

        # Quadrilateral elements
        if len(ind4) > 0:
            quad_shape = self._quad9_shape(q.x4, q.y4)
            quad_dx, quad_dy = self._quad9_deriv(q.x4, q.y4)

            for i in ind4:
                it = slice(offset, offset + m4)
                face_idx = faces[i, :9].astype(int)

                pos[it] = quad_shape @ self.verts2[face_idx]
                posx = quad_dx @ self.verts2[face_idx]
                posy = quad_dy @ self.verts2[face_idx]

                nvec = np.cross(posx, posy)
                jac = np.linalg.norm(nvec, axis=1)

                weight[it] = q.w4 * jac
                row[it] = i
                offset += m4

        return pos, weight, row

    def quad(self, ind=None):
        """
        Quadrature points and weights for boundary element integration.

        MATLAB: [pos, w, iface] = quad(obj, ind)

        Parameters
        ----------
        ind : array_like, optional
            Face indices

        Returns
        -------
        pos : ndarray, shape (n_points, 3)
            Integration point positions
        w : scipy.sparse matrix, shape (n_faces, n_points)
            Integration weights (sparse for efficiency)
        iface : ndarray, shape (n_points,)
            Face index for each integration point
        """
        if self.interp == 'flat':
            return self._quad_flat(ind)
        else:
            return self._quad_curv(ind)

    def _quad_flat(self, ind=None):
        """
        Boundary element quadrature for flat surfaces.

        MATLAB: /Particles/@particle/private/quad_flat.m
        """
        if ind is None:
            ind = np.arange(self.nfaces)
        ind = np.asarray(ind)

        # Decompose quads into triangles
        faces_tri, ind4 = self._totriangles(ind)

        # Index to triangles
        ind3 = np.arange(len(ind))
        if len(ind4) > 0:
            ind3 = np.hstack([ind3, ind4[:, 0]])

        # Normal vectors of triangular elements
        v1 = self.verts[faces_tri[:, 1].astype(int)] - self.verts[faces_tri[:, 0].astype(int)]
        v2 = self.verts[faces_tri[:, 2].astype(int)] - self.verts[faces_tri[:, 0].astype(int)]
        nvec = np.cross(v1, v2)
        area = 0.5 * np.linalg.norm(nvec, axis=1)

        # Integration points and weights
        q = self.quad
        x, y, w = q.x, q.y, q.w
        m = len(w)  # Number of integration points

        # Total number of points
        n_total = m * len(ind3)

        # Allocate arrays
        pos = np.zeros((n_total, 3))
        weight = np.zeros(n_total)
        row = np.zeros(n_total, dtype=int)
        col = np.zeros(n_total, dtype=int)

        # Triangular shape functions
        tri_shape = np.column_stack([x, y, 1 - x - y])                    # (m, 3)

        # v1.6.1: vectorise per-triangle Python loop into tensor contractions.
        # face_idx: (n_tri, 3) — vertex indices for each triangle.
        face_idx = faces_tri[:, :3].astype(int)
        verts_block = self.verts[face_idx]                                # (n_tri, 3, 3)
        # pos[i, t, :] = tri_shape[t, c] * verts_block[i, c, :]
        pos_block = np.einsum('tc,icd->itd', tri_shape, verts_block)      # (n_tri, m, 3)
        pos = pos_block.reshape(-1, 3)
        # weights = w[t] * area[i]
        w_block = w[np.newaxis, :] * area[:, np.newaxis]                  # (n_tri, m)
        weight = w_block.ravel()
        # row[i, t] = ind3[i]
        row = np.repeat(np.asarray(ind3), m)
        col = np.arange(n_total)

        # Create sparse weight matrix
        from scipy.sparse import csr_matrix
        w_sparse = csr_matrix((weight, (row, col)), shape=(len(ind), n_total))

        # Face index for integration points
        iface = row

        return pos, w_sparse, iface

    def _quad_curv(self, ind=None):
        """
        Boundary element quadrature for curved surfaces.

        MATLAB: /Particles/@particle/private/quad_curv.m
        """
        if ind is None:
            ind = np.arange(self.nfaces)
        ind = np.asarray(ind)

        if self.verts2 is None or self.faces2 is None:
            raise ValueError("Curved integration requires verts2 and faces2")

        # Get curved faces
        faces = self.faces2[ind]

        # Decompose into triangles
        faces_tri, ind4 = self._totriangles_curv(ind)

        # Index to triangles
        ind3 = np.arange(len(ind))
        if len(ind4) > 0:
            ind3 = np.hstack([ind3, ind4[:, 0]])

        # Integration points and weights
        q = self.quad
        x, y, w = q.x, q.y, q.w
        m = len(w)

        # Total number of points
        n_total = m * len(ind3)

        # Allocate arrays
        pos = np.zeros((n_total, 3))
        weight = np.zeros(n_total)
        row = np.zeros(n_total, dtype=int)
        col = np.zeros(n_total, dtype=int)

        # 6-node triangle shape functions
        tri_shape = TriangleShape(6)

        # Loop over triangular elements
        offset = 0
        for i, idx in enumerate(ind3):
            it = slice(offset, offset + m)

            # 6-node vertices for this triangle
            face_idx = faces_tri[i, :6].astype(int)
            verts_6 = self.verts2[face_idx]

            # Interpolate positions
            N = tri_shape(x, y)
            pos[it] = N @ verts_6

            # Compute Jacobian for area
            Nx = tri_shape.x(x, y)
            Ny = tri_shape.y(x, y)
            posx = Nx @ verts_6
            posy = Ny @ verts_6
            nvec = np.cross(posx, posy)
            jac = np.linalg.norm(nvec, axis=1)

            # Integration weights
            weight[it] = 0.5 * w * jac  # 0.5 factor for triangle

            # Row and column indices
            row[it] = idx
            col[it] = np.arange(offset, offset + m)

            offset += m

        # Create sparse weight matrix
        from scipy.sparse import csr_matrix
        w_sparse = csr_matrix((weight, (row, col)), shape=(len(ind), n_total))

        # Face index for integration points
        iface = row

        return pos, w_sparse, iface

    def _totriangles(self, ind):
        """Decompose quadrilaterals into triangles (flat)."""
        faces = self.faces[ind]
        n = len(ind)

        # Check which are quads
        is_quad = ~np.isnan(faces[:, 3])
        n_quads = np.sum(is_quad)

        if n_quads == 0:
            return faces[:, :3], np.array([]).reshape(0, 2)

        # All elements become triangles
        faces_tri = np.zeros((n + n_quads, 3))
        faces_tri[:n, :] = faces[:, :3]

        # Second triangle for quads
        quad_indices = np.where(is_quad)[0]
        faces_tri[n:, 0] = faces[is_quad, 0]
        faces_tri[n:, 1] = faces[is_quad, 2]
        faces_tri[n:, 2] = faces[is_quad, 3]

        # Index mapping: which original faces do the extra triangles belong to
        ind4 = np.column_stack([quad_indices, np.arange(n, n + n_quads)])

        return faces_tri, ind4

    # ==================== Shape functions ====================

    @staticmethod
    def _tri6_shape(x, y):
        """6-node triangle shape functions."""
        x, y = np.atleast_1d(x), np.atleast_1d(y)
        L1, L2, L3 = x, y, 1 - x - y

        # Shape functions for 6-node triangle (corners + midpoints)
        N = np.zeros((len(x), 6))
        N[:, 0] = L1 * (2*L1 - 1)  # Corner 1
        N[:, 1] = L2 * (2*L2 - 1)  # Corner 2
        N[:, 2] = L3 * (2*L3 - 1)  # Corner 3
        N[:, 3] = 4 * L1 * L2      # Midpoint 12
        N[:, 4] = 4 * L2 * L3      # Midpoint 23
        N[:, 5] = 4 * L3 * L1      # Midpoint 31

        return N

    @staticmethod
    def _tri6_deriv(x, y):
        """Derivatives of 6-node triangle shape functions."""
        x, y = np.atleast_1d(x), np.atleast_1d(y)
        L1, L2, L3 = x, y, 1 - x - y

        # dN/dL1, dN/dL2 derivatives, then chain rule for dN/dx, dN/dy
        dNdx = np.zeros((len(x), 6))
        dNdy = np.zeros((len(x), 6))

        # dN/dx = dN/dL1 (since L1=x)
        dNdx[:, 0] = 4*L1 - 1
        dNdx[:, 1] = 0
        dNdx[:, 2] = -4*L3 + 1
        dNdx[:, 3] = 4*L2
        dNdx[:, 4] = -4*L2
        dNdx[:, 5] = 4*(L3 - L1)

        # dN/dy = dN/dL2 (since L2=y)
        dNdy[:, 0] = 0
        dNdy[:, 1] = 4*L2 - 1
        dNdy[:, 2] = -4*L3 + 1
        dNdy[:, 3] = 4*L1
        dNdy[:, 4] = 4*(L3 - L2)
        dNdy[:, 5] = -4*L1

        return dNdx, dNdy

    @staticmethod
    def _quad4_shape(x, y):
        """4-node quadrilateral shape functions (bilinear)."""
        x, y = np.atleast_1d(x), np.atleast_1d(y)

        N = np.zeros((len(x), 4))
        N[:, 0] = 0.25 * (1 - x) * (1 - y)
        N[:, 1] = 0.25 * (1 + x) * (1 - y)
        N[:, 2] = 0.25 * (1 + x) * (1 + y)
        N[:, 3] = 0.25 * (1 - x) * (1 + y)

        return N

    @staticmethod
    def _quad4_deriv(x, y):
        """Derivatives of 4-node quad shape functions."""
        x, y = np.atleast_1d(x), np.atleast_1d(y)

        dNdx = np.zeros((len(x), 4))
        dNdx[:, 0] = -0.25 * (1 - y)
        dNdx[:, 1] = 0.25 * (1 - y)
        dNdx[:, 2] = 0.25 * (1 + y)
        dNdx[:, 3] = -0.25 * (1 + y)

        dNdy = np.zeros((len(x), 4))
        dNdy[:, 0] = -0.25 * (1 - x)
        dNdy[:, 1] = -0.25 * (1 + x)
        dNdy[:, 2] = 0.25 * (1 + x)
        dNdy[:, 3] = 0.25 * (1 - x)

        return dNdx, dNdy

    @staticmethod
    def _quad9_shape(x, y):
        """9-node quadrilateral shape functions (biquadratic)."""
        x, y = np.atleast_1d(x), np.atleast_1d(y)

        N = np.zeros((len(x), 9))

        # Corner nodes
        N[:, 0] = 0.25 * x * (x - 1) * y * (y - 1)
        N[:, 1] = 0.25 * x * (x + 1) * y * (y - 1)
        N[:, 2] = 0.25 * x * (x + 1) * y * (y + 1)
        N[:, 3] = 0.25 * x * (x - 1) * y * (y + 1)

        # Edge midpoints
        N[:, 4] = 0.5 * (1 - x**2) * y * (y - 1)
        N[:, 5] = 0.5 * x * (x + 1) * (1 - y**2)
        N[:, 6] = 0.5 * (1 - x**2) * y * (y + 1)
        N[:, 7] = 0.5 * x * (x - 1) * (1 - y**2)

        # Center node
        N[:, 8] = (1 - x**2) * (1 - y**2)

        return N

    @staticmethod
    def _quad9_deriv(x, y):
        """Derivatives of 9-node quad shape functions."""
        x, y = np.atleast_1d(x), np.atleast_1d(y)

        dNdx = np.zeros((len(x), 9))
        dNdy = np.zeros((len(x), 9))

        # Corner nodes
        dNdx[:, 0] = 0.25 * (2*x - 1) * y * (y - 1)
        dNdx[:, 1] = 0.25 * (2*x + 1) * y * (y - 1)
        dNdx[:, 2] = 0.25 * (2*x + 1) * y * (y + 1)
        dNdx[:, 3] = 0.25 * (2*x - 1) * y * (y + 1)

        dNdy[:, 0] = 0.25 * x * (x - 1) * (2*y - 1)
        dNdy[:, 1] = 0.25 * x * (x + 1) * (2*y - 1)
        dNdy[:, 2] = 0.25 * x * (x + 1) * (2*y + 1)
        dNdy[:, 3] = 0.25 * x * (x - 1) * (2*y + 1)

        # Edge midpoints
        dNdx[:, 4] = -x * y * (y - 1)
        dNdx[:, 5] = 0.5 * (2*x + 1) * (1 - y**2)
        dNdx[:, 6] = -x * y * (y + 1)
        dNdx[:, 7] = 0.5 * (2*x - 1) * (1 - y**2)

        dNdy[:, 4] = 0.5 * (1 - x**2) * (2*y - 1)
        dNdy[:, 5] = -x * (x + 1) * y
        dNdy[:, 6] = 0.5 * (1 - x**2) * (2*y + 1)
        dNdy[:, 7] = -x * (x - 1) * y

        # Center node
        dNdx[:, 8] = -2 * x * (1 - y**2)
        dNdy[:, 8] = -2 * (1 - x**2) * y

        return dNdx, dNdy

    # ==================== Interpolation mode ====================

    def flat(self):
        """
        Set to flat interpolation mode.

        MATLAB: obj = flat(obj)
        """
        self.interp = 'flat'
        self._norm()
        return self

    def curved(self, key='flat'):
        """
        Set to curved interpolation mode.

        MATLAB: obj = curved(obj)

        Parameters
        ----------
        key : str
            'flat' or 'curv' for midpoint computation
        """
        # MATLAB: if verts2 is already set (e.g. from trispheresegment
        # which projects midpoints onto sphere), skip midpoints/refine.
        if self.verts2 is None:
            self.midpoints(key)

        self.interp = 'curv'
        self._norm()
        return self

    def midpoints(self, key='flat'):
        """
        Add midpoints for curved particle boundaries.

        MATLAB: obj = midpoints(obj, key)

        Parameters
        ----------
        key : str
            'flat' or 'curv' for midpoint computation

        Returns
        -------
        self : Particle
            Particle with added midpoints
        """
        self.interp = key

        if key == 'flat':
            # Add midpoints for flat boundary elements
            net, edge_faces = self.edges()
            n = self.nverts

            # Midpoint vertices
            midpts = 0.5 * (self.verts[net[:, 0]] + self.verts[net[:, 1]])
            self.verts2 = np.vstack([self.verts, midpts])

            ind3, ind4 = self.index34()

            # Allocate faces2
            self.faces2 = np.full((self.nfaces, 9), np.nan)
            self.faces2[:, :4] = self.faces

            # Extend face list for triangles
            if len(ind3) > 0:
                self.faces2[ind3, 4:7] = n + edge_faces[ind3, :3]

            # Extend face list for quadrilaterals
            if len(ind4) > 0:
                self.faces2[ind4, 4:8] = n + edge_faces[ind4, :4]
                # Add centroids
                centroid_idx = self.verts2.shape[0] + np.arange(len(ind4))
                self.faces2[ind4, 8] = centroid_idx

                # Compute centroids
                f4 = self.faces[ind4].astype(int)
                centroids = 0.25 * (self.verts[f4[:, 0]] + self.verts[f4[:, 1]] +
                                    self.verts[f4[:, 2]] + self.verts[f4[:, 3]])
                self.verts2 = np.vstack([self.verts2, centroids])
        else:
            # Use curvature-based refinement
            self._refine()

        if self.interp == 'curv':
            self._norm()

        return self

    def _refine(self):
        """
        Refine particle boundary using curvature (B-spline interpolation).

        MATLAB: refine.m — uses vertex_neighbours, edge_tangents,
        make_halfway_vertices to compute B-spline interpolated midpoints.
        """
        from .particle import Particle as _P  # avoid circular

        # Step 1: split quads into triangles for neighbor finding
        flat_p = _P(self.verts.copy(), self.faces.copy())
        flat_p.interp = 'flat'
        faces_tri, ind4_split = flat_p._totriangles_flat()
        ind3_mask = np.isnan(self.faces[:, 3]) if self.faces.shape[1] > 3 else np.ones(self.nfaces, dtype=bool)

        # Step 2: vertex neighbours
        neighbours = self._vertex_neighbours()

        # Step 3: edge tangents and velocities
        tangents, velocities, edge_index = self._edge_tangents(neighbours)

        # Step 4: make halfway vertices using B-spline
        verts2_new, halfway_map = self._make_halfway_vertices(
            tangents, velocities, edge_index, neighbours)
        self.verts2 = verts2_new

        # Step 5: build faces2 (MATLAB: refine.m lines 22-49)
        # Reconstruct faces using halfway vertex indices
        faces_tri_int = faces_tri[:, :3].astype(int)
        n_tri = len(faces_tri_int)

        # Find halfway vertex for each edge of each triangle
        new_mid = np.zeros((n_tri, 3), dtype=int)
        for i in range(n_tri):
            v1, v2, v3 = faces_tri_int[i]
            e12 = tuple(sorted([v1, v2]))
            e23 = tuple(sorted([v2, v3]))
            e31 = tuple(sorted([v3, v1]))
            new_mid[i, 0] = halfway_map.get(e12, 0)
            new_mid[i, 1] = halfway_map.get(e23, 0)
            new_mid[i, 2] = halfway_map.get(e31, 0)

        # Build faces2
        n_faces = self.nfaces
        self.faces2 = np.full((n_faces, 9), np.nan)
        self.faces2[:, :4] = self.faces

        ind3 = np.where(ind3_mask)[0]
        ind4 = np.where(~ind3_mask)[0]

        # Triangular elements: faces2[i, 4:7] = midpoints of edges 01, 12, 20
        if len(ind3) > 0:
            self.faces2[ind3, 4:7] = new_mid[ind3]

        # Quadrilateral elements
        if len(ind4) > 0:
            # ind4_split maps quad faces to pairs of triangles
            for qi, face_idx in enumerate(ind4):
                tri_a = face_idx  # first triangle in faces_tri
                tri_b = n_faces + qi  # second triangle (appended by totriangles)
                # faces2[face, 4:9] = [mid_01, mid_12, mid_23, mid_30, centroid]
                # MATLAB: faces2(ind4a, 5:9) = [faces(ind4a, [1,2]), faces(ind4b, [1,2,3])]
                self.faces2[face_idx, 4:6] = new_mid[tri_a, :2]
                if tri_b < n_tri:
                    self.faces2[face_idx, 6:9] = new_mid[tri_b]

    def _vertex_neighbours(self):
        """
        Find neighboring vertices for each vertex.

        MATLAB: vertex_neighbours.m

        Returns
        -------
        neighbours : list of ndarray
            List where neighbours[i] contains indices of vertices adjacent to vertex i
        """
        # Only works with triangular faces
        ind3, ind4 = self.index34()

        if len(ind4) > 0:
            # Convert quads to triangles for neighbor finding
            faces_tri, _ = self._totriangles_flat()
        else:
            faces_tri = self.faces[:, :3]

        faces_tri = faces_tri.astype(int)

        # Neighbor cell array
        neighbours = [[] for _ in range(self.nverts)]

        # Loop through all faces
        for face in faces_tri:
            if not np.isnan(face).any():
                v1, v2, v3 = face[:3]
                # Add neighbors for each vertex
                neighbours[v1].extend([v2, v3])
                neighbours[v2].extend([v3, v1])
                neighbours[v3].extend([v1, v2])

        # Sort neighbors in rotational order
        for i in range(self.nverts):
            if not neighbours[i]:
                neighbours[i] = np.array([], dtype=int)
                continue

            neigh_flat = np.array(neighbours[i])

            # Find starting edge (for boundary vertices)
            start = 0
            for idx in range(0, len(neigh_flat), 2):
                found = False
                for idx2 in range(1, len(neigh_flat), 2):
                    if neigh_flat[idx] == neigh_flat[idx2]:
                        found = True
                        break
                if not found:
                    start = idx
                    break

            # Build ordered neighbor list
            ordered = []
            if len(neigh_flat) >= 2:
                ordered.append(neigh_flat[start])
                ordered.append(neigh_flat[start + 1])

                # Add remaining neighbors in rotational order
                for _ in range(len(neigh_flat) // 2 - 1):
                    found = False
                    for idx in range(0, len(neigh_flat), 2):
                        if neigh_flat[idx] == ordered[-1]:
                            if neigh_flat[idx + 1] not in ordered:
                                ordered.append(neigh_flat[idx + 1])
                                found = True
                                break
                    if not found:
                        # Handle boundary vertices
                        for val in neigh_flat:
                            if val not in ordered:
                                ordered.append(val)

            neighbours[i] = np.array(ordered, dtype=int)

        return neighbours

    def _edge_tangents(self, neighbours):
        """
        Compute edge tangent vectors and velocities.

        MATLAB: edge_tangents.m

        Parameters
        ----------
        neighbours : list of ndarray
            Neighbor list from _vertex_neighbours

        Returns
        -------
        tangents : ndarray, shape (n_edges, 3)
            Tangent vectors for each edge
        velocities : ndarray, shape (n_edges,)
            Edge velocities
        edge_index : ndarray, shape (n_edges, 2)
            Edge vertex indices [v1, v2]
        """
        tangents = []
        velocities = []
        edge_index = []

        for i in range(self.nverts):
            P = self.verts[i]
            Pneig_idx = neighbours[i]

            if len(Pneig_idx) == 0:
                continue

            # Find opposite vertices
            n_neigh = len(Pneig_idx)
            Pn = np.zeros((n_neigh, 3))
            Pnop = np.zeros((n_neigh, 3))

            for k in range(n_neigh):
                Pn[k] = self.verts[Pneig_idx[k]]

                if n_neigh % 2 == 0:
                    # Even number of neighbors
                    opp_idx = (k + n_neigh // 2) % n_neigh
                    Pnop[k] = self.verts[Pneig_idx[opp_idx]]
                else:
                    # Odd number - interpolate
                    opp = k + n_neigh / 2
                    idx1 = int(mfloor(opp)) % n_neigh
                    idx2 = int(mceil(opp)) % n_neigh
                    Pnop[k] = 0.5 * (self.verts[Pneig_idx[idx1]] +
                                     self.verts[Pneig_idx[idx2]])

            # Compute tangent for each edge
            for j in range(n_neigh):
                # Edge lengths
                Ec = np.linalg.norm(Pn[j] - P) + 1e-14
                Eb = np.linalg.norm(Pnop[j] - P) + 1e-14
                Ea = np.linalg.norm(Pn[j] - Pnop[j]) + 1e-14

                # Triangle area using Heron's formula
                s = (Ea + Eb + Ec) / 2
                h = (2 / Ea) * msqrt(s * (s - Ea) * (s - Eb) * (s - Ec)) + 1e-14
                x = (Ea**2 - Eb**2 + Ec**2) / (2 * Ea)

                # 2D triangle tangent
                Np = np.array([-h, x])
                Np = Np / (np.linalg.norm(Np) + 1e-14)
                Ns = np.array([h, Ea - x])
                Ns = Ns / (np.linalg.norm(Ns) + 1e-14)
                Nb = Np + Ns
                Tb = np.array([Nb[1], -Nb[0]])

                # Back to 3D coordinates
                Pm = (Pn[j] * x + Pnop[j] * (Ea - x)) / Ea
                X3 = (Pn[j] - Pnop[j]) / Ea
                Y3 = (P - Pm) / h

                # 2D tangent to 3D tangent
                Tb3D = X3 * Tb[0] + Y3 * Tb[1]
                Tb3D = Tb3D / (np.linalg.norm(Tb3D) + 1e-14)

                # Edge velocity
                Vv = 0.5 * (Ec + 0.5 * Ea)

                tangents.append(Tb3D)
                velocities.append(Vv)
                edge_index.append([i, Pneig_idx[j]])

        return (np.array(tangents), np.array(velocities),
                np.array(edge_index, dtype=int))

    def _make_halfway_vertices(self, tangents, velocities, edge_index, neighbours):
        """
        Create halfway vertices using B-spline interpolation.

        MATLAB: make_halfway_vertices.m

        Returns
        -------
        verts_out : ndarray
            Extended vertex array including halfway points
        halfway_map : dict
            Map from edge tuple to halfway vertex index
        """
        # Build edge lookup
        edge_lookup = {}
        for idx, (v1, v2) in enumerate(edge_index):
            edge_lookup[(v1, v2)] = idx

        verts_out = [self.verts.copy()]
        halfway_map = {}

        for i in range(self.nverts):
            Pneig = neighbours[i]

            for j in Pneig:
                edge = tuple(sorted([i, j]))
                if edge in halfway_map:
                    continue

                # Get tangent and velocity for edge i -> j
                if (i, j) in edge_lookup:
                    idx_a = edge_lookup[(i, j)]
                    Va, Ea = velocities[idx_a], tangents[idx_a]
                else:
                    Va, Ea = 0, np.zeros(3)

                # Get tangent and velocity for edge j -> i
                if (j, i) in edge_lookup:
                    idx_b = edge_lookup[(j, i)]
                    Vb, Eb = velocities[idx_b], tangents[idx_b]
                else:
                    Vb, Eb = 0, np.zeros(3)

                # B-spline control points
                P0 = self.verts[i]
                P3 = self.verts[j]
                P1 = P0 + Ea * Va / 3
                P2 = P3 + Eb * Vb / 3

                # Cubic Bezier at t=0.5
                c = 3 * (P1 - P0)
                b = 3 * (P2 - P1) - c
                a = P3 - P0 - c - b

                halfway = a * 0.125 + b * 0.25 + c * 0.5 + P0

                halfway_idx = len(verts_out[0]) + len(halfway_map)
                halfway_map[edge] = halfway_idx
                verts_out.append(halfway[np.newaxis, :])

        return np.vstack(verts_out), halfway_map

    def _makenewfacelist(self, halfway_map):
        """
        Create new face list using 4-split method.

        MATLAB: makenewfacelist.m

        Parameters
        ----------
        halfway_map : dict
            Map from edge tuple to halfway vertex index

        Returns
        -------
        faces_new : ndarray
            New refined face list (4x original faces)
        """
        ind3, _ = self.index34()

        if len(ind3) == 0:
            return self.faces

        faces_new = []

        for i in ind3:
            face = self.faces[i, :3].astype(int)
            v1, v2, v3 = face

            # Get halfway vertices
            e12 = tuple(sorted([v1, v2]))
            e23 = tuple(sorted([v2, v3]))
            e31 = tuple(sorted([v3, v1]))

            va = halfway_map.get(e12, v1)
            vb = halfway_map.get(e23, v2)
            vc = halfway_map.get(e31, v3)

            # Create 4 new triangles
            faces_new.append([v1, va, vc])
            faces_new.append([va, v2, vb])
            faces_new.append([vc, vb, v3])
            faces_new.append([va, vb, vc])

        return np.array(faces_new)

    # ==================== Curvature computation ====================

    def curvature(self):
        """
        Compute curvature of discretized particle surface.

        MATLAB: curvature(obj)

        Returns
        -------
        curv : dict
            Dictionary with curvature information:
            - mean: Mean curvature at vertices
            - gauss: Gaussian curvature at vertices
            - dir1: First principal direction
            - dir2: Second principal direction
            - lambda1: First principal curvature
            - lambda2: Second principal curvature

        Note
        ----
        This is a simplified implementation. MATLAB version uses patchcurvature
        which implements more sophisticated curvature estimation algorithms.
        """
        # Get triangular faces only
        ind3, ind4 = self.index34()

        # Convert to triangles
        faces = self.faces[ind3, :3].copy()
        if len(ind4) > 0:
            # Split quads into triangles
            quad_faces = self.faces[ind4, :4]
            tri1 = quad_faces[:, [0, 1, 2]]
            tri2 = quad_faces[:, [2, 3, 0]]
            faces = np.vstack([faces, tri1, tri2])

        # Clean the mesh
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            temp_particle = Particle(self.verts, faces)
            temp_particle = temp_particle.clean()

        # Simplified curvature calculation using discrete operators
        nverts = temp_particle.nverts
        nfaces = temp_particle.nfaces

        # Initialize outputs
        mean_curv = np.zeros(nverts)
        gauss_curv = np.zeros(nverts)
        dir1 = np.zeros((nverts, 3))
        dir2 = np.zeros((nverts, 3))
        lambda1 = np.zeros(nverts)
        lambda2 = np.zeros(nverts)

        # Compute vertex normals by averaging face normals
        vert_normals = np.zeros((nverts, 3))
        vert_area = np.zeros(nverts)

        for i in range(nfaces):
            face = temp_particle.faces[i, :3].astype(int)
            v0, v1, v2 = temp_particle.verts[face]

            # Face normal (already computed in temp_particle.nvec)
            normal = temp_particle.nvec[i]
            area = temp_particle.area[i]

            # Add weighted normal to each vertex
            for v_idx in face:
                vert_normals[v_idx] += normal * area
                vert_area[v_idx] += area

        # Normalize vertex normals
        vert_area[vert_area == 0] = 1
        vert_normals = vert_normals / vert_area[:, np.newaxis]
        vert_normals = vert_normals / (np.linalg.norm(vert_normals, axis=1, keepdims=True) + 1e-14)

        # Estimate curvature using angle defect and edge-based methods
        neighbours = temp_particle._vertex_neighbours()

        for i in range(nverts):
            neigh = neighbours[i]
            if len(neigh) < 3:
                continue

            # Local coordinate system
            n = vert_normals[i]
            p = temp_particle.verts[i]

            # Create local tangent vectors
            edge = temp_particle.verts[neigh[0]] - p
            edge = edge - np.dot(edge, n) * n
            edge_norm = np.linalg.norm(edge)
            if edge_norm > 1e-10:
                u = edge / edge_norm
                v = np.cross(n, u)

                # Compute mean curvature using edge-based method
                H = 0
                K = 2 * np.pi  # Start with full angle for Gaussian curvature
                total_area = 0

                for j in range(len(neigh)):
                    v_j = temp_particle.verts[neigh[j]]
                    edge_j = v_j - p
                    len_j = np.linalg.norm(edge_j)

                    if len_j > 1e-10:
                        edge_j = edge_j / len_j
                        # Angle with normal (for mean curvature)
                        angle = masin(np.clip(np.dot(edge_j, n), -1, 1))
                        H += angle * len_j

                        # Angle defect for Gaussian curvature
                        if j < len(neigh) - 1:
                            v_next = temp_particle.verts[neigh[j + 1]]
                            e1 = v_j - p
                            e2 = v_next - p
                            len_e1 = np.linalg.norm(e1)
                            len_e2 = np.linalg.norm(e2)
                            if len_e1 > 1e-10 and len_e2 > 1e-10:
                                cos_angle = np.dot(e1, e2) / (len_e1 * len_e2)
                                angle_at_vertex = macos(np.clip(cos_angle, -1, 1))
                                K -= angle_at_vertex
                                total_area += 0.5 * len_e1 * len_e2 * msin(angle_at_vertex)

                # Normalize
                if total_area > 1e-10:
                    mean_curv[i] = H / total_area
                    gauss_curv[i] = K / total_area

                # Principal curvatures from mean and Gaussian
                H_val = mean_curv[i]
                K_val = gauss_curv[i]
                discriminant = H_val**2 - K_val
                if discriminant >= 0:
                    lambda1[i] = H_val + msqrt(discriminant)
                    lambda2[i] = H_val - msqrt(discriminant)
                else:
                    lambda1[i] = H_val
                    lambda2[i] = H_val

                # Principal directions (simplified - use local coords)
                dir1[i] = u
                dir2[i] = v

        return {
            'mean': mean_curv,
            'gauss': gauss_curv,
            'dir1': dir1,
            'dir2': dir2,
            'lambda1': lambda1,
            'lambda2': lambda2
        }

    # ==================== Mesh cleaning ====================

    def clean(self, cutoff=1e-10):
        """
        Remove multiple vertices and elements with too small areas.

        MATLAB: obj = clean(obj, cutoff)

        Parameters
        ----------
        cutoff : float
            Keep only elements with area > cutoff * mean(area)

        Returns
        -------
        self : Particle
            Cleaned particle
        """
        # Round vertices to avoid floating point issues
        verts_rounded = mround(self.verts, 8)

        # Find unique vertices
        unique_verts, inv_idx = np.unique(verts_rounded, axis=0, return_inverse=True)

        if len(unique_verts) != len(self.verts):
            # Remap faces
            ind3, ind4 = self.index34()
            faces = self.faces.copy()

            if len(ind3) > 0:
                for j in range(3):
                    faces[ind3, j] = inv_idx[faces[ind3, j].astype(int)]
            if len(ind4) > 0:
                for j in range(4):
                    faces[ind4, j] = inv_idx[faces[ind4, j].astype(int)]

            self.verts = unique_verts
            self.faces = faces

        # Remove quads with duplicate vertices (MATLAB: unique + sort(order))
        ind4 = np.where(~np.isnan(self.faces[:, 3]))[0]
        for i in ind4:
            face = self.faces[i]
            valid = face[~np.isnan(face)].astype(int)
            _, order = np.unique(valid, return_index=True)
            if len(order) == 3:
                # Degenerate quad -> triangle, preserve original winding
                sorted_order = np.sort(order)
                self.faces[i] = np.array([valid[sorted_order[0]],
                                          valid[sorted_order[1]],
                                          valid[sorted_order[2]], np.nan])

        self._norm()

        # Keep only elements with sufficient area
        mean_area = np.mean(self.area)
        valid_idx = np.where(self.area > cutoff * mean_area)[0]

        if len(valid_idx) < self.nfaces:
            result, _ = self.select(index=valid_idx)
            self.verts = result.verts
            self.faces = result.faces
            self.verts2 = result.verts2
            self.faces2 = result.faces2
            self._norm()

        return self

    # ==================== Derivatives ====================

    def deriv(self, v):
        """
        Tangential derivative of function defined on surface.

        MATLAB: [v1, v2, t1, t2] = deriv(obj, v)

        Parameters
        ----------
        v : ndarray, shape (nverts,) or (nverts, n)
            Function values given at vertices

        Returns
        -------
        v1, v2 : ndarray
            Derivatives along tvec at boundary centroids
        t1, t2 : ndarray
            Triangular or quadrilateral direction vectors
        """
        # Array size and reshape
        v = np.atleast_2d(v)
        if v.shape[0] == 1:
            v = v.T
        original_shape = v.shape
        v_reshaped = v.reshape(v.shape[0], -1)

        # Index to triangles and quadrilaterals
        ind3, ind4 = self.index34()

        n = self.nfaces
        ncols = v_reshaped.shape[1]

        # Initialize outputs
        v1 = np.zeros((n, ncols))
        v2 = np.zeros((n, ncols))
        t1 = np.zeros((n, 3))
        t2 = np.zeros((n, 3))

        # Function derivative - triangles
        if len(ind3) > 0:
            # Linear triangle shape function derivatives at centroid (1/3, 1/3)
            # dN/dy = [0, 1, -1]
            # dN/dx = [1, 0, -1]
            faces3 = self.faces[ind3, :3].astype(int)
            v1[ind3] = (0 * v_reshaped[faces3[:, 0]] +
                        1 * v_reshaped[faces3[:, 1]] +
                        (-1) * v_reshaped[faces3[:, 2]])
            v2[ind3] = (1 * v_reshaped[faces3[:, 0]] +
                        0 * v_reshaped[faces3[:, 1]] +
                        (-1) * v_reshaped[faces3[:, 2]])

        # Function derivative - quadrilaterals
        if len(ind4) > 0:
            # Bilinear quad shape function derivatives at center (0, 0)
            # dN/dy = [-0.25, -0.25, 0.25, 0.25]
            # dN/dx = [-0.25, 0.25, 0.25, -0.25]
            faces4 = self.faces[ind4, :4].astype(int)
            v1[ind4] = (-0.25 * v_reshaped[faces4[:, 0]] +
                        -0.25 * v_reshaped[faces4[:, 1]] +
                        0.25 * v_reshaped[faces4[:, 2]] +
                        0.25 * v_reshaped[faces4[:, 3]])
            v2[ind4] = (-0.25 * v_reshaped[faces4[:, 0]] +
                        0.25 * v_reshaped[faces4[:, 1]] +
                        0.25 * v_reshaped[faces4[:, 2]] +
                        -0.25 * v_reshaped[faces4[:, 3]])

        # Tangential vectors - derivatives of position
        if self.interp == 'flat':
            # Flat boundary elements
            if len(ind3) > 0:
                faces3 = self.faces[ind3, :3].astype(int)
                t1[ind3] = (0 * self.verts[faces3[:, 0]] +
                           1 * self.verts[faces3[:, 1]] +
                           (-1) * self.verts[faces3[:, 2]])
                t2[ind3] = (1 * self.verts[faces3[:, 0]] +
                           0 * self.verts[faces3[:, 1]] +
                           (-1) * self.verts[faces3[:, 2]])

            if len(ind4) > 0:
                faces4 = self.faces[ind4, :4].astype(int)
                t1[ind4] = (-0.25 * self.verts[faces4[:, 0]] +
                           -0.25 * self.verts[faces4[:, 1]] +
                           0.25 * self.verts[faces4[:, 2]] +
                           0.25 * self.verts[faces4[:, 3]])
                t2[ind4] = (-0.25 * self.verts[faces4[:, 0]] +
                           0.25 * self.verts[faces4[:, 1]] +
                           0.25 * self.verts[faces4[:, 2]] +
                           -0.25 * self.verts[faces4[:, 3]])

        else:  # curved
            # Curved boundary elements
            if len(ind3) > 0:
                # 6-node triangle shape function derivatives at centroid
                faces_idx = self.faces2[ind3][:, [0, 1, 2, 4, 5, 6]].astype(int)
                # Use 6-node shape derivatives
                dx, dy = self._tri6_deriv(np.array([1/3]), np.array([1/3]))
                for i, idx in enumerate(ind3):
                    face = faces_idx[i]
                    t1[idx] = dy[0] @ self.verts2[face]
                    t2[idx] = dx[0] @ self.verts2[face]

            if len(ind4) > 0:
                # 9-node quad shape function derivatives at center
                faces_idx = self.faces2[ind4, :9].astype(int)
                dx, dy = self._quad9_deriv(np.array([0]), np.array([0]))
                for i, idx in enumerate(ind4):
                    face = faces_idx[i]
                    t1[idx] = dy[0] @ self.verts2[face]
                    t2[idx] = dx[0] @ self.verts2[face]

        # Reshape output arrays
        if original_shape[1] == 1 or len(original_shape) == 1:
            v1 = v1.flatten()
            v2 = v2.flatten()
        else:
            v1 = v1.reshape(n, *original_shape[1:])
            v2 = v2.reshape(n, *original_shape[1:])

        return v1, v2, t1, t2

    # ==================== Interpolation ====================

    def interp_values(self, v, method='area'):
        """
        Interpolate values from faces to vertices or vice versa.

        MATLAB: [vi, mat] = interp(obj, v, key)

        Parameters
        ----------
        v : ndarray
            Values at faces (nfaces,) or vertices (nverts,)
        method : str
            'area' for area-weighted, 'pinv' for pseudo-inverse

        Returns
        -------
        vi : ndarray
            Interpolated values
        mat : sparse matrix
            Interpolation matrix
        """
        ind3, ind4 = self.index34()
        nfaces, nverts = self.nfaces, self.nverts

        # Build connectivity
        faces3 = self.faces[ind3, :3].astype(int) if len(ind3) > 0 else np.array([]).reshape(0, 3).astype(int)
        faces4 = self.faces[ind4, :4].astype(int) if len(ind4) > 0 else np.array([]).reshape(0, 4).astype(int)

        n = len(v)

        if n == nfaces:
            # Interpolate from faces to vertices
            if method == 'area':
                # Area-weighted average
                data, rows, cols = [], [], []

                if len(ind3) > 0:
                    for j in range(3):
                        rows.extend(faces3[:, j].tolist())
                        cols.extend(ind3.tolist())
                        data.extend(self.area[ind3].tolist())

                if len(ind4) > 0:
                    for j in range(4):
                        rows.extend(faces4[:, j].tolist())
                        cols.extend(ind4.tolist())
                        data.extend(self.area[ind4].tolist())

                mat = csr_matrix((data, (rows, cols)), shape=(nverts, nfaces))
                # Normalize
                row_sums = np.array(mat.sum(axis=1)).flatten()
                row_sums[row_sums == 0] = 1
                mat = diags(1.0 / row_sums) @ mat
            else:
                # Pseudo-inverse method
                data, rows, cols = [], [], []

                if len(ind3) > 0:
                    for j in range(3):
                        rows.extend(ind3.tolist())
                        cols.extend(faces3[:, j].tolist())
                        data.extend([1/3] * len(ind3))

                if len(ind4) > 0:
                    for j in range(4):
                        rows.extend(ind4.tolist())
                        cols.extend(faces4[:, j].tolist())
                        data.extend([1/4] * len(ind4))

                con = csr_matrix((data, (rows, cols)), shape=(nfaces, nverts))
                mat = np.linalg.pinv(con.toarray())
        else:
            # Interpolate from vertices to faces
            data, rows, cols = [], [], []

            if len(ind3) > 0:
                for j in range(3):
                    rows.extend(ind3.tolist())
                    cols.extend(faces3[:, j].tolist())
                    data.extend([1/3] * len(ind3))

            if len(ind4) > 0:
                for j in range(4):
                    rows.extend(ind4.tolist())
                    cols.extend(faces4[:, j].tolist())
                    data.extend([1/4] * len(ind4))

            mat = csr_matrix((data, (rows, cols)), shape=(nfaces, nverts))

        vi = mat @ v
        return vi, mat

    # ==================== Visualization ====================

    def plot2(self, val=None, **kwargs):
        """
        Advanced plot of discretized particle surface with vectors and cones.

        MATLAB: plot2(obj, val, 'PropertyName', PropertyValue, ...)

        Parameters
        ----------
        val : ndarray, optional
            Values to display on surface (nfaces x 3 or nverts x 3 RGB colors)
        **kwargs : dict
            Plotting options:
            - EdgeColor: Color of edges ('none', 'k', or RGB)
            - FaceAlpha: Transparency (0-1)
            - FaceColor: Face color override
            - nvec: Plot normal vectors (True/False)
            - vec: nfaces x 3 vector array to plot
            - cone: nfaces x 3 vector array for cone plot
            - color: Color for vectors
            - scale: Scale factor for vectors
        """
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        except ImportError:
            print("matplotlib not available for plotting")
            return None

        # Check if new figure is needed
        if plt.get_fignums():
            fig = plt.gcf()
            if fig.axes:
                ax = fig.axes[0]
                new_plot = False
            else:
                ax = fig.add_subplot(111, projection='3d')
                new_plot = True
        else:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            new_plot = True

        # Default values
        face_alpha = kwargs.get('FaceAlpha', 1.0)
        edge_color = kwargs.get('EdgeColor', 'none')

        # Handle value input
        if val is None:
            val = np.array([1.0, 0.7, 0.0])

        if 'FaceColor' in kwargs:
            val = kwargs['FaceColor']

        # Interpolate face values to vertices if needed
        if val.ndim == 1:
            val = np.tile(val, (self.nverts, 1))
        elif val.shape[0] == self.nfaces:
            # Interpolate from faces to vertices
            val, _ = self.interp_values(val)
        elif val.shape[0] != self.nverts:
            val = np.tile(val, (self.nverts, 1))

        # Get face vertices
        face_verts = []
        for i in range(self.nfaces):
            face = self.faces[i]
            idx = face[~np.isnan(face)].astype(int)
            face_verts.append(self.verts[idx])

        # Create 3D collection
        if val.ndim > 1 and val.shape[1] == 3:
            # RGB colors
            collection = Poly3DCollection(face_verts,
                                          facecolors=val,
                                          edgecolor='none',
                                          alpha=face_alpha)
        else:
            # Single color
            from matplotlib.colors import Normalize
            from matplotlib.cm import ScalarMappable, viridis

            if val.ndim == 1:
                norm = Normalize(vmin=np.min(val), vmax=np.max(val))
                mapper = ScalarMappable(norm=norm, cmap=viridis)
                facecolors = [mapper.to_rgba(val[i]) for i in range(self.nverts)]
            else:
                facecolors = [[0.8, 0.8, 0.9]] * self.nverts

            collection = Poly3DCollection(face_verts,
                                          facecolors=facecolors,
                                          edgecolor='none',
                                          alpha=face_alpha)

        ax.add_collection3d(collection)

        # Plot edges if requested
        if edge_color != 'none':
            net, _ = self.edges()
            for edge in net:
                v1, v2 = edge.astype(int)
                points = np.array([self.verts[v1], self.verts[v2]])
                if isinstance(edge_color, str):
                    ax.plot3D(points[:, 0], points[:, 1], points[:, 2],
                              color=edge_color, linewidth=0.5)
                else:
                    ax.plot3D(points[:, 0], points[:, 1], points[:, 2],
                              color=edge_color, linewidth=0.5)

        # Plot normal vectors if requested
        if kwargs.get('nvec', False):
            scale = kwargs.get('scale', 0.1)
            color = kwargs.get('color', 'r')
            for i in range(self.nfaces):
                start = self.pos[i]
                direction = self.nvec[i] * scale
                ax.quiver(start[0], start[1], start[2],
                         direction[0], direction[1], direction[2],
                         color=color, arrow_length_ratio=0.3)

        # Plot custom vectors if requested
        if 'vec' in kwargs:
            vec = kwargs['vec']
            scale = kwargs.get('scale', 0.1)
            color = kwargs.get('color', 'b')
            for i in range(min(self.nfaces, vec.shape[0])):
                start = self.pos[i]
                direction = vec[i] * scale
                ax.quiver(start[0], start[1], start[2],
                         direction[0], direction[1], direction[2],
                         color=color, arrow_length_ratio=0.3)

        # Plot cones if requested
        if 'cone' in kwargs:
            cone = kwargs['cone']
            scale = kwargs.get('scale', 0.1)
            color = kwargs.get('color', 'g')
            # Simplified cone plot as quiver
            for i in range(min(self.nfaces, cone.shape[0])):
                start = self.pos[i]
                direction = cone[i] * scale
                length = np.linalg.norm(direction)
                if length > 1e-10:
                    ax.quiver(start[0], start[1], start[2],
                             direction[0], direction[1], direction[2],
                             color=color, arrow_length_ratio=0.5,
                             linewidth=2)

        # Set axis properties
        if new_plot:
            all_verts = self.verts
            max_range = np.max(all_verts.max(axis=0) - all_verts.min(axis=0)) / 2
            mid = (all_verts.max(axis=0) + all_verts.min(axis=0)) / 2

            ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
            ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
            ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.view_init(elev=40, azim=1)

        plt.draw()
        return fig, ax

    def plot(self, val=None, **kwargs):
        """
        Plot particle surface.

        MATLAB: plot(obj, val, 'PropertyName', PropertyValue)

        Parameters
        ----------
        val : ndarray, optional
            Values to display on surface
        **kwargs : dict
            Plotting options (EdgeColor, FaceAlpha, etc.)
        """
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        except ImportError:
            print("matplotlib not available for plotting")
            return

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Get face vertices
        face_verts = []
        for i in range(self.nfaces):
            face = self.faces[i]
            idx = face[~np.isnan(face)].astype(int)
            face_verts.append(self.verts[idx])

        # Create 3D collection
        if val is not None:
            # Color by value
            from matplotlib.colors import Normalize
            from matplotlib.cm import ScalarMappable, viridis

            norm = Normalize(vmin=np.min(val), vmax=np.max(val))
            mapper = ScalarMappable(norm=norm, cmap=viridis)

            facecolors = [mapper.to_rgba(val[i]) for i in range(self.nfaces)]
            collection = Poly3DCollection(face_verts, facecolors=facecolors,
                                          edgecolor=kwargs.get('EdgeColor', 'k'),
                                          alpha=kwargs.get('FaceAlpha', 1.0))
        else:
            collection = Poly3DCollection(face_verts,
                                          facecolor=kwargs.get('FaceColor', [0.8, 0.8, 0.9]),
                                          edgecolor=kwargs.get('EdgeColor', 'k'),
                                          alpha=kwargs.get('FaceAlpha', 1.0))

        ax.add_collection3d(collection)

        # Set axis limits
        all_verts = np.vstack([self.verts[f[~np.isnan(f)].astype(int)]
                               for f in self.faces])
        max_range = np.max(all_verts.max(axis=0) - all_verts.min(axis=0)) / 2
        mid = (all_verts.max(axis=0) + all_verts.min(axis=0)) / 2

        ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
        ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
        ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        plt.show()
        return fig, ax

    def bradius(self):
        """
        Minimal radius for spheres enclosing boundary elements.

        MATLAB: Misc/+misc/bradius.m

        Returns
        -------
        r : ndarray
            Minimal radius for spheres enclosing each boundary element
        """
        r = np.zeros(self.nfaces)

        # Triangular and quadrilateral faces
        ind3, ind4 = self.index34(np.arange(self.nfaces))

        # Distance between two points
        def dist(x, y):
            return np.linalg.norm(x - y, axis=1)

        # Maximal distance between centroids and triangle edges
        if len(ind3) > 0:
            for i in range(3):
                vert_coords = self.verts[self.faces[ind3, i].astype(int)]
                r[ind3] = np.maximum(r[ind3], dist(self.pos[ind3], vert_coords))

        # Maximal distance between centroids and quadface edges
        if len(ind4) > 0:
            for i in range(4):
                vert_coords = self.verts[self.faces[ind4, i].astype(int)]
                r[ind4] = np.maximum(r[ind4], dist(self.pos[ind4], vert_coords))

        return r

    # ==================== String representations ====================

    def __repr__(self):
        return "Particle(nverts={}, nfaces={}, interp='{}')".format(
            self.nverts, self.nfaces, self.interp)

    def __str__(self):
        return (
            "Particle:\n"
            "  Vertices: {}\n"
            "  Faces: {}\n"
            "  Total area: {:.2f}\n"
            "  Interpolation: {}".format(
                self.nverts, self.nfaces, self.area.sum(), self.interp)
        )
