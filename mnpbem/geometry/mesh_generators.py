"""
Mesh generation functions for various particle shapes.
"""

import numpy as np
import os
from scipy.spatial import Delaunay, ConvexHull
# scipy.io.loadmat no longer needed — sphere data stored as .bin
from .particle import Particle
from ..utils.matlab_compat import mlinspace, mcos, msin, matan2, macos, msqrt


def trisphere(n, diameter=1.0, **kwargs):
    """
    Generate a triangulated sphere with curved boundaries.

    Loads pre-computed sphere vertices from MATLAB trisphere.mat file and
    creates curved triangular elements by adding midpoints on sphere surface.

    MATLAB: Particles/particleshapes/trisphere.m

    Parameters
    ----------
    n : int
        Number of vertices. Will use closest available value from:
        [32, 60, 144, 169, 225, 256, 289, 324, 361, 400, 441, 484, 529, 576,
         625, 676, 729, 784, 841, 900, 961, 1024, 1225, 1444]
    diameter : float, optional
        Diameter of sphere in nm. Default: 1.0
    **kwargs : dict
        Additional arguments passed to Particle constructor

    Returns
    -------
    particle : Particle
        Triangulated sphere with curved boundaries (verts2, faces2)

    Examples
    --------
    >>> # Create 80nm sphere with ~144 vertices
    >>> sphere = trisphere(144, 80.0)
    >>> print("Vertices: {}, Faces: {}".format(sphere.nverts, sphere.nfaces))
    """
    # Validate inputs
    if n is None:
        raise ValueError("trisphere: 'n' must be a positive integer, got None.")
    try:
        n_int = int(n)
    except (TypeError, ValueError):
        raise ValueError("trisphere: 'n' must be a positive integer, got {!r}.".format(n))
    if n_int <= 0:
        raise ValueError("trisphere: 'n' must be > 0, got {}.".format(n_int))
    if not np.isfinite(diameter) or diameter <= 0:
        raise ValueError("trisphere: 'diameter' must be a positive finite float, got {!r}.".format(diameter))

    # Saved vertex counts in MATLAB trisphere.mat
    # MATLAB: trisphere.m line 20-21
    nsav = np.array([32, 60, 144, 169, 225, 256, 289, 324, 361, 400, 441, 484,
                     529, 576, 625, 676, 729, 784, 841, 900, 961, 1024, 1225, 1444])

    # Find closest available number
    # MATLAB: trisphere.m line 24
    ind = np.argmin(np.abs(nsav - n))
    n_actual = nsav[ind]

    if n != n_actual:
        print('trisphere: using {} vertices (closest to requested {})'.format(n_actual, n))

    # Load sphere data from pre-computed binary files
    # Original data: energy-minimized point distributions on sphere
    # (http://www.maths.unsw.edu.au/school/articles/me100.html)
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    faces = None

    # Try pre-triangulated sphere (sphere144 has pre-computed faces)
    tri_file = os.path.join(data_dir, 'sphere{}_tri.bin'.format(n_actual))
    if os.path.exists(tri_file):
        with open(tri_file, 'rb') as f:
            nv, nf = np.fromfile(f, dtype = np.int32, count = 2)
            x = np.fromfile(f, dtype = np.float64, count = nv)
            y = np.fromfile(f, dtype = np.float64, count = nv)
            z = np.fromfile(f, dtype = np.float64, count = nv)
            faces_raw = np.fromfile(f, dtype = np.int32, count = nf * 3).reshape(nf, 3)
        verts = np.column_stack([x, y, z])
        faces = faces_raw - 1  # 1-indexed → 0-indexed

    # Load sphere vertices and triangulate
    if faces is None:
        bin_file = os.path.join(data_dir, 'sphere{}.bin'.format(n_actual))
        if os.path.exists(bin_file):
            with open(bin_file, 'rb') as f:
                nv = np.fromfile(f, dtype = np.int32, count = 1)[0]
                x = np.fromfile(f, dtype = np.float64, count = nv)
                y = np.fromfile(f, dtype = np.float64, count = nv)
                z = np.fromfile(f, dtype = np.float64, count = nv)
            verts = np.column_stack([x, y, z])
            faces = _sphere_triangulate(verts)
        else:
            print('Warning: sphere{}.bin not found, using Fibonacci sphere'.format(n_actual))
            return _trisphere_fibonacci(n, diameter, **kwargs)

    # Rescale to diameter
    # MATLAB: trisphere.m line 49
    verts = 0.5 * verts * diameter

    # Create particle without computing normals
    # MATLAB: trisphere.m line 52
    p = Particle(verts, faces, norm=False)

    # Add midpoints for curved particle boundary
    # MATLAB: trisphere.m line 56
    p = _add_midpoints_flat(p)

    # Project midpoints onto sphere surface
    # MATLAB: trisphere.m line 58-59
    norms = np.linalg.norm(p.verts2, axis=1, keepdims=True)
    verts2 = 0.5 * diameter * (p.verts2 / norms)

    # Create final particle with verts2/faces2 and curved interpolation.
    # MATLAB: trisphere.m line 61 — p = particle(verts2, p.faces2, varargin{:})
    # MATLAB uses curved interpolation by default when used with
    # bemoptions('interp', 'curv'), which is the standard setup for BEM
    # simulations. We default to 'curv' here to match MATLAB behavior and
    # ensure correct area computation for BEM accuracy.
    if 'interp' not in kwargs:
        kwargs = dict(kwargs)
        kwargs['interp'] = 'curv'
    p = Particle(verts2, p.faces2, **kwargs)

    return p


def _add_midpoints_flat(p):
    """
    Add midpoints for curved particle boundaries (flat interpolation).

    MATLAB: Particles/@particle/midpoints.m (case 'flat')

    Handles both triangular and quadrilateral faces.

    Parameters
    ----------
    p : Particle
        Particle with verts and faces

    Returns
    -------
    p : Particle
        Particle with verts2 and faces2 added
    """
    # Determine which faces are triangles and which are quads
    ind3, ind4 = _index34(p.faces)

    # Get edges of particle (handles both tri and quad)
    edges, edge_indices = _get_edges(p.verts, p.faces, ind3, ind4)

    # Number of vertices
    # MATLAB: midpoints.m line 22
    n = len(p.verts)

    # Add edge midpoints to vertex list
    # MATLAB: midpoints.m line 24-25
    edge_midpoints = 0.5 * (p.verts[edges[:, 0]] + p.verts[edges[:, 1]])
    p.verts2 = np.vstack([p.verts, edge_midpoints])

    # Extend face list with 5 extra columns
    # MATLAB: midpoints.m line 29
    nfaces = len(p.faces)
    p.faces2 = np.column_stack([p.faces, np.full((nfaces, 5), np.nan)])

    # For triangular faces, add edge midpoint indices
    # MATLAB: midpoints.m line 32
    # faces2 columns: [v0, v1, v2, nan, e01, e12, e20, nan, nan]
    if len(ind3) > 0:
        p.faces2[ind3, 4] = n + edge_indices[ind3, 0]  # edge 0-1
        p.faces2[ind3, 5] = n + edge_indices[ind3, 1]  # edge 1-2
        p.faces2[ind3, 6] = n + edge_indices[ind3, 2]  # edge 2-0

    # For quadrilateral faces, add edge midpoint indices + centroids
    # MATLAB: midpoints.m line 36-46
    # faces2 columns: [v0, v1, v2, v3, e01, e12, e23, e30, centroid]
    if len(ind4) > 0:
        p.faces2[ind4, 4] = n + edge_indices[ind4, 0]  # edge 0-1
        p.faces2[ind4, 5] = n + edge_indices[ind4, 1]  # edge 1-2
        p.faces2[ind4, 6] = n + edge_indices[ind4, 2]  # edge 2-3
        p.faces2[ind4, 7] = n + edge_indices[ind4, 3]  # edge 3-0

        # Add centroids to face list and vertex list
        f4 = p.faces[ind4].astype(int)
        centroids = 0.25 * (p.verts[f4[:, 0]] + p.verts[f4[:, 1]] +
                            p.verts[f4[:, 2]] + p.verts[f4[:, 3]])
        centroid_start = len(p.verts2)
        p.verts2 = np.vstack([p.verts2, centroids])
        p.faces2[ind4, 8] = centroid_start + np.arange(len(ind4))

    return p


def _index34(faces):
    """
    Get indices of triangular and quadrilateral faces.

    Parameters
    ----------
    faces : ndarray, shape (nfaces, 3 or 4)
        Face array where NaN in column 3 indicates triangle

    Returns
    -------
    ind3 : ndarray
        Indices of triangular faces
    ind4 : ndarray
        Indices of quadrilateral faces
    """
    if faces.shape[1] < 4:
        return np.arange(len(faces)), np.array([], dtype = int)
    is_tri = np.isnan(faces[:, 3])
    return np.where(is_tri)[0], np.where(~is_tri)[0]


def _get_edges(verts, faces, ind3 = None, ind4 = None):
    """
    Get unique edges and their indices for each face.

    MATLAB: Particles/@particle/edges.m

    Handles both triangular (3 edges) and quadrilateral (4 edges) faces.

    Parameters
    ----------
    verts : ndarray
        Vertices
    faces : ndarray
        Face indices (shape nfaces x 3 or nfaces x 4)
    ind3 : ndarray, optional
        Indices of triangular faces
    ind4 : ndarray, optional
        Indices of quadrilateral faces

    Returns
    -------
    edges : ndarray, shape (n_edges, 2)
        Unique edges as pairs of vertex indices
    edge_indices : ndarray, shape (n_faces, max_edges)
        Index of each edge in the edges array for each face.
        For triangles: 3 valid columns. For quads: 4 valid columns.
        Shape is (nfaces, 4) with -1 for unused entries.
    """
    nfaces = len(faces)

    if ind3 is None or ind4 is None:
        ind3, ind4 = _index34(faces)

    # Collect all edges
    all_edges_list = []
    # Map: face_index -> list of edge positions in all_edges_list
    face_edge_positions = {}  # face_idx -> list of global edge positions

    pos = 0  # running position counter

    # Triangular faces: edges (0,1), (1,2), (2,0)
    if len(ind3) > 0:
        f3 = faces[ind3, :3].astype(int)
        tri_edges = np.empty((len(ind3) * 3, 2), dtype = int)
        tri_edges[0::3] = f3[:, [0, 1]]
        tri_edges[1::3] = f3[:, [1, 2]]
        tri_edges[2::3] = f3[:, [2, 0]]
        for i, idx in enumerate(ind3):
            face_edge_positions[idx] = [pos + i * 3, pos + i * 3 + 1, pos + i * 3 + 2]
        all_edges_list.append(tri_edges)
        pos += len(tri_edges)

    # Quadrilateral faces: edges (0,1), (1,2), (2,3), (3,0)
    if len(ind4) > 0:
        f4 = faces[ind4, :4].astype(int)
        quad_edges = np.empty((len(ind4) * 4, 2), dtype = int)
        quad_edges[0::4] = f4[:, [0, 1]]
        quad_edges[1::4] = f4[:, [1, 2]]
        quad_edges[2::4] = f4[:, [2, 3]]
        quad_edges[3::4] = f4[:, [3, 0]]
        for i, idx in enumerate(ind4):
            face_edge_positions[idx] = [pos + i * 4, pos + i * 4 + 1,
                                        pos + i * 4 + 2, pos + i * 4 + 3]
        all_edges_list.append(quad_edges)
        pos += len(quad_edges)

    if len(all_edges_list) == 0:
        return np.array([], dtype = int).reshape(0, 2), np.full((nfaces, 4), -1, dtype = int)

    all_edges = np.vstack(all_edges_list)

    # Sort edge vertices so (v1, v2) and (v2, v1) are treated as same edge
    all_edges_sorted = np.sort(all_edges, axis = 1)

    # Find unique edges
    unique_edges, inverse_indices = np.unique(
        all_edges_sorted, axis = 0, return_inverse = True
    )

    # Build edge_indices array: (nfaces, max_edges_per_face)
    max_edges = 4  # quads have 4 edges
    edge_indices = np.full((nfaces, max_edges), -1, dtype = int)
    for face_idx, edge_positions in face_edge_positions.items():
        for j, edge_pos in enumerate(edge_positions):
            edge_indices[face_idx, j] = inverse_indices[edge_pos]

    return unique_edges, edge_indices


def _trisphere_fibonacci(n, diameter=1.0, **kwargs):
    """
    Fallback: Generate sphere using Fibonacci algorithm.

    This is used when the MATLAB .mat file is not available.
    """
    # Generate Fibonacci sphere points for quasi-uniform distribution
    verts = _fibonacci_sphere(n)

    # Perform spherical Delaunay triangulation
    faces = _sphere_triangulate(verts)

    # Scale to desired diameter
    verts = verts * (diameter / 2.0)

    # Create particle
    p = Particle(verts, faces, **kwargs)

    # Add midpoints and project to sphere
    p = _add_midpoints_flat(p)
    norms = np.linalg.norm(p.verts2, axis=1, keepdims=True)
    verts2 = (diameter / 2.0) * (p.verts2 / norms)
    p = Particle(verts2, p.faces2, **kwargs)

    # Set curved interpolation mode
    p.interp = 'curv'
    p._norm()  # Recompute normals for curved boundaries

    return p


def _fibonacci_sphere(n):
    """
    Generate approximately n points uniformly distributed on unit sphere
    using Fibonacci spiral.

    Parameters
    ----------
    n : int
        Approximate number of points

    Returns
    -------
    points : ndarray, shape (m, 3)
        Points on unit sphere (m ≈ n)
    """
    indices = np.arange(0, n, dtype=float) + 0.5

    phi = macos(1 - 2 * indices / n)  # Latitude
    theta = np.pi * (1 + 5**0.5) * indices  # Golden angle spiral

    x = msin(phi) * mcos(theta)
    y = msin(phi) * msin(theta)
    z = mcos(phi)

    points = np.column_stack([x, y, z])

    # Normalize to ensure points are exactly on unit sphere
    points = points / np.linalg.norm(points, axis=1, keepdims=True)

    return points


def _sphere_triangulate(verts):
    """
    Triangulate points on sphere using stereographic projection.

    Based on MATLAB MNPBEM sphtriangulate.m

    Parameters
    ----------
    verts : ndarray, shape (n, 3)
        Vertices on sphere surface

    Returns
    -------
    faces : ndarray, shape (m, 3)
        Triangle face indices (0-indexed)
    """
    n = len(verts)

    # Step 1: Use first vertex as projection center
    center = verts[0]

    # Build rotation matrix that rotates first point to [0, 0, -1]
    r3 = -center
    if center[2] != 0:
        r2 = np.array([0, -r3[2], r3[1]])
    else:
        r2 = np.array([-r3[1], r3[0], 0])
    r2 = r2 / np.linalg.norm(r2)
    r1 = np.cross(r3, r2)

    rot = np.array([r1, r2, r3])

    # Rotate all vertices except first
    vertr = verts[1:] @ rot.T

    # Project to z=0 plane from center [0, 0, -1]
    tp = -1.0 / (vertr[:, 2] + 1)
    xp = vertr[:, 0] * tp
    yp = vertr[:, 1] * tp

    # Step 2: Delaunay triangulation of projected points
    points_2d = np.column_stack([xp, yp])
    tri = Delaunay(points_2d)
    faces = tri.simplices

    # Ensure outward-pointing normals
    vertp = np.column_stack([xp, yp, np.zeros(n - 1)])
    u = vertp[faces[:, 0]] - vertp[faces[:, 1]]
    v = vertp[faces[:, 2]] - vertp[faces[:, 1]]
    w = np.cross(u, v)

    # Flip faces with inward normals
    flip_idx = w[:, 2] > 0
    faces[flip_idx, :] = faces[flip_idx, :][:, [0, 2, 1]]

    # Step 3: Connect projection center to convex hull
    hull = ConvexHull(points_2d)
    hull_verts = hull.vertices

    # Create triangles connecting to center (vertex 0)
    n_hull = len(hull_verts)
    hull_faces = np.zeros((n_hull, 3), dtype=int)
    for i in range(n_hull):
        hull_faces[i] = [hull_verts[(i+1) % n_hull] + 1,
                         hull_verts[i] + 1,
                         0]

    # Combine all faces (shift indices by +1 for Delaunay part)
    all_faces = np.vstack([faces + 1, hull_faces])

    return all_faces


def triellipsoid(n, axes):
    """
    Generate triangulated ellipsoid.

    Parameters
    ----------
    n : int
        Number of vertices
    axes : array_like, shape (3,)
        Semi-axes lengths [a, b, c] in nm

    Returns
    -------
    particle : Particle
        Triangulated ellipsoid
    """
    # Start with unit sphere
    verts = _fibonacci_sphere(n)
    faces = _sphere_triangulate(verts)

    # Scale by axes
    verts = verts * np.array(axes)

    return Particle(verts, faces)


# ===========================================================================
# Helper: surf2patch equivalent
# ===========================================================================

def _surf2patch(x, y, z, triangles = False):
    # Python equivalent of MATLAB surf2patch(x, y, z) / surf2patch(x, y, z, 'triangles')
    # x, y, z are 2D arrays of shape (m, n). Returns (faces, verts).
    # Output bit-matches MATLAB surf2patch:
    #   quad      : [v00, v01, v11, v10] per cell, looping (j outer, i inner)
    #   triangles : all "first" triangles [v00, v01, v11] in order, then all
    #               "second" triangles [v00, v11, v10] (MATLAB groups, not interleaves)
    m, n = x.shape
    verts = np.column_stack([x.ravel(order = 'F'), y.ravel(order = 'F'), z.ravel(order = 'F')])

    if triangles:
        tri1, tri2 = [], []
        for j in range(n - 1):
            for i in range(m - 1):
                v00 = j * m + i
                v10 = j * m + (i + 1)
                v01 = (j + 1) * m + i
                v11 = (j + 1) * m + (i + 1)
                tri1.append([v00, v01, v11])
                tri2.append([v00, v11, v10])
        faces = np.array(tri1 + tri2, dtype = int)
    else:
        faces_list = []
        for j in range(n - 1):
            for i in range(m - 1):
                v00 = j * m + i
                v10 = j * m + (i + 1)
                v01 = (j + 1) * m + i
                v11 = (j + 1) * m + (i + 1)
                faces_list.append([v00, v01, v11, v10])
        faces = np.array(faces_list, dtype = int)

    return faces, verts


# ===========================================================================
# fvgrid: convert parametric surface to face-vertex structure
# ===========================================================================

def fvgrid(x: np.ndarray,
        y: np.ndarray,
        triangles: bool = False) -> tuple:
    # MATLAB: Particles/particleshapes/misc/fvgrid.m
    # Convert 2D grid to face-vertex structure
    x = np.asarray(x, dtype = float)
    y = np.asarray(y, dtype = float)

    # If 1D, meshgrid them
    if x.ndim == 1 and y.ndim == 1:
        x, y = np.meshgrid(x, y)

    z = np.zeros_like(x)

    # Use surf2patch equivalent
    faces, verts = _surf2patch(x, y, z, triangles = triangles)

    # MATLAB fvgrid applies fliplr() on surf2patch output. Mirror that here so
    # the resulting (verts, faces) is bit-identical to MATLAB's fvgrid output.
    faces = faces[:, ::-1]

    # Create particle with norm='off'
    p = Particle(verts, faces, norm = 'off')

    # Add midpoints (flat)
    p = _add_midpoints_flat(p)

    return p.verts2, p.faces2


# ===========================================================================
# trispheresegment: discretized surface of sphere segment
# ===========================================================================

def trispheresegment(phi: np.ndarray,
        theta: np.ndarray,
        diameter: float = 1.0,
        triangles: bool = False,
        **kwargs) -> 'Particle':
    # MATLAB: Particles/particleshapes/trispheresegment.m
    phi = np.asarray(phi, dtype = float)
    theta = np.asarray(theta, dtype = float)

    # Meshgrid phi and theta
    phi_grid, theta_grid = np.meshgrid(phi, theta)

    # Spherical to cartesian
    x = diameter / 2.0 * msin(theta_grid) * mcos(phi_grid)
    y = diameter / 2.0 * msin(theta_grid) * msin(phi_grid)
    z = diameter / 2.0 * mcos(theta_grid)

    # Use surf2patch to create faces
    # _surf2patch already matches MATLAB surf2patch winding; no permutation needed
    faces, verts = _surf2patch(x, y, z, triangles = triangles)

    p = Particle(verts, faces)
    p = p.clean()

    # Add midpoints for curved particle boundary
    p = _add_midpoints_flat(p)

    # Rescale vertices to sphere surface
    norms = msqrt(np.sum(p.verts2 ** 2, axis = 1, keepdims = True))
    # Avoid division by zero for points at origin
    norms = np.maximum(norms, 1e-30)
    verts2 = 0.5 * diameter * (p.verts2 / norms)

    # Create particle with midpoints
    p = Particle(verts2, p.faces2, **kwargs)

    return p


# ===========================================================================
# trirod: cylinder with hemispherical caps
# ===========================================================================

def trirod(diameter: float,
        height: float,
        n: list = None,
        triangles: bool = False,
        **kwargs) -> 'Particle':
    # MATLAB: Particles/particleshapes/trirod.m
    if n is None:
        n = [15, 20, 20]
    assert len(n) == 3, '[error] n must have 3 elements [nphi, ntheta, nz]'

    nphi, ntheta, nz_cyl = n

    # Angles
    phi = mlinspace(0, 2 * np.pi, nphi)
    theta = mlinspace(0, 0.5 * np.pi, ntheta)

    # Upper cap: sphere segment shifted up
    # MATLAB trispheresegment uses quadrilateral faces by default
    cap1 = trispheresegment(phi, theta, diameter, triangles = triangles, **kwargs)
    cap1.shift([0, 0, 0.5 * (height - diameter)])

    # Lower cap: flip cap1 along z-axis
    cap2 = cap1.flip(2)

    # z-values for cylinder
    z_vals = 0.5 * mlinspace(-1, 1, nz_cyl) * (height - diameter)

    # Grid for cylinder
    verts_grid, faces_grid = fvgrid(phi, z_vals, triangles = triangles)
    # Extract phi and z from grid vertices
    phi_cyl = verts_grid[:, 0]
    z_cyl = verts_grid[:, 1]

    # Cylinder coordinates
    x_cyl = 0.5 * diameter * mcos(phi_cyl)
    y_cyl = 0.5 * diameter * msin(phi_cyl)

    # Create cylinder particle
    cyl_verts = np.column_stack([x_cyl, y_cyl, z_cyl])
    cyl = Particle(cyl_verts, faces_grid, **kwargs)

    # Compose particle: cap1 + cap2 + cylinder, then clean
    p = (cap1 + cap2 + cyl).clean()

    return p


# ===========================================================================
# tricube: cube with rounded edges
# ===========================================================================

def _square_grid(n: int, e: float) -> tuple:
    # MATLAB: square() subfunction in tricube.m
    u = mlinspace(-0.5 ** e, 0.5 ** e, n)

    verts, faces = fvgrid(u, u)

    x = np.sign(verts[:, 0]) * np.abs(verts[:, 0]) ** (1.0 / e)
    y = np.sign(verts[:, 1]) * np.abs(verts[:, 1]) ** (1.0 / e)

    return x, y, faces


def tricube(n: int,
        length: float = 1.0,
        e: float = 0.25,
        **kwargs) -> 'Particle':
    # MATLAB: Particles/particleshapes/tricube.m
    # Make length an array of 3
    if np.isscalar(length):
        length = np.array([length, length, length], dtype = float)
    else:
        length = np.asarray(length, dtype = float)
        if length.size != 3:
            length = np.full(3, length.flat[0])

    # Discretize single side of cube
    x, y, faces = _square_grid(n, e)
    z = 0.5 * np.ones_like(x)

    # Put together 6 cube sides
    # MATLAB:  [x, y, z], [y, x, -z], [y, z, x], [x, -z, y], [z, x, y], [-z, y, x]
    p1 = Particle(np.column_stack([x, y, z]), faces)
    p2 = Particle(np.column_stack([y, x, -z]), faces)
    p3 = Particle(np.column_stack([y, z, x]), faces)
    p4 = Particle(np.column_stack([x, -z, y]), faces)
    p5 = Particle(np.column_stack([z, x, y]), faces)
    p6 = Particle(np.column_stack([-z, y, x]), faces)

    p = (p1 + p2 + p3 + p4 + p5 + p6).clean()

    # Convert to spherical coordinates for super-sphere rounding
    phi_sph, theta_sph = _cart2sph(p.verts2[:, 0], p.verts2[:, 1], p.verts2[:, 2])

    # Signed power functions
    def isin(x):
        return np.sign(msin(x)) * np.abs(msin(x)) ** e

    def icos(x):
        return np.sign(mcos(x)) * np.abs(mcos(x)) ** e

    # Super-sphere vertices
    x_new = 0.5 * icos(theta_sph) * icos(phi_sph)
    y_new = 0.5 * icos(theta_sph) * isin(phi_sph)
    z_new = 0.5 * isin(theta_sph)

    # Create final particle and scale
    p = Particle(np.column_stack([x_new, y_new, z_new]), p.faces2, **kwargs)
    p.scale(length)

    return p


def _cart2sph(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple:
    # MATLAB cart2sph equivalent
    # Returns (azimuth, elevation) -- note MATLAB convention
    hxy = msqrt(x ** 2 + y ** 2)
    phi = matan2(y, x)  # azimuth
    theta = matan2(z, hxy)  # elevation
    return phi, theta


# ===========================================================================
# tritorus: triangulated torus
# ===========================================================================

def tritorus(diameter: float,
        rad: float,
        n: list = None,
        **kwargs) -> 'Particle':
    # MATLAB: Particles/particleshapes/tritorus.m
    if n is None:
        n = [21, 21]
    if np.isscalar(n):
        n = [n, n]

    # Grid triangulation
    verts_grid, faces_grid = fvgrid(
        mlinspace(0, 2 * np.pi, n[0]),
        mlinspace(0, 2 * np.pi, n[1]))

    # Angles
    phi = verts_grid[:, 0]
    theta = verts_grid[:, 1]

    # Coordinates of torus
    x = (0.5 * diameter + rad * mcos(theta)) * mcos(phi)
    y = (0.5 * diameter + rad * mcos(theta)) * msin(phi)
    z = rad * msin(theta)

    # Make torus
    p = Particle(np.column_stack([x, y, z]), faces_grid, **kwargs).clean()

    return p


# ===========================================================================
# trispherescale: deform surface of sphere
# ===========================================================================

def trispherescale(p: 'Particle',
        scale: np.ndarray,
        unit: bool = False) -> 'Particle':
    # MATLAB: Particles/particleshapes/trispherescale.m
    scale = np.asarray(scale, dtype = float)

    if unit:
        scale = scale / np.max(scale)

    # If scale has same length as nfaces, interpolate to vertices
    if scale.size == p.nfaces:
        scale = p.interp_to_verts(scale)

    # Scale vertex positions
    p.verts = np.reshape(scale, (-1, 1)) * p.verts
    if p.verts2 is not None:
        # For verts2, we need to handle the extra midpoints
        # The first nverts entries match verts, rest are midpoints
        scale2 = np.empty(len(p.verts2), dtype = float)
        scale2[:len(scale)] = scale
        # Interpolate scale for midpoints
        if len(p.verts2) > len(scale):
            edges, _ = _get_edges(p.verts, p.faces)
            for i in range(len(edges)):
                scale2[len(scale) + i] = 0.5 * (scale[edges[i, 0]] + scale[edges[i, 1]])
        p.verts2 = np.reshape(scale2, (-1, 1)) * p.verts2

    p._norm()
    return p


# ===========================================================================
# tripolygon: 3D particle from 2D polygon + edge profile
# ===========================================================================

def tripolygon(poly, edge, **kwargs):
    # MATLAB: Particles/particleshapes/tripolygon.m
    # Creates 3D nanostructure from 2D polygon cross-section + edge profile.
    # If return_poly=True, also returns the enriched edge polygon(s)
    # produced during mesh refinement (MATLAB's second output).
    from .polygon3 import Polygon3
    from .edgeprofile import EdgeProfile

    return_poly = kwargs.pop('return_poly', False)

    # handle single polygon or list of polygons
    if not isinstance(poly, (list, tuple)):
        polys = [poly]
        single_input = True
    else:
        polys = list(poly)
        single_input = False

    # check edge profile type: rounded or sharp edges
    has_nan = np.any(np.isnan(edge.pos[:, 0]))
    nan_count_at_zero = np.sum(edge.pos[:, 0] == 0)
    all_not_nan = not has_nan

    if all_not_nan or (has_nan and nan_count_at_zero != 1):
        # both edges rounded (or both sharp -- mode '11')
        p, enriched = _tripolygon_both_rounded(polys, edge, **kwargs)
    elif np.isnan(edge.pos[0, 0]):
        # sharp lower edge
        p, enriched = _tripolygon_sharp_lower(polys, edge, **kwargs)
    else:
        # sharp upper edge
        p, enriched = _tripolygon_sharp_upper(polys, edge, **kwargs)

    if not return_poly:
        return p

    if single_input:
        return p, enriched[0]
    return p, enriched


def _tripolygon_both_rounded(polys, edge, **kwargs):
    # MATLAB tripolygon.m -- case: both edges rounded
    from .polygon3 import Polygon3

    # Extract sym option (used for mirror symmetry, not passed to plate/ribbon)
    sym = kwargs.pop('sym', None)

    # create polygon3 objects at zmin and zmax
    polys1 = [Polygon3(p, edge.zmin) for p in polys]
    polys2 = [Polygon3(p, edge.zmax) for p in polys]

    # lower plate (dir = -1)
    plates1 = []
    for p3 in polys1:
        plate, _ = p3.plate(dir = -1, edge = edge, sym = sym, **kwargs)
        plates1.append(plate)

    # upper plate (dir = +1)
    polys_out = []
    plates2 = []
    for p3 in polys2:
        plate, p3_out = p3.plate(dir = 1, edge = edge, sym = sym, **kwargs)
        plates2.append(plate)
        polys_out.append(p3_out)

    # vertical ribbon (side walls)
    ribbons = []
    for p3_out in polys_out:
        ribbon, _, _ = p3_out.vribbon(edge = edge, sym = sym)
        ribbons.append(ribbon)

    # combine all particles
    all_parts = plates1 + plates2 + ribbons
    p = all_parts[0]
    for part in all_parts[1:]:
        p = p + part

    p = p.clean()
    # enriched poly = upper plate's output (MATLAB tripolygon.m L25)
    return p, polys_out


def _tripolygon_sharp_lower(polys, edge, **kwargs):
    # MATLAB tripolygon.m -- case: sharp lower edge (NaN at start)
    from .polygon3 import Polygon3

    # Extract sym option (used for mirror symmetry); MATLAB passes sym through
    # plate/vribbon via varargin just like the both-rounded branch.
    sym = kwargs.pop('sym', None)

    # polygon3 objects at zmax
    polys3 = [Polygon3(p, edge.zmax) for p in polys]

    # upper plate
    polys_out = []
    plates1 = []
    for p3 in polys3:
        plate, p3_out = p3.plate(dir = 1, edge = edge, sym = sym, **kwargs)
        plates1.append(plate)
        polys_out.append(p3_out)

    # vertical ribbon
    ribbons = []
    lo_polys = []
    for p3_out in polys_out:
        ribbon, _, lo = p3_out.vribbon(edge = edge, sym = sym)
        ribbons.append(ribbon)
        lo_polys.append(lo)

    # lower plate (at zmin, using the lower boundary polygon)
    plates2 = []
    for lo_p3 in lo_polys:
        lo_p3.z = edge.zmin
        plate, _ = lo_p3.plate(dir = -1, edge = edge, sym = sym, **kwargs)
        plates2.append(plate)

    # combine
    all_parts = plates1 + ribbons + plates2
    p = all_parts[0]
    for part in all_parts[1:]:
        p = p + part

    p = p.clean()
    # enriched poly = lower ribbon boundary (MATLAB tripolygon.m L36)
    return p, lo_polys


def _tripolygon_sharp_upper(polys, edge, **kwargs):
    # MATLAB tripolygon.m -- case: sharp upper edge (NaN at end)
    from .polygon3 import Polygon3

    # Extract sym option (used for mirror symmetry); MATLAB passes sym through
    # plate/vribbon via varargin just like the both-rounded branch.
    sym = kwargs.pop('sym', None)

    # polygon3 objects at zmin
    polys3 = [Polygon3(p, edge.zmin) for p in polys]

    # lower plate
    polys_out = []
    plates1 = []
    for p3 in polys3:
        plate, p3_out = p3.plate(dir = -1, edge = edge, sym = sym, **kwargs)
        plates1.append(plate)
        polys_out.append(p3_out)

    # vertical ribbon
    ribbons = []
    up_polys = []
    for p3_out in polys_out:
        ribbon, up, _ = p3_out.vribbon(edge = edge, sym = sym)
        ribbons.append(ribbon)
        up_polys.append(up)

    # upper plate (at zmax, using the upper boundary polygon)
    plates2 = []
    for up_p3 in up_polys:
        up_p3.z = edge.zmax
        plate, _ = up_p3.plate(dir = 1, edge = edge, sym = sym, **kwargs)
        plates2.append(plate)

    # combine
    all_parts = plates1 + ribbons + plates2
    p = all_parts[0]
    for part in all_parts[1:]:
        p = p + part

    p = p.clean()
    # enriched poly = upper ribbon boundary (MATLAB tripolygon.m L47)
    return p, up_polys


# ===========================================================================
# particle_from_mat: load MATLAB mesh dump as Particle (bit-identical inject)
# ===========================================================================

def particle_from_mat(path, key, interp = 'curv', inject_geom = False):
    """
    Load a MATLAB-dumped particle from a .mat file.

    Expects the .mat file to contain four arrays named
    ``{key}_verts``, ``{key}_faces``, ``{key}_verts2``, ``{key}_faces2``.
    The curved (verts2, faces2) tables are used directly so the resulting
    Particle is bit-identical to the MATLAB mesh — bypassing the
    `polymesh2d`/`plate` floating-point drift documented in
    MESH2D_FP_LIMIT.md.

    Parameters
    ----------
    path : str
        Path to the .mat file.
    key : str
        Variable name prefix inside the .mat file (e.g. ``'up'`` picks
        up ``up_verts``, ``up_faces``, ``up_verts2``, ``up_faces2``).
    interp : str, optional
        Particle interpolation mode. Default ``'curv'`` to match MATLAB
        ``bemoptions('interp','curv')``.
    inject_geom : bool, optional
        If True, also inject ``pos``, ``nvec`` (vec[3]), ``tvec1`` (vec[1]),
        ``tvec2`` (vec[2]), and ``area`` directly from the .mat file,
        bypassing Python's ``_norm_curv`` recomputation. Useful when even
        the ~1e-14 ULP drift from ``_norm_curv`` matters. Requires the .mat
        file to contain ``{key}_pos``, ``{key}_nvec``, ``{key}_tvec1``,
        ``{key}_tvec2``, and ``{key}_area``.

    Returns
    -------
    p : Particle
        Particle reconstructed with MATLAB verts/faces and curved
        midpoint tables.
    """
    from scipy.io import loadmat
    mat = loadmat(path)
    verts  = np.ascontiguousarray(mat[key + '_verts'], dtype = float)
    faces  = np.asarray(mat[key + '_faces'],  dtype = float)
    verts2 = np.ascontiguousarray(mat[key + '_verts2'], dtype = float)
    faces2 = np.asarray(mat[key + '_faces2'], dtype = float)

    # MATLAB uses 1-based indexing; stored NaN entries stay NaN.
    faces  = _matidx_to_py(faces)
    faces2 = _matidx_to_py(faces2)

    # Build particle from (verts2, faces2) so Particle.__init__ takes the
    # 9-column curved path and rebuilds verts/faces via corner extraction.
    # Then overwrite verts/faces with the MATLAB-dumped arrays so the
    # ComParticle indexing matches MATLAB exactly (no re-derivation).
    p = Particle(verts2, faces2, interp = interp, norm = 'off')
    p.verts = verts
    p.faces = faces
    p.verts2 = verts2
    p.faces2 = faces2

    if inject_geom:
        p.pos  = np.ascontiguousarray(mat[key + '_pos'],   dtype = float)
        nvec   = np.ascontiguousarray(mat[key + '_nvec'],  dtype = float)
        tvec1  = np.ascontiguousarray(mat[key + '_tvec1'], dtype = float)
        tvec2  = np.ascontiguousarray(mat[key + '_tvec2'], dtype = float)
        p.vec  = [tvec1, tvec2, nvec]
        p.area = np.ascontiguousarray(mat[key + '_area'],  dtype = float).flatten()
    else:
        # Recompute geometric properties (pos, vec, area) under MATLAB verts.
        p._norm()
    return p


def _matidx_to_py(arr):
    """Convert 1-based MATLAB face indices to 0-based, preserving NaN."""
    out = np.array(arr, dtype = float, copy = True)
    mask = ~np.isnan(out)
    out[mask] = out[mask] - 1.0
    return out
