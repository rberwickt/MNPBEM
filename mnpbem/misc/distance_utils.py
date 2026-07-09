"""
Distance computation utilities for MNPBEM.

MATLAB: Misc/distmin3.m, +misc/bdist2.m, +misc/pdist2.m, +misc/bradius.m
"""

import numpy as np
from typing import Tuple, Optional


def pdist2(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """
    MATLAB: +misc/pdist2.m

    Distance array between position arrays P1 and P2.

    Parameters
    ----------
    p1 : ndarray, shape (n1, 3)
    p2 : ndarray, shape (n2, 3)

    Returns
    -------
    d : ndarray, shape (n1, n2)
    """
    d = (np.sum(p1 ** 2, axis = 1, keepdims = True)
         + np.sum(p2 ** 2, axis = 1, keepdims = True).T
         - 2 * p1 @ p2.T)
    # avoid rounding errors
    d[d < 1e-10] = 0
    d = np.sqrt(d)
    return d


def bradius(p: object) -> np.ndarray:
    """
    MATLAB: +misc/bradius.m

    Minimal radius for spheres enclosing boundary elements.

    Parameters
    ----------
    p : particle object with .n, .pos, .verts, .faces attributes

    Returns
    -------
    r : ndarray, shape (n,)
    """
    r = np.zeros(p.n)
    faces = p.faces

    # triangular faces (4th column is NaN)
    if faces.shape[1] >= 4:
        ind3 = np.where(np.isnan(faces[:, 3]))[0]
        ind4 = np.where(~np.isnan(faces[:, 3]))[0]
    else:
        ind3 = np.arange(faces.shape[0])
        ind4 = np.array([], dtype = int)

    def dist(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.sqrt(np.sum((x - y) ** 2, axis = 1))

    # maximal distance between centroids and triangle vertices
    if len(ind3) > 0:
        for i in range(3):
            r[ind3] = np.maximum(r[ind3],
                dist(p.pos[ind3, :], p.verts[faces[ind3, i].astype(int), :]))

    # maximal distance between centroids and quad vertices
    if len(ind4) > 0:
        for i in range(4):
            r[ind4] = np.maximum(r[ind4],
                dist(p.pos[ind4, :], p.verts[faces[ind4, i].astype(int), :]))

    return r


def _point_edge_dist(pos: np.ndarray,
        verts: np.ndarray,
        faces: np.ndarray) -> np.ndarray:
    """
    MATLAB: bdist2/PointEdgeDist

    Minimal distance between points and boundary edges.
    """
    n_pos = pos.shape[0]
    n_faces = faces.shape[0]
    d = np.full((n_pos, n_faces), np.inf)

    if n_faces == 0:
        return d

    # close faces
    faces_closed_shape = (faces.shape[0], faces.shape[1] + 1)
    faces_closed = np.empty(faces_closed_shape, dtype = faces.dtype)
    faces_closed[:, :faces.shape[1]] = faces
    faces_closed[:, -1] = faces[:, 0]

    for i in range(1, faces_closed.shape[1]):
        v1 = verts[faces_closed[:, i - 1].astype(int), :]
        v2 = verts[faces_closed[:, i].astype(int), :]
        a = v2 - v1

        # parameter for minimal distance
        dot_a_a = np.sum(a * a, axis = 1)  # (n_faces,)
        dot_v1_a = np.sum(v1 * a, axis = 1)  # (n_faces,)

        # lambda = (pos * a' - dot(v1, a, 2)') / dot(a, a, 2)'
        lam = (pos @ a.T - dot_v1_a[np.newaxis, :]) / (dot_a_a[np.newaxis, :] + 1e-30)
        lam = np.clip(lam, 0, 1)

        # distance
        dist_sq = (pdist2(pos, v1) ** 2
                   + lam ** 2 * dot_a_a[np.newaxis, :]
                   - 2 * lam * (pos @ a.T - dot_v1_a[np.newaxis, :]))
        dist_sq = np.maximum(dist_sq, 0)
        d = np.minimum(d, np.sqrt(dist_sq))

    return d


def _point_triangle_dist(pos: np.ndarray,
        verts: np.ndarray,
        faces: np.ndarray) -> np.ndarray:
    """
    MATLAB: bdist2/PointTriangleDist

    Normal distance between points and triangle plane.
    """
    n_pos = pos.shape[0]
    n_faces = faces.shape[0]
    d = np.full((n_pos, n_faces), np.inf)

    if n_faces == 0:
        return d

    v1 = verts[faces[:, 0].astype(int), :]
    v2 = verts[faces[:, 1].astype(int), :]
    v3 = verts[faces[:, 2].astype(int), :]

    a1 = v2 - v1
    a2 = v3 - v1

    in1 = pos @ a1.T - np.sum(v1 * a1, axis = 1)[np.newaxis, :]
    in2 = pos @ a2.T - np.sum(v1 * a2, axis = 1)[np.newaxis, :]

    a11 = np.sum(a1 * a1, axis = 1)
    a12 = np.sum(a1 * a2, axis = 1)
    a22 = np.sum(a2 * a2, axis = 1)

    det = a11 * a22 - a12 ** 2

    mu1 = (in1 * a22[np.newaxis, :] - in2 * a12[np.newaxis, :]) / (det[np.newaxis, :] + 1e-30)
    mu2 = (in2 * a11[np.newaxis, :] - in1 * a12[np.newaxis, :]) / (det[np.newaxis, :] + 1e-30)

    inside = (mu1 >= 0) & (mu1 <= 1) & (mu2 >= 0) & (mu2 <= 1) & (mu1 + mu2 <= 1)

    # normal vector
    nvec = np.cross(a1, a2)
    nvec_norm = np.sqrt(np.sum(nvec ** 2, axis = 1, keepdims = True))
    nvec = nvec / (nvec_norm + 1e-30)

    d = np.abs(pos @ nvec.T - np.sum(v1 * nvec, axis = 1)[np.newaxis, :])
    d[~inside] = np.inf

    return d


def bdist2(p1: object, p2: object) -> np.ndarray:
    """
    MATLAB: +misc/bdist2.m

    Minimal distance between positions P1 and boundary elements P2.

    Parameters
    ----------
    p1 : object with .n, .pos attributes
    p2 : object with .n, .verts, .faces attributes

    Returns
    -------
    d : ndarray, shape (p1.n, p2.n)
    """
    d = np.zeros((p1.n, p2.n))
    faces = p2.faces

    if faces.shape[1] >= 4:
        ind3 = np.where(np.isnan(faces[:, 3]))[0]
        ind4 = np.where(~np.isnan(faces[:, 3]))[0]
    else:
        ind3 = np.arange(faces.shape[0])
        ind4 = np.array([], dtype = int)

    # edge distances
    if len(ind3) > 0:
        d[:, ind3] = _point_edge_dist(p1.pos, p2.verts, faces[ind3, :3])
    if len(ind4) > 0:
        d[:, ind4] = _point_edge_dist(p1.pos, p2.verts, faces[ind4, :4])

    # triangle plane distances
    if len(ind3) > 0:
        d[:, ind3] = np.minimum(d[:, ind3],
            _point_triangle_dist(p1.pos, p2.verts, faces[ind3, :3]))
    if len(ind4) > 0:
        d[:, ind4] = np.minimum(
            np.minimum(d[:, ind4],
                _point_triangle_dist(p1.pos, p2.verts, faces[ind4][:, [0, 1, 2]])),
            _point_triangle_dist(p1.pos, p2.verts, faces[ind4][:, [2, 3, 0]]))

    return d


def _p_poly_dist(x: float, y: float,
        xv: np.ndarray, yv: np.ndarray) -> float:
    """
    MATLAB: distmin3/p_poly_dist

    Distance from point to polygon.
    """
    xv = xv.ravel()
    yv = yv.ravel()
    nv = len(xv)

    # close polygon if not closed
    if xv[0] != xv[-1] or yv[0] != yv[-1]:
        xv = np.append(xv, xv[0])
        yv = np.append(yv, yv[0])
        nv = nv + 1

    # line parameters
    a_coeff = -np.diff(yv)
    b_coeff = np.diff(xv)
    c_coeff = yv[1:] * xv[:-1] - xv[1:] * yv[:-1]

    # projection of point onto each rib
    ab = 1.0 / (a_coeff ** 2 + b_coeff ** 2 + 1e-30)
    vv = a_coeff * x + b_coeff * y + c_coeff
    xp = x - (a_coeff * ab) * vv
    yp = y - (b_coeff * ab) * vv

    # find where projected point is inside segment
    idx_x = ((xp >= np.minimum(xv[:-1], xv[1:])) & (xp <= np.maximum(xv[:-1], xv[1:])))
    idx_y = ((yp >= np.minimum(yv[:-1], yv[1:])) & (yp <= np.maximum(yv[:-1], yv[1:])))
    idx = idx_x & idx_y

    # distance to vertices
    dv = np.sqrt((xv[:-1] - x) ** 2 + (yv[:-1] - y) ** 2)

    if not np.any(idx):
        d = np.min(dv)
    else:
        dp = np.sqrt((xp[idx] - x) ** 2 + (yp[idx] - y) ** 2)
        d = min(np.min(dv), np.min(dp))

    # check if point is inside polygon
    from matplotlib.path import Path
    polygon = Path(np.column_stack([xv, yv]))
    if polygon.contains_point((x, y)):
        d = -d

    return d


def distmin3(p: object,
        pos: np.ndarray,
        cutoff: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    MATLAB: Misc/distmin3.m

    Minimum distance in 3D between particle faces and positions.

    Parameters
    ----------
    p : particle object
    pos : ndarray, shape (m, 3)
    cutoff : float, optional

    Returns
    -------
    dmin : ndarray, shape (m,)
    ind : ndarray, shape (m,)
    """
    # find nearest faces using pdist2
    d_all = pdist2(pos, p.pos)
    ind = np.argmin(d_all, axis = 1)

    # signed distance along normal direction
    dmin = np.sum((pos - p.pos[ind, :]) * p.nvec[ind, :], axis = 1)

    if cutoff is not None and cutoff == 0:
        return dmin, ind

    if cutoff is None:
        cutoff = np.inf

    # refine for close positions
    close_idx = np.where(np.abs(dmin) <= cutoff)[0]

    for i in close_idx:
        pos0 = p.pos[ind[i], :]
        # project onto plane perpendicular to nvec
        x = np.dot(pos[i, :] - pos0, p.tvec1[ind[i], :])
        y = np.dot(pos[i, :] - pos0, p.tvec2[ind[i], :])

        face = p.faces[ind[i], :]
        valid = face[~np.isnan(face)].astype(int)
        verts_local = p.verts[valid, :]

        xv = np.sum((verts_local - pos0[np.newaxis, :]) * p.tvec1[ind[i], :][np.newaxis, :], axis = 1)
        yv = np.sum((verts_local - pos0[np.newaxis, :]) * p.tvec2[ind[i], :][np.newaxis, :], axis = 1)

        rmin = _p_poly_dist(x, y, xv, yv)
        if rmin > 0:
            dmin[i] = np.sign(dmin[i]) * np.sqrt(dmin[i] ** 2 + rmin ** 2)

    return dmin, ind
