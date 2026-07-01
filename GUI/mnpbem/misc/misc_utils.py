"""
Miscellaneous utility functions for MNPBEM.

MATLAB: Misc/nettable.m, Misc/patchcurvature.m, Misc/subarray.m,
        +misc/memsize.m, +misc/round.m, @mem/mem.m, Misc/multiWaitbar.m
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union


def nettable(faces: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    MATLAB: Misc/nettable.m

    Table of connections between vertices.

    Parameters
    ----------
    faces : ndarray, shape (n_faces, 3 or 4)
        Face connectivity (4th column NaN for triangles)

    Returns
    -------
    net : ndarray, shape (m, 2)
        List of vertex connections
    inet : ndarray, shape (m,)
        Face-to-net index (which face each connection belongs to)
    """
    net_list = []
    inet_list = []

    # triangular faces
    if faces.shape[1] >= 4:
        i3 = np.where(np.isnan(faces[:, 3]))[0]
    else:
        i3 = np.arange(faces.shape[0])

    if len(i3) > 0:
        f = faces[i3, :3].astype(int)
        # three edges per triangle
        edges_3 = np.empty((3 * len(i3), 2), dtype = int)
        edges_3[:len(i3), 0] = f[:, 0]
        edges_3[:len(i3), 1] = f[:, 1]
        edges_3[len(i3):2 * len(i3), 0] = f[:, 1]
        edges_3[len(i3):2 * len(i3), 1] = f[:, 2]
        edges_3[2 * len(i3):, 0] = f[:, 2]
        edges_3[2 * len(i3):, 1] = f[:, 0]
        net_list.append(edges_3)

        inet_3 = np.tile(i3, 3)
        inet_list.append(inet_3)

    # quadrilateral faces
    if faces.shape[1] >= 4:
        i4 = np.where(~np.isnan(faces[:, 3]))[0]
    else:
        i4 = np.array([], dtype = int)

    if len(i4) > 0:
        f = faces[i4, :4].astype(int)
        # four edges per quad
        edges_4 = np.empty((4 * len(i4), 2), dtype = int)
        edges_4[:len(i4), 0] = f[:, 0]
        edges_4[:len(i4), 1] = f[:, 1]
        edges_4[len(i4):2 * len(i4), 0] = f[:, 1]
        edges_4[len(i4):2 * len(i4), 1] = f[:, 2]
        edges_4[2 * len(i4):3 * len(i4), 0] = f[:, 2]
        edges_4[2 * len(i4):3 * len(i4), 1] = f[:, 3]
        edges_4[3 * len(i4):, 0] = f[:, 3]
        edges_4[3 * len(i4):, 1] = f[:, 0]
        net_list.append(edges_4)

        inet_4 = np.tile(i4, 4)
        inet_list.append(inet_4)

    if len(net_list) == 0:
        return np.empty((0, 2), dtype = int), np.empty(0, dtype = int)

    total_net = sum(e.shape[0] for e in net_list)
    net = np.empty((total_net, 2), dtype = int)
    inet = np.empty(total_net, dtype = int)

    offset = 0
    for edges, faces_idx in zip(net_list, inet_list):
        n = edges.shape[0]
        net[offset:offset + n] = edges
        inet[offset:offset + n] = faces_idx
        offset += n

    return net, inet


def patchcurvature(verts: np.ndarray, faces: np.ndarray,
        usethird: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    MATLAB: Misc/patchcurvature.m

    Principal curvature computation for triangulated mesh.

    Parameters
    ----------
    verts : ndarray, shape (nv, 3)
        Vertex coordinates
    faces : ndarray, shape (nf, 3)
        Face connectivity (integer, 0-based)
    usethird : bool
        Use third-order neighbors for smoother curvature

    Returns
    -------
    c_mean : ndarray, shape (nv,)
        Mean curvature
    c_gaussian : ndarray, shape (nv,)
        Gaussian curvature
    dir1 : ndarray, shape (nv, 3)
        First principal direction
    dir2 : ndarray, shape (nv, 3)
        Second principal direction
    lambda1 : ndarray, shape (nv,)
        First principal curvature value
    lambda2 : ndarray, shape (nv,)
        Second principal curvature value
    """
    nv = verts.shape[0]
    faces_int = faces[:, :3].astype(int)

    # compute vertex normals
    normals = _patch_normals(verts, faces_int)

    # compute rotation matrices
    m_arr = np.zeros((nv, 3, 3))
    m_inv_arr = np.zeros((nv, 3, 3))
    for i in range(nv):
        m_arr[i], m_inv_arr[i] = _vector_rotation_matrix(normals[i])

    # get vertex neighbors
    neighbors = _vertex_neighbours(verts, faces_int)

    # compute curvature per vertex
    lambda1 = np.zeros(nv)
    lambda2 = np.zeros(nv)
    dir1 = np.zeros((nv, 3))
    dir2 = np.zeros((nv, 3))

    for i in range(nv):
        # get first and second (and optionally third) ring neighbors
        ne_i = neighbors[i]
        if len(ne_i) == 0:
            continue

        if not usethird:
            # second ring
            ne_set = set()
            for j in ne_i:
                ne_set.update(neighbors[j])
            nce = list(ne_set)
        else:
            # third ring
            ne_set2 = set()
            for j in ne_i:
                ne_set2.update(neighbors[j])
            ne_set3 = set()
            for j in ne_set2:
                ne_set3.update(neighbors[j])
            nce = list(ne_set3)

        if len(nce) < 6:
            continue

        ve = verts[nce, :]

        # rotate to make normal [-1 0 0]
        we = ve @ m_inv_arr[i].T
        f_val = we[:, 0]
        x_val = we[:, 1]
        y_val = we[:, 2]

        # fit quadratic patch: f(x,y) = ax^2 + by^2 + cxy + dx + ey + f
        fm = np.column_stack([
            x_val ** 2, y_val ** 2, x_val * y_val,
            x_val, y_val, np.ones(len(x_val))])

        # least squares solve
        try:
            abcdef, _, _, _ = np.linalg.lstsq(fm, f_val, rcond = None)
        except np.linalg.LinAlgError:
            continue

        a_coef = abcdef[0]
        b_coef = abcdef[1]
        c_coef = abcdef[2]

        # eigenvalues of Hessian
        dxx = 2 * a_coef
        dxy = c_coef
        dyy = 2 * b_coef

        l1, l2, i1, i2 = _eig2(dxx, dxy, dyy)
        lambda1[i] = l1
        lambda2[i] = l2

        d1 = np.array([0, i1[0], i1[1]]) @ m_arr[i].T
        d2 = np.array([0, i2[0], i2[1]]) @ m_arr[i].T

        norm1 = np.sqrt(np.sum(d1 ** 2))
        norm2 = np.sqrt(np.sum(d2 ** 2))
        if norm1 > 0:
            dir1[i] = d1 / norm1
        if norm2 > 0:
            dir2[i] = d2 / norm2

    c_mean = (lambda1 + lambda2) / 2
    c_gaussian = lambda1 * lambda2

    return c_mean, c_gaussian, dir1, dir2, lambda1, lambda2


def _eig2(dxx: float, dxy: float,
        dyy: float) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    MATLAB: patchcurvature/eig2

    2x2 eigenvalue decomposition.
    """
    tmp = np.sqrt((dxx - dyy) ** 2 + 4 * dxy ** 2)
    v2x = 2 * dxy
    v2y = dyy - dxx + tmp

    mag = np.sqrt(v2x ** 2 + v2y ** 2)
    if mag != 0:
        v2x /= mag
        v2y /= mag

    v1x = -v2y
    v1y = v2x

    mu1 = abs(0.5 * (dxx + dyy + tmp))
    mu2 = abs(0.5 * (dxx + dyy - tmp))

    if mu1 < mu2:
        return mu1, mu2, np.array([v1x, v1y]), np.array([v2x, v2y])
    else:
        return mu2, mu1, np.array([v2x, v2y]), np.array([v1x, v1y])


def _patch_normals(verts: np.ndarray,
        faces: np.ndarray) -> np.ndarray:
    """
    MATLAB: patchcurvature/patchnormals

    Compute vertex normals from face normals weighted by face angles.
    """
    fa = faces[:, 0]
    fb = faces[:, 1]
    fc = faces[:, 2]

    # edge vectors
    e1 = verts[fa] - verts[fb]
    e2 = verts[fb] - verts[fc]
    e3 = verts[fc] - verts[fa]

    # normalize
    e1_n = np.sqrt(np.sum(e1 ** 2, axis = 1, keepdims = True)) + 1e-30
    e2_n = np.sqrt(np.sum(e2 ** 2, axis = 1, keepdims = True)) + 1e-30
    e3_n = np.sqrt(np.sum(e3 ** 2, axis = 1, keepdims = True)) + 1e-30

    e1_norm = e1 / e1_n
    e2_norm = e2 / e2_n
    e3_norm = e3 / e3_n

    # angles at each vertex
    angle_a = np.arccos(np.clip(np.sum(e1_norm * (-e3_norm), axis = 1), -1, 1))
    angle_b = np.arccos(np.clip(np.sum(e2_norm * (-e1_norm), axis = 1), -1, 1))
    angle_c = np.arccos(np.clip(np.sum(e3_norm * (-e2_norm), axis = 1), -1, 1))

    # face normals
    normal = np.cross(e1, e3)

    # accumulate vertex normals
    nv = verts.shape[0]
    vert_normals = np.zeros((nv, 3))
    for i in range(len(faces)):
        vert_normals[fa[i]] += normal[i] * angle_a[i]
        vert_normals[fb[i]] += normal[i] * angle_b[i]
        vert_normals[fc[i]] += normal[i] * angle_c[i]

    # normalize
    v_norm = np.sqrt(np.sum(vert_normals ** 2, axis = 1, keepdims = True)) + 1e-30
    vert_normals = vert_normals / v_norm

    return vert_normals


def _vector_rotation_matrix(v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    MATLAB: patchcurvature/VectorRotationMatrix

    Rotation matrix to align vector v with [1, 0, 0].
    """
    v = v / (np.sqrt(np.sum(v ** 2)) + 1e-30)
    np.random.seed(42)
    k = np.random.randn(3)

    # cross product to get orthogonal vector
    l_vec = np.cross(k, v)
    l_norm = np.sqrt(np.sum(l_vec ** 2))
    if l_norm < 1e-10:
        k = np.array([1.0, 0.0, 0.0])
        l_vec = np.cross(k, v)
        l_norm = np.sqrt(np.sum(l_vec ** 2))
    l_vec = l_vec / (l_norm + 1e-30)

    k_vec = np.cross(l_vec, v)
    k_norm = np.sqrt(np.sum(k_vec ** 2))
    k_vec = k_vec / (k_norm + 1e-30)

    m_inv = np.column_stack([v, l_vec, k_vec])
    m = np.linalg.inv(m_inv)

    return m, m_inv


def _vertex_neighbours(verts: np.ndarray,
        faces: np.ndarray) -> List[List[int]]:
    """
    MATLAB: patchcurvature/vertex_neighbours

    Find vertex neighbors from face connectivity.
    """
    nv = verts.shape[0]
    neighbors = [set() for _ in range(nv)]

    for i in range(faces.shape[0]):
        fa, fb, fc = faces[i, 0], faces[i, 1], faces[i, 2]
        neighbors[fa].add(fb)
        neighbors[fa].add(fc)
        neighbors[fb].add(fa)
        neighbors[fb].add(fc)
        neighbors[fc].add(fa)
        neighbors[fc].add(fb)

    return [list(s) for s in neighbors]


def memsize() -> int:
    """
    MATLAB: +misc/memsize.m

    Memory size used for working through large arrays.
    """
    return 5000 * 5000


def round_left(x: Union[float, np.ndarray],
        n: int) -> Union[float, np.ndarray]:
    """
    MATLAB: +misc/round.m

    Round each element of x to the left of the decimal point.
    """
    return np.round(x * 10 ** n) * 10 ** (-n)


class Mem(object):
    """
    Memory information tracker.

    MATLAB: @mem

    Simple memory monitoring utility.
    The MATLAB version is Windows-specific; this Python version
    uses psutil if available.
    """

    def __init__(self) -> None:
        self._flag = False
        self._report = []

    def on(self) -> None:
        self._flag = True
        self._report = []

    def off(self) -> None:
        self._flag = False

    def clear(self) -> None:
        self._report = []

    def set(self, key: str = '') -> None:
        if not self._flag:
            return

        try:
            import psutil
            mem = psutil.Process().memory_info().rss / 1024 / 1024
            self._report.append((key, '{:.1f} MB'.format(mem)))
        except ImportError:
            self._report.append((key, 'psutil not available'))

    @property
    def flag(self) -> bool:
        return self._flag

    @property
    def report(self) -> List[tuple]:
        return self._report

    def __repr__(self) -> str:
        lines = ['Mem report:']
        for key, val in self._report:
            lines.append('  {} : {}'.format(key, val))
        return '\n'.join(lines)


def multi_waitbar(label: str, progress: float,
        **kwargs: Any) -> None:
    """
    MATLAB: Misc/multiWaitbar.m

    Multi-progress bar.
    The MATLAB version is a complex GUI widget (~800 lines).
    In Python, use tqdm instead. This function provides a simple fallback.
    """
    try:
        from tqdm import tqdm
        # tqdm handles progress bars natively
        pass
    except ImportError:
        pass

    # simple text-based progress
    bar_len = 40
    filled = int(bar_len * progress)
    bar = '=' * filled + '-' * (bar_len - filled)
    print('\r{}: [{}] {:.0f}%'.format(label, bar, progress * 100), end = '')
    if progress >= 1.0:
        print()
