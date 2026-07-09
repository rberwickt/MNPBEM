import numpy as np
from typing import Tuple, Optional, Dict, Any, Callable, List
from scipy.spatial import Delaunay, ConvexHull

from ..utils.matlab_compat import mlinspace, mcos, msin, matan2, mround, msqrt


def inpoly(p: np.ndarray,
        node: np.ndarray,
        edge: Optional[np.ndarray] = None,
        reltol: float = 1.0e-12) -> Tuple[np.ndarray, np.ndarray]:

    # MATLAB Mesh2d/inpoly.m - point-in-polygon test using crossing number
    p = np.asarray(p, dtype = float)
    node = np.asarray(node, dtype = float)

    if p.ndim == 1:
        p = p.reshape(1, -1)

    assert p.shape[1] == 2, '[error] P must be an Nx2 array.'
    assert node.shape[1] == 2, '[error] NODE must be an Mx2 array.'

    nnode = node.shape[0]
    if edge is None:
        idx = np.arange(nnode)
        edge = np.empty((nnode, 2), dtype = int)
        edge[:nnode - 1, 0] = idx[:nnode - 1]
        edge[:nnode - 1, 1] = idx[1:nnode]
        edge[nnode - 1, 0] = nnode - 1
        edge[nnode - 1, 1] = 0
    else:
        edge = np.asarray(edge, dtype = int)

    assert edge.shape[1] == 2, '[error] EDGE must be an Mx2 array.'

    n = p.shape[0]
    nc = edge.shape[0]

    # choose direction with biggest range as y-coordinate
    dxy = np.max(p, axis = 0) - np.min(p, axis = 0)
    if dxy[0] > dxy[1]:
        p = p[:, [1, 0]]
        node = node[:, [1, 0]]

    # polygon bounding-box tolerance
    dxy_node = np.max(node, axis = 0) - np.min(node, axis = 0)
    tol = reltol * min(dxy_node)
    if tol == 0.0:
        tol = reltol

    # sort test points by y-value
    sort_idx = np.argsort(p[:, 1])
    y = p[sort_idx, 1]
    x = p[sort_idx, 0]

    cn = np.zeros(n, dtype = bool)
    on = np.zeros(n, dtype = bool)

    for k in range(nc):
        n1 = edge[k, 0]
        n2 = edge[k, 1]

        y1 = node[n1, 1]
        y2 = node[n2, 1]
        if y1 < y2:
            x1 = node[n1, 0]
            x2 = node[n2, 0]
        else:
            yt = y1
            y1 = y2
            y2 = yt
            x1 = node[n2, 0]
            x2 = node[n1, 0]

        if x1 > x2:
            xmin = x2
            xmax = x1
        else:
            xmin = x1
            xmax = x2

        # binary search for first point with y >= y1
        if y[0] >= y1:
            start = 0
        elif y[n - 1] < y1:
            start = n
        else:
            lower = 0
            upper = n - 1
            for _bs in range(n):
                start = (lower + upper) // 2
                if y[start] < y1:
                    lower = start + 1
                elif start > 0 and y[start - 1] < y1:
                    break
                else:
                    upper = start - 1
            else:
                start = lower

        for j in range(start, n):
            Y = y[j]
            if Y <= y2:
                X = x[j]
                if X >= xmin:
                    if X <= xmax:
                        on[j] = on[j] or (abs((y2 - Y) * (x1 - X) - (y1 - Y) * (x2 - X)) <= tol)
                        if (Y < y2) and ((y2 - y1) * (X - x1) < (Y - y1) * (x2 - x1)):
                            cn[j] = not cn[j]
                elif Y < y2:
                    cn[j] = not cn[j]
            else:
                break

    # re-index to undo the sorting
    result_cn = np.zeros(n, dtype = bool)
    result_on = np.zeros(n, dtype = bool)
    result_cn[sort_idx] = cn | on
    result_on[sort_idx] = on

    return result_cn, result_on


def triarea(p: np.ndarray,
        t: np.ndarray) -> np.ndarray:

    # MATLAB Mesh2d/triarea.m - signed triangle area (CCW positive)
    d12 = p[t[:, 1], :] - p[t[:, 0], :]
    d13 = p[t[:, 2], :] - p[t[:, 0], :]
    A = d12[:, 0] * d13[:, 1] - d12[:, 1] * d13[:, 0]
    return A


def quality(p: np.ndarray,
        t: np.ndarray) -> np.ndarray:

    # MATLAB Mesh2d/quality.m - triangle quality 0 <= q <= 1
    p1 = p[t[:, 0], :]
    p2 = p[t[:, 1], :]
    p3 = p[t[:, 2], :]

    d12 = p2 - p1
    d13 = p3 - p1
    d23 = p3 - p2

    # 3.4641 = 4 * sqrt(3)
    q = 3.4641 * np.abs(d12[:, 0] * d13[:, 1] - d12[:, 1] * d13[:, 0]) / np.sum(d12 ** 2 + d13 ** 2 + d23 ** 2, axis = 1)
    return q


def circumcircle(p: np.ndarray,
        t: np.ndarray) -> np.ndarray:

    # MATLAB Mesh2d/circumcircle.m - circumcircle center and radius^2
    cc = np.zeros((t.shape[0], 3))

    p1 = p[t[:, 0], :]
    p2 = p[t[:, 1], :]
    p3 = p[t[:, 2], :]

    a1 = p2 - p1
    a2 = p3 - p1
    b1 = np.sum(a1 * (p2 + p1), axis = 1)
    b2 = np.sum(a2 * (p3 + p1), axis = 1)

    idet = 0.5 / (a1[:, 0] * a2[:, 1] - a2[:, 0] * a1[:, 1] + np.finfo(float).eps)

    cc[:, 0] = (a2[:, 1] * b1 - a1[:, 1] * b2) * idet
    cc[:, 1] = (-a2[:, 0] * b1 + a1[:, 0] * b2) * idet
    cc[:, 2] = np.sum((p1 - cc[:, :2]) ** 2, axis = 1)

    return cc


def fixmesh(p: np.ndarray,
        t: np.ndarray,
        pfun: Optional[np.ndarray] = None,
        tfun: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:

    # MATLAB Mesh2d/fixmesh.m - clean up mesh
    TOL = 1.0e-10

    # remove duplicate nodes (MATLAB L64: unique(p,'rows') — no rounding)
    _, i_unique, j_map = np.unique(p, axis = 0, return_index = True, return_inverse = True)
    if pfun is not None:
        pfun = pfun[i_unique]
    p = p[i_unique]
    t = j_map[t]

    # triangle area
    A = triarea(p, t)
    Ai = A < 0.0
    Aj = np.abs(A) > TOL * np.max(np.abs(A)) if len(A) > 0 else np.ones(len(A), dtype = bool)

    # flip node numbering to give CCW order
    t_flip = t[Ai].copy()
    t_flip[:, [0, 1]] = t_flip[:, [1, 0]]
    t[Ai] = t_flip

    # remove zero area triangles
    t = t[Aj]
    if tfun is not None:
        tfun = tfun[Aj]

    # remove unused nodes
    used = np.unique(t.ravel())
    if len(used) < p.shape[0]:
        remap = np.full(p.shape[0], -1, dtype = int)
        remap[used] = np.arange(len(used))
        p = p[used]
        if pfun is not None:
            pfun = pfun[used]
        t = remap[t]

    return p, t, pfun, tfun


def smoothmesh(p: np.ndarray,
        t: np.ndarray,
        maxit: int = 20,
        tol: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:

    # MATLAB Mesh2d/smoothmesh.m - Laplacian smoothing
    p, t, _, _ = fixmesh(p, t)

    n = p.shape[0]
    numt = t.shape[0]

    # sparse connectivity matrix
    rows = np.empty(6 * numt, dtype = int)
    cols = np.empty(6 * numt, dtype = int)
    pairs = [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]
    for idx, (a, b) in enumerate(pairs):
        rows[idx * numt:(idx + 1) * numt] = t[:, a]
        cols[idx * numt:(idx + 1) * numt] = t[:, b]

    from scipy.sparse import csr_matrix
    S = csr_matrix((np.ones(6 * numt), (rows, cols)), shape = (n, n))
    W = np.array(S.sum(axis = 1)).ravel()

    # find boundary nodes
    edge_all = np.empty((3 * numt, 2), dtype = int)
    edge_all[:numt] = t[:, [0, 1]]
    edge_all[numt:2 * numt] = t[:, [0, 2]]
    edge_all[2 * numt:] = t[:, [1, 2]]
    edge_sorted = np.sort(edge_all, axis = 1)

    # find unique and boundary edges
    _, counts = np.unique(edge_sorted, axis = 0, return_counts = True)
    # use lexsort for proper identification
    sorted_idx = np.lexsort((edge_sorted[:, 1], edge_sorted[:, 0]))
    edge_sorted_2 = edge_sorted[sorted_idx]
    is_dup = np.zeros(len(edge_sorted_2), dtype = bool)
    is_dup[:-1] |= np.all(edge_sorted_2[:-1] == edge_sorted_2[1:], axis = 1)
    is_dup[1:] |= np.all(edge_sorted_2[:-1] == edge_sorted_2[1:], axis = 1)
    bnd_edges = edge_sorted_2[~is_dup]
    bnd_nodes = np.unique(bnd_edges.ravel())

    # unique edges for length computation
    unique_edges = np.unique(edge_sorted, axis = 0)

    L = np.maximum(msqrt(np.sum((p[unique_edges[:, 0]] - p[unique_edges[:, 1]]) ** 2, axis = 1)), np.finfo(float).eps)

    for it in range(maxit):
        pnew = np.zeros_like(p)
        sp = S.dot(p)
        pnew[:, 0] = sp[:, 0] / W
        pnew[:, 1] = sp[:, 1] / W
        pnew[bnd_nodes] = p[bnd_nodes]
        p = pnew

        Lnew = np.maximum(msqrt(np.sum((p[unique_edges[:, 0]] - p[unique_edges[:, 1]]) ** 2, axis = 1)), np.finfo(float).eps)
        move = np.max(np.abs((Lnew - L) / Lnew))
        if move < tol:
            break
        L = Lnew

    return p, t


def refine(p: np.ndarray,
        t: np.ndarray,
        ti: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:

    # MATLAB Mesh2d/refine.m - refine triangulation (uniform or selective)
    p, t, _, _ = fixmesh(p, t)

    if ti is None:
        ti = np.ones(t.shape[0], dtype = bool)
    else:
        ti = np.asarray(ti, dtype = bool)

    numt = t.shape[0]
    vect = np.arange(numt)

    # edge connectivity
    e_all = np.empty((3 * numt, 2), dtype = int)
    e_all[:numt] = t[:, [0, 1]]
    e_all[numt:2 * numt] = t[:, [1, 2]]
    e_all[2 * numt:] = t[:, [2, 0]]

    e_sorted = np.sort(e_all, axis = 1)
    e_unique, j_map = np.unique(e_sorted, axis = 0, return_inverse = True)

    te = np.empty((numt, 3), dtype = int)
    te[:, 0] = j_map[vect]
    te[:, 1] = j_map[vect + numt]
    te[:, 2] = j_map[vect + 2 * numt]

    split = np.zeros(e_unique.shape[0], dtype = bool)
    split[te[ti].ravel()] = True

    # propagate splits to maintain compatibility
    while True:
        split3 = np.sum(split[te].astype(int), axis = 1) >= 2
        old_count = np.sum(split)
        split[te[split3].ravel()] = True
        if np.sum(split) == old_count:
            break

    split1 = np.sum(split[te].astype(int), axis = 1) == 1

    np_count = p.shape[0]
    nsplit = np.sum(split)
    pm = 0.5 * (p[e_unique[split, 0]] + p[e_unique[split, 1]])

    total_p = np_count + nsplit
    p_new = np.empty((total_p, 2))
    p_new[:np_count] = p
    p_new[np_count:] = pm

    # map split edges to new node indices
    i_map = np.full(e_unique.shape[0], -1, dtype = int)
    i_map[split] = np.arange(nsplit) + np_count

    # new triangles in split3 case
    tnew_list = []
    keep = ~(split1 | (np.sum(split[te].astype(int), axis = 1) >= 2))
    tnew_list.append(t[keep])

    split3_mask = np.sum(split[te].astype(int), axis = 1) >= 2
    if np.any(split3_mask):
        n1 = t[split3_mask, 0]
        n2 = t[split3_mask, 1]
        n3 = t[split3_mask, 2]
        n4 = i_map[te[split3_mask, 0]]
        n5 = i_map[te[split3_mask, 1]]
        n6 = i_map[te[split3_mask, 2]]

        tnew_list.append(np.column_stack([n1, n4, n6]))
        tnew_list.append(np.column_stack([n4, n2, n5]))
        tnew_list.append(np.column_stack([n5, n3, n6]))
        tnew_list.append(np.column_stack([n4, n5, n6]))

    # new triangles in split1 case
    if np.any(split1):
        split1_idx = np.where(split1)[0]
        for k in split1_idx:
            col = -1
            for c in range(3):
                if split[te[k, c]]:
                    col = c
                    break
            N1 = col
            N2 = (col + 1) % 3
            N3 = (col + 2) % 3
            nn1 = t[k, N1]
            nn2 = t[k, N2]
            nn3 = t[k, N3]
            nn4 = i_map[te[k, col]]
            tnew_list.append(np.array([[nn1, nn4, nn3], [nn4, nn2, nn3]]))

    t_new = np.vstack(tnew_list) if len(tnew_list) > 0 else np.empty((0, 3), dtype = int)

    return p_new, t_new


def connectivity(p: np.ndarray,
        t: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    # MATLAB Mesh2d/connectivity.m
    numt = t.shape[0]
    vect = np.arange(numt)

    e_all = np.empty((3 * numt, 2), dtype = int)
    e_all[:numt] = t[:, [0, 1]]
    e_all[numt:2 * numt] = t[:, [1, 2]]
    e_all[2 * numt:] = t[:, [2, 0]]

    e_sorted = np.sort(e_all, axis = 1)
    e_unique, j_map = np.unique(e_sorted, axis = 0, return_inverse = True)

    te = np.empty((numt, 3), dtype = int)
    te[:, 0] = j_map[vect]
    te[:, 1] = j_map[vect + numt]
    te[:, 2] = j_map[vect + 2 * numt]

    # edge-to-triangle connectivity
    nume = e_unique.shape[0]
    e2t = np.zeros((nume, 2), dtype = int)
    for k in range(numt):
        for j in range(3):
            ce = te[k, j]
            if e2t[ce, 0] == 0:
                e2t[ce, 0] = k + 1  # 1-indexed for MATLAB compatibility
            else:
                e2t[ce, 1] = k + 1

    # boundary nodes
    bnd = np.zeros(p.shape[0], dtype = bool)
    bnd_edges = e_unique[e2t[:, 1] == 0]
    bnd[bnd_edges.ravel()] = True

    return e_unique, te, e2t, bnd


def findedge(p: np.ndarray,
        node: np.ndarray,
        edge: np.ndarray,
        tol: float = 1.0e-08) -> np.ndarray:

    # MATLAB Mesh2d/findedge.m - locate points on edges
    n = p.shape[0]
    nc = edge.shape[0]

    dxy = np.max(p, axis = 0) - np.min(p, axis = 0)
    if dxy[0] > dxy[1]:
        p = p[:, [1, 0]].copy()
        node = node[:, [1, 0]].copy()
    tol_abs = tol * min(dxy)

    sort_idx = np.argsort(p[:, 1])
    y = p[sort_idx, 1]
    x = p[sort_idx, 0]

    enum = np.zeros(n, dtype = int)
    for k in range(nc):
        n1 = edge[k, 0]
        n2 = edge[k, 1]

        y1 = node[n1, 1]
        y2 = node[n2, 1]
        if y1 < y2:
            x1 = node[n1, 0]
            x2 = node[n2, 0]
        else:
            yt = y1
            y1 = y2
            y2 = yt
            x1 = node[n2, 0]
            x2 = node[n1, 0]

        # binary search
        if n == 0:
            continue
        if y[0] >= y1:
            start = 0
        elif y[n - 1] < y1:
            start = n
        else:
            lower = 0
            upper = n - 1
            start = 0
            for _bs in range(n):
                start = (lower + upper) // 2
                if y[start] < y1:
                    lower = start + 1
                elif start > 0 and y[start - 1] < y1:
                    break
                else:
                    upper = start - 1

        for j in range(start, n):
            Y = y[j]
            if Y <= y2:
                X = x[j]
                if abs((y2 - Y) * (x1 - X) - (y1 - Y) * (x2 - X)) < tol_abs:
                    enum[j] = k + 1  # 1-indexed
            else:
                break

    result = np.zeros(n, dtype = int)
    result[sort_idx] = enum
    return result


def dist2poly(p: np.ndarray,
        edgexy: np.ndarray,
        lim: Optional[np.ndarray] = None) -> np.ndarray:

    # MATLAB Mesh2d/dist2poly.m - distance from points to polygon boundary
    np_count = p.shape[0]
    ne = edgexy.shape[0]

    if lim is None:
        lim = np.full(np_count, np.inf)
    else:
        lim = np.asarray(lim, dtype = float).copy()

    dxy = np.max(p, axis = 0) - np.min(p, axis = 0)
    if dxy[0] > dxy[1]:
        p = p[:, [1, 0]].copy()
        edgexy = edgexy[:, [1, 0, 3, 2]].copy()

    # ensure edgexy[:, [0,1]] has the lower y value
    swap = edgexy[:, 3] < edgexy[:, 1]
    edgexy_swap = edgexy[swap].copy()
    edgexy[swap] = edgexy_swap[:, [2, 3, 0, 1]]

    tol = 1000.0 * np.finfo(float).eps * max(dxy)
    L = np.zeros(np_count)

    for k in range(np_count):
        x_pt = p[k, 0]
        y_pt = p[k, 1]
        d = lim[k]

        for j in range(ne):
            y1 = edgexy[j, 1]
            y2 = edgexy[j, 3]
            if y2 < y_pt - d:
                continue
            if y1 > y_pt + d:
                continue

            x1 = edgexy[j, 0]
            x2 = edgexy[j, 2]
            xmin = min(x1, x2)
            xmax = max(x1, x2)

            if xmin > x_pt + d or xmax < x_pt - d:
                continue

            x2mx1 = x2 - x1
            y2my1 = y2 - y1
            denom = x2mx1 ** 2 + y2my1 ** 2
            if denom < np.finfo(float).eps:
                continue

            r = ((x_pt - x1) * x2mx1 + (y_pt - y1) * y2my1) / denom
            r = max(min(r, 1.0), 0.0)

            dj = (x1 + r * x2mx1 - x_pt) ** 2 + (y1 + r * y2my1 - y_pt) ** 2
            if dj < d ** 2 and dj > tol:
                d = msqrt(dj)

        L[k] = d

    return L


def _mydelaunayn(p: np.ndarray) -> np.ndarray:

    # MATLAB Mesh2d/mydelaunayn.m - Delaunay triangulation with scaling.
    # MATLAB's `delaunayn` in 2D uses qhull options 'Qt Qbb Qc' (triangulated
    # output, bounding-box scaling, max-coord output). scipy's default is
    # 'Qbb Qc Qz' which adds a 'Qz' cosphere point and gives a different
    # tie-break for cocircular input groups. Passing 'Qt Qbb Qc' reproduces
    # MATLAB's triangulation set bit-identically (vertex order within each
    # row may still differ — qhull facet ordering is wrapper-specific).
    maxxy = np.max(p, axis = 0)
    minxy = np.min(p, axis = 0)
    center = 0.5 * (minxy + maxxy)
    scale = 0.5 * min(maxxy - minxy)
    if scale < np.finfo(float).eps:
        scale = 1.0

    ps = (p - center) / scale

    try:
        tri = Delaunay(ps, qhull_options = 'Qt Qbb Qc')
        t = tri.simplices
    except Exception:
        # add small jitter and retry (seeded for determinism)
        rng = np.random.default_rng(0)
        jitter = rng.standard_normal(ps.shape) * 1e-10
        tri = Delaunay(ps + jitter, qhull_options = 'Qt Qbb Qc')
        t = tri.simplices

    return t


def _tricentre(t: np.ndarray,
        f: np.ndarray) -> np.ndarray:

    return (f[t[:, 0]] + f[t[:, 1]] + f[t[:, 2]]) / 3.0


def _longest(p: np.ndarray,
        t: np.ndarray) -> np.ndarray:

    d1 = np.sum((p[t[:, 1]] - p[t[:, 0]]) ** 2, axis = 1)
    d2 = np.sum((p[t[:, 2]] - p[t[:, 1]]) ** 2, axis = 1)
    d3 = np.sum((p[t[:, 0]] - p[t[:, 2]]) ** 2, axis = 1)
    return msqrt(np.maximum(np.maximum(d1, d2), d3))


def _getedges(t: np.ndarray,
        n: int) -> np.ndarray:

    # unique edges and boundary edges
    e_all = np.empty((3 * t.shape[0], 2), dtype = int)
    e_all[:t.shape[0]] = np.sort(t[:, [0, 1]], axis = 1)
    e_all[t.shape[0]:2 * t.shape[0]] = np.sort(t[:, [0, 2]], axis = 1)
    e_all[2 * t.shape[0]:] = np.sort(t[:, [1, 2]], axis = 1)

    e_sorted_idx = np.lexsort((e_all[:, 1], e_all[:, 0]))
    e_sorted = e_all[e_sorted_idx]

    is_shared = np.zeros(len(e_sorted), dtype = bool)
    is_shared[:-1] |= np.all(e_sorted[:-1] == e_sorted[1:], axis = 1)
    is_shared[1:] |= np.all(e_sorted[:-1] == e_sorted[1:], axis = 1)

    bnd = e_sorted[~is_shared]
    internal = e_sorted[is_shared]

    # take every other internal edge (they come in pairs)
    internal_unique = internal[::2]

    total_len = bnd.shape[0] + internal_unique.shape[0]
    e = np.empty((total_len, 2), dtype = int)
    e[:bnd.shape[0]] = bnd
    e[bnd.shape[0]:] = internal_unique

    return e


def _rotate(p: np.ndarray,
        theta: float) -> np.ndarray:

    s = msin(theta)
    c = mcos(theta)
    rot = np.array([[c, s], [-s, c]])
    return p @ rot


def _minrectangle(p: np.ndarray) -> float:

    n = p.shape[0]
    if n <= 2:
        return 0.0

    try:
        hull = ConvexHull(p)
    except Exception:
        return 0.0

    # MATLAB minrectangle: e = convhulln(p); i = unique(e(:)); pp = p(i,:).
    # `unique()` returns indices ascending, so MATLAB's `pp` is pos subset
    # in input-index order. The edge iteration then starts at `pp(1)→pp(2)`.
    #
    # BUT MATLAB's `convhulln` returns edge pairs in qhull's internal facet
    # order, NOT input order. For a regular n-gon with all points on the
    # hull, qhull's first facet empirically starts at vertex ceil(n/2)
    # (1-indexed MATLAB), i.e. ceil(n/2)-1 (0-indexed) = ceil(n/2) after
    # subtracting 1 and converting, simplifying to floor((n-1)/2)+1 = n//2
    # for even n, (n+1)//2 for odd n — i.e. roughly the midpoint of the
    # input CCW listing. Because MATLAB then rotates convhulln's output
    # through `j(e)` using `cumsum(j)` tracking, the effective iteration
    # order starts from that qhull-chosen vertex. For regular polygons many
    # hull edges give FP-tied rectangle areas, so the starting vertex
    # determines which tied theta wins — and therefore which mesh
    # orientation emerges from quadtree (critical for refun-based demos
    # like demodipstat7 where the size function is x-dependent and not
    # rotation-invariant).
    #
    # For non-regular polygons the min-area rectangle is unique, so start
    # offset doesn't matter. Using ceil(n_hull/2) therefore works for both
    # regular (matches MATLAB's qhull start) and non-regular (irrelevant).
    hull_idx_sorted = np.sort(hull.vertices)
    n_hull = len(hull_idx_sorted)
    matlab_start = (n_hull + 1) // 2  # ceil(n_hull / 2), 0-indexed offset
    hull_idx = np.roll(hull_idx_sorted, -matlab_start)
    p_hull = p[hull_idx]

    best_theta = 0.0
    best_area = np.inf

    for k in range(n_hull):
        dxy = p_hull[(k + 1) % n_hull] - p_hull[k]
        ang = matan2(dxy[1], dxy[0])
        theta = -ang

        pr = _rotate(p_hull, theta)
        dxy_r = np.max(pr, axis = 0) - np.min(pr, axis = 0)
        area = dxy_r[0] * dxy_r[1]
        if area < best_area:
            best_area = area
            best_theta = theta

    # ensure long axis aligned with Y
    pr = _rotate(p_hull, best_theta)
    dxy_r = np.max(pr, axis = 0) - np.min(pr, axis = 0)
    if dxy_r[0] > dxy_r[1]:
        best_theta += 0.5 * np.pi

    return best_theta


def _tinterp(p: np.ndarray,
        t: np.ndarray,
        f: np.ndarray,
        pi: np.ndarray,
        i: np.ndarray) -> np.ndarray:

    # MATLAB Mesh2d/tinterp.m - triangle-based linear interpolation
    fi = np.zeros(pi.shape[0])

    out = (i < 0) | np.isnan(i.astype(float))
    if np.any(out):
        # nearest neighbour extrapolation
        from scipy.spatial import cKDTree
        tree = cKDTree(p)
        _, nn_idx = tree.query(pi[out])
        fi[out] = f[nn_idx]

    valid = ~out
    if np.any(valid):
        pin = pi[valid]
        tin = t[i[valid].astype(int)]

        t1 = tin[:, 0]
        t2 = tin[:, 1]
        t3 = tin[:, 2]

        dp1 = pin - p[t1]
        dp2 = pin - p[t2]
        dp3 = pin - p[t3]

        A3 = np.abs(dp1[:, 0] * dp2[:, 1] - dp1[:, 1] * dp2[:, 0])
        A2 = np.abs(dp1[:, 0] * dp3[:, 1] - dp1[:, 1] * dp3[:, 0])
        A1 = np.abs(dp3[:, 0] * dp2[:, 1] - dp3[:, 1] * dp2[:, 0])

        denom = A1 + A2 + A3
        denom[denom < np.finfo(float).eps] = np.finfo(float).eps
        fi[valid] = (A1 * f[t1] + A2 * f[t2] + A3 * f[t3]) / denom

    return fi


def _mytsearch(x: np.ndarray,
        y: np.ndarray,
        t: np.ndarray,
        xi: np.ndarray,
        yi: np.ndarray,
        i_guess: Optional[np.ndarray] = None) -> np.ndarray:

    # MATLAB Mesh2d/mytsearch.m - find enclosing triangle
    p = np.column_stack([x, y])
    pi = np.column_stack([xi, yi])

    # scale to avoid precision issues
    maxxy = np.max(p, axis = 0)
    minxy = np.min(p, axis = 0)
    den = 0.5 * min(maxxy - minxy) if min(maxxy - minxy) > 0 else 1.0

    ps = (p - 0.5 * (minxy + maxxy)) / den
    pis = (pi - 0.5 * (minxy + maxxy)) / den

    ni = len(xi)
    result = np.full(ni, -1, dtype = int)

    # check initial guess if provided (MATLAB mytsearch.m L48 treats `i>0`
    # as valid. Python is 0-indexed; the caller should pass -1 for "no
    # initial guess". We still accept 0 as a valid guess, matching MATLAB's
    # 1-indexed value 1 == Python's 0-indexed value 0.)
    if i_guess is not None and len(i_guess) == ni:
        valid_guess = (i_guess >= 0) & (i_guess < t.shape[0])
        if np.any(valid_guess):
            k_idx = np.where(valid_guess)[0]
            tri_idx = i_guess[k_idx]

            n1 = t[tri_idx, 0]
            n2 = t[tri_idx, 1]
            n3 = t[tri_idx, 2]

            # check if point is inside triangle using cross products
            def _sameside(xa, ya, xb, yb, x1, y1, x2, y2):
                dx = xb - xa
                dy = yb - ya
                a1 = (x1 - xa) * dy - (y1 - ya) * dx
                a2 = (x2 - xa) * dy - (y2 - ya) * dx
                return a1 * a2 >= 0.0

            ok = (_sameside(ps[n1, 0], ps[n1, 1], ps[n2, 0], ps[n2, 1], pis[k_idx, 0], pis[k_idx, 1], ps[n3, 0], ps[n3, 1]) &
                  _sameside(ps[n2, 0], ps[n2, 1], ps[n3, 0], ps[n3, 1], pis[k_idx, 0], pis[k_idx, 1], ps[n1, 0], ps[n1, 1]) &
                  _sameside(ps[n3, 0], ps[n3, 1], ps[n1, 0], ps[n1, 1], pis[k_idx, 0], pis[k_idx, 1], ps[n2, 0], ps[n2, 1]))

            result[k_idx[ok]] = tri_idx[ok]

    # full search for points that failed
    # MATLAB mytsearch.m L70-73 falls back to `for k=1:nt: inpolygon(...);
    # i(temp)=k`, i.e. the LAST triangle containing each point wins. Python
    # must mirror this (not "first matching") for bit-identical meshing.
    need_search = result < 0
    if np.any(need_search):
        search_idx = np.where(need_search)[0]
        # Use inpoly (matches MATLAB inpolygon semantics) per triangle
        for k in range(t.shape[0]):
            tri_verts = ps[t[k, :]]
            inside, _ = inpoly(pis[search_idx], tri_verts)
            if np.any(inside):
                result[search_idx[inside]] = k

    return result


def _userhfun(x: np.ndarray,
        y: np.ndarray,
        fun: Optional[Callable],
        args: List,
        hmax: float,
        xymin: np.ndarray,
        xymax: np.ndarray) -> np.ndarray:

    if fun is not None:
        h = fun(x, y, *args)
    else:
        h = np.full_like(x, np.inf)

    h = np.minimum(h, hmax)
    out = (x > xymax[0]) | (x < xymin[0]) | (y > xymax[1]) | (y < xymin[1])
    h[out] = np.inf
    return h


def _gethdata(hdata: Optional[Dict[str, Any]]) -> Tuple[float, Optional[np.ndarray], Optional[Callable], List]:

    d_hmax = np.inf
    d_edgeh = None
    d_fun = None
    d_args = []

    if hdata is None:
        return d_hmax, d_edgeh, d_fun, d_args

    hmax = hdata.get('hmax', d_hmax)
    edgeh = hdata.get('edgeh', d_edgeh)
    fun = hdata.get('fun', d_fun)
    args = hdata.get('args', d_args)

    return hmax, edgeh, fun, args


def quadtree(node: np.ndarray,
        edge: np.ndarray,
        hdata: Optional[Dict[str, Any]],
        dhmax: float,
        output: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    # MATLAB Mesh2d/quadtree.m - quadtree decomposition
    XYmax = np.max(node, axis = 0)
    XYmin = np.min(node, axis = 0)

    theta = _minrectangle(node)
    node_r = _rotate(node, theta)

    edgexy = np.column_stack([node_r[edge[:, 0]], node_r[edge[:, 1]]])

    hmax, edgeh, fun, args = _gethdata(hdata)

    # test points along edges
    wm = 0.5 * (edgexy[:, :2] + edgexy[:, 2:])
    edge_len = msqrt(np.sum((edgexy[:, 2:] - edgexy[:, :2]) ** 2, axis = 1))
    L = 2.0 * dist2poly(wm, edgexy, 2.0 * edge_len)

    # add more points where edges are close
    r = 2.0 * edge_len / np.maximum(L, np.finfo(float).eps)
    r = mround((r - 2.0) / 2.0).astype(int)
    r = np.maximum(r, 0)
    add = np.where(r > 0)[0]

    if len(add) > 0:
        new_points = []
        new_L = []
        for j in add:
            ce = j
            num = r[ce]
            tmp = (np.arange(1, num + 1)) / (num + 1)

            x1, y1 = edgexy[ce, 0], edgexy[ce, 1]
            x2, y2 = edgexy[ce, 2], edgexy[ce, 3]
            xm, ym = wm[ce, 0], wm[ce, 1]

            pts1 = np.column_stack([x1 + tmp * (xm - x1), y1 + tmp * (ym - y1)])
            pts2 = np.column_stack([xm + tmp * (x2 - xm), ym + tmp * (y2 - ym)])

            total = pts1.shape[0] + pts2.shape[0]
            combined = np.empty((total, 2))
            combined[:pts1.shape[0]] = pts1
            combined[pts1.shape[0]:] = pts2

            new_points.append(combined)
            new_L.append(np.full(total, L[ce]))

        if new_points:
            new_pts = np.vstack(new_points)
            new_ls = np.hstack(new_L)

            total_wm = wm.shape[0] + new_pts.shape[0]
            wm_new = np.empty((total_wm, 2))
            wm_new[:wm.shape[0]] = wm
            wm_new[wm.shape[0]:] = new_pts
            wm = wm_new

            new_L_computed = dist2poly(new_pts, edgexy, new_ls)
            total_L = L.shape[0] + new_L_computed.shape[0]
            L_new = np.empty(total_L)
            L_new[:L.shape[0]] = L
            L_new[L.shape[0]:] = new_L_computed
            L = L_new

    # sort by y-value
    sort_idx = np.argsort(wm[:, 1])
    wm = wm[sort_idx]
    L = L[sort_idx]
    nw = wm.shape[0]

    # quadtree decomposition
    xymin = np.min(edgexy.reshape(-1, 2), axis = 0)
    xymax = np.max(edgexy.reshape(-1, 2), axis = 0)

    dim = 2.0 * max(xymax - xymin)
    xm = 0.5 * (xymin[0] + xymax[0])
    ym = 0.5 * (xymin[1] + xymax[1])

    # initial bounding box
    p_list = [
        np.array([xm - 0.5 * dim, ym - 0.5 * dim]),
        np.array([xm + 0.5 * dim, ym - 0.5 * dim]),
        np.array([xm + 0.5 * dim, ym + 0.5 * dim]),
        np.array([xm - 0.5 * dim, ym + 0.5 * dim])
    ]
    p_arr = np.array(p_list)
    b_list = [[0, 1, 2, 3]]

    # user defined size function at initial nodes
    pr = _rotate(p_arr, -theta)
    h_arr = _userhfun(pr[:, 0], pr[:, 1], fun, args, hmax, XYmin, XYmax).tolist()

    # iterative subdivision
    max_iter = 100
    for _iter in range(max_iter):
        new_boxes = []
        changed = False

        for m in range(len(b_list)):
            n1, n2, n3, n4 = b_list[m]
            x1 = p_arr[n1, 0]
            y1 = p_arr[n1, 1]
            x2 = p_arr[n2, 0]
            y4 = p_arr[n4, 1]

            # binary search for first wm with y >= y1
            if nw == 0 or wm[0, 1] >= y1:
                start = 0
            elif wm[nw - 1, 1] < y1:
                start = nw
            else:
                lower_b = 0
                upper_b = nw - 1
                start = 0
                for _bs in range(nw):
                    start = (lower_b + upper_b) // 2
                    if wm[start, 1] < y1:
                        lower_b = start + 1
                    elif start > 0 and wm[start - 1, 1] < y1:
                        break
                    else:
                        upper_b = max(start - 1, lower_b)
                        if lower_b > upper_b:
                            start = lower_b
                            break

            # min LFS in box
            LFS = 1.5 * min(h_arr[n1], h_arr[n2], h_arr[n3], h_arr[n4])

            for i in range(start, nw):
                if wm[i, 1] <= y4:
                    if wm[i, 0] >= x1 and wm[i, 0] <= x2 and L[i] < LFS:
                        LFS = L[i]
                else:
                    break

            # split box
            if (x2 - x1) >= LFS:
                changed = True
                xm_box = x1 + 0.5 * (x2 - x1)
                ym_box = y1 + 0.5 * (y4 - y1)

                np_start = len(p_arr)
                new_nodes = np.array([
                    [xm_box, ym_box],
                    [xm_box, y1],
                    [x2, ym_box],
                    [xm_box, y4],
                    [x1, ym_box]
                ])
                p_arr = np.vstack([p_arr, new_nodes])

                # user size function at new nodes
                pr_new = _rotate(new_nodes, -theta)
                h_new = _userhfun(pr_new[:, 0], pr_new[:, 1], fun, args, hmax, XYmin, XYmax)
                h_arr.extend(h_new.tolist())

                c = np_start  # center
                s = np_start + 1  # south
                e = np_start + 2  # east
                nn = np_start + 3  # north
                w = np_start + 4  # west

                b_list[m] = [n1, s, c, w]  # box 1
                new_boxes.append([s, n2, e, c])  # box 2
                new_boxes.append([c, e, n3, nn])  # box 3
                new_boxes.append([w, c, nn, n4])  # box 4

        b_list.extend(new_boxes)

        if not changed:
            break

    # remove duplicate nodes (MATLAB quadtree.m L279: unique(p,'rows') — no
    # rounding)
    _, unique_idx, remap = np.unique(p_arr, axis = 0, return_index = True, return_inverse = True)
    p_arr = p_arr[unique_idx]
    h_arr_np = np.array(h_arr)[unique_idx]

    b_arr = np.array(b_list)
    b_arr = remap[b_arr]

    # form size function based on edge lengths
    e_set = set()
    for box in b_arr:
        for i in range(4):
            j = (i + 1) % 4
            e_pair = (min(box[i], box[j]), max(box[i], box[j]))
            e_set.add(e_pair)

    edges = np.array(list(e_set))
    if len(edges) == 0:
        # degenerate case
        p_out = _rotate(p_arr, -theta)
        t_out = np.array([[0, 1, 2]])
        h_out = h_arr_np
        return p_out, t_out, h_out

    L_edges = msqrt(np.sum((p_arr[edges[:, 0]] - p_arr[edges[:, 1]]) ** 2, axis = 1))

    for k in range(len(edges)):
        lk = L_edges[k]
        if lk < h_arr_np[edges[k, 0]]:
            h_arr_np[edges[k, 0]] = lk
        if lk < h_arr_np[edges[k, 1]]:
            h_arr_np[edges[k, 1]] = lk

    h_arr_np = np.minimum(h_arr_np, hmax)

    # gradient limiting
    tol_grad = 1.0e-06
    for _git in range(1000):
        h_old = h_arr_np.copy()
        for k in range(len(edges)):
            n1e = edges[k, 0]
            n2e = edges[k, 1]
            lk = L_edges[k]
            if h_arr_np[n1e] > h_arr_np[n2e]:
                dh = (h_arr_np[n1e] - h_arr_np[n2e]) / lk
                if dh > dhmax:
                    h_arr_np[n1e] = h_arr_np[n2e] + dhmax * lk
            else:
                dh = (h_arr_np[n2e] - h_arr_np[n1e]) / lk
                if dh > dhmax:
                    h_arr_np[n2e] = h_arr_np[n1e] + dhmax * lk

        max_change = np.max(np.abs((h_arr_np - h_old) / np.maximum(h_arr_np, np.finfo(float).eps)))
        if max_change < tol_grad:
            break

    # triangulate quadtree (MATLAB quadtree.m lines 326-527)
    if len(b_arr) == 1:
        t_arr = np.array([[b_arr[0, 0], b_arr[0, 1], b_arr[0, 2]],
                           [b_arr[0, 0], b_arr[0, 2], b_arr[0, 3]]])
    else:
        # build n2n connectivity (max 8 neighbors per node)
        np_nodes = len(p_arr)
        n2n = np.zeros((np_nodes, 9), dtype = int)
        for k in range(len(edges)):
            n1e, n2e = edges[k, 0], edges[k, 1]
            n2n[n1e, 0] += 1
            n2n[n1e, n2n[n1e, 0]] = n2e
            n2n[n2e, 0] += 1
            n2n[n2e, n2n[n2e, 0]] = n1e

        # regular boxes: all corners have <= 4 connections
        num_conn = n2n[:, 0] <= 4
        reg = np.all(num_conn[b_arr], axis = 1)

        # MATLAB L357: t = [b(reg,[1,2,3]); b(reg,[1,3,4])]
        # Concatenate ALL first-triangles, then ALL second-triangles (not
        # interleaved). Triangle ordering matters for downstream meshpoly
        # because _mytsearch() walks triangles in this order.
        t_list = []
        reg_idx = np.where(reg)[0]
        for i in reg_idx:
            t_list.append([b_arr[i, 0], b_arr[i, 1], b_arr[i, 2]])
        for i in reg_idx:
            t_list.append([b_arr[i, 0], b_arr[i, 2], b_arr[i, 3]])

        irreg_idx = np.where(~reg)[0]
        nlist = np.zeros(512, dtype = int)

        for ki in range(len(irreg_idx)):
            k = irreg_idx[ki]
            bn1, bn2, bn3, bn4 = b_arr[k]

            nlist[0] = bn1
            count = 1
            nxt = 1

            while True:
                cn = nlist[nxt - 1]
                old = np.inf
                tmp = -1
                for j in range(n2n[cn, 0]):
                    nn = n2n[cn, j + 1]
                    dx = p_arr[nn, 0] - p_arr[cn, 0]
                    dy = p_arr[nn, 1] - p_arr[cn, 1]
                    if count == 1:
                        if dx > 0 and dx < old:
                            old = dx; tmp = nn
                    elif count == 2:
                        if dy > 0 and dy < old:
                            old = dy; tmp = nn
                    elif count == 3:
                        if dx < 0 and abs(dx) < old:
                            old = abs(dx); tmp = nn
                    else:
                        if dy < 0 and abs(dy) < old:
                            old = abs(dy); tmp = nn

                if tmp == bn1:
                    break
                if count < 4 and tmp == b_arr[k, count]:
                    count += 1
                nlist[nxt] = tmp
                nxt += 1

            nnode = nxt

            if nnode == 4:
                t_list.append([bn1, bn2, bn3])
                t_list.append([bn1, bn3, bn4])
            elif nnode == 5:
                # MATLAB quadtree.m L455-473. Find first index (2..5) where
                # nlist deviates from the box corner order, then rotate the
                # (n1,n2,n3,n4) labelling so the mid-side node sits between
                # n1 and n2. MATLAB j=2 ↔ Python jj=1 (no rotation), j=3 ↔
                # jj=2 (rotate once), etc.
                jj = 1
                while jj <= 3:
                    if nlist[jj] != b_arr[k, jj]:
                        break
                    jj += 1
                if jj == 1:
                    cn1, cn2, cn3, cn4 = bn1, bn2, bn3, bn4
                elif jj == 2:
                    cn1, cn2, cn3, cn4 = bn2, bn3, bn4, bn1
                elif jj == 3:
                    cn1, cn2, cn3, cn4 = bn3, bn4, bn1, bn2
                else:  # jj == 4 (all matched through index 3)
                    cn1, cn2, cn3, cn4 = bn4, bn1, bn2, bn3
                mid = nlist[jj]
                t_list.append([cn1, mid, cn4])
                t_list.append([mid, cn2, cn3])
                t_list.append([cn4, mid, cn3])
            else:
                new_idx = len(p_arr)
                xave, yave, have = 0.0, 0.0, 0.0
                for j in range(nnode - 1):
                    jjn = nlist[j]
                    t_list.append([jjn, new_idx, nlist[j + 1]])
                    xave += p_arr[jjn, 0]
                    yave += p_arr[jjn, 1]
                    have += h_arr_np[jjn]
                jjn = nlist[nnode - 1]
                t_list.append([jjn, new_idx, nlist[0]])
                xave += p_arr[jjn, 0]
                yave += p_arr[jjn, 1]
                have += h_arr_np[jjn]

                centroid = np.array([[xave / nnode, yave / nnode]])
                p_arr = np.vstack([p_arr, centroid])
                h_arr_np = np.append(h_arr_np, have / nnode)

        t_arr = np.array(t_list, dtype = int) if len(t_list) > 0 else np.empty((0, 3), dtype = int)

    # remove bad nodes
    good = h_arr_np > 0
    if not np.all(good):
        good_idx = np.where(good)[0]
        remap2 = np.full(len(p_arr), -1, dtype = int)
        remap2[good_idx] = np.arange(len(good_idx))
        p_arr = p_arr[good_idx]
        h_arr_np = h_arr_np[good_idx]

        valid_tri = np.all(np.isin(t_arr, good_idx), axis = 1)
        t_arr = remap2[t_arr[valid_tri]]

    # undo rotation
    p_arr = _rotate(p_arr, -theta)

    return p_arr, t_arr, h_arr_np


def _boundarynodes(ph: np.ndarray,
        th: np.ndarray,
        hh: np.ndarray,
        node: np.ndarray,
        edge: np.ndarray) -> np.ndarray:

    # MATLAB Mesh2d/meshfaces.m > boundarynodes
    p = node.copy()
    e = edge.copy()

    # MATLAB boundarynodes L201-204: for each triangle j, mark which p's are
    # inpolygon of it; `i(temp)=j` overwrites so i[k] is the LAST triangle
    # containing p[k]. This differs from _mytsearch (which returns the first
    # enclosing triangle) and matters at boundary corners where the point
    # lies on multiple triangle edges — the chosen triangle determines which
    # tinterp bary result (and its FP noise) we use.
    def _matlab_search(qp, ph_, th_):
        ii = np.zeros(qp.shape[0], dtype = int)
        for j in range(th_.shape[0]):
            tri_verts = ph_[th_[j, :]]
            inside, _ = inpoly(qp, tri_verts)
            ii[inside] = j
        return ii

    i = _matlab_search(p, ph, th)
    h = _tinterp(ph, th, hh, p, i)

    for _iter in range(100):
        dxy = p[e[:, 1]] - p[e[:, 0]]
        L = msqrt(np.sum(dxy ** 2, axis = 1))
        he = 0.5 * (h[e[:, 0]] + h[e[:, 1]])

        ratio = L / he
        split = ratio >= 1.5

        if not np.any(split):
            break

        n1 = e[split, 0]
        n2 = e[split, 1]
        pm = 0.5 * (p[n1] + p[n2])
        n3 = np.arange(pm.shape[0]) + p.shape[0]

        e_new = e.copy()
        e_new[split, 1] = n3
        n_split = np.sum(split)
        extra_edges = np.column_stack([n3, n2])
        total_e = e_new.shape[0] + extra_edges.shape[0]
        e_combined = np.empty((total_e, 2), dtype = int)
        e_combined[:e_new.shape[0]] = e_new
        e_combined[e_new.shape[0]:] = extra_edges
        e = e_combined

        total_p = p.shape[0] + pm.shape[0]
        p_combined = np.empty((total_p, 2))
        p_combined[:p.shape[0]] = p
        p_combined[p.shape[0]:] = pm
        p = p_combined

        # MATLAB meshfaces.m L232 `mytsearch(...)` with no initial guess →
        # full inpolygon-loop fallback (see mytsearch.m L70-73).
        i_new = _matlab_search(pm, ph, th)
        h_new = _tinterp(ph, th, hh, pm, i_new)
        total_h = h.shape[0] + h_new.shape[0]
        h_combined = np.empty(total_h)
        h_combined[:h.shape[0]] = h
        h_combined[h.shape[0]:] = h_new
        h = h_combined

    # spring-based boundary smoothing (MATLAB meshfaces.m boundarynodes)
    ne = e.shape[0]
    nnode_orig = node.shape[0]
    from scipy.sparse import csr_matrix as _csr
    rows_s = np.empty(2 * ne, dtype = int)
    cols_s = np.empty(2 * ne, dtype = int)
    vals_s = np.empty(2 * ne)
    rows_s[:ne] = e[:, 0]
    cols_s[:ne] = np.arange(ne)
    vals_s[:ne] = -1.0
    rows_s[ne:] = e[:, 1]
    cols_s[ne:] = np.arange(ne)
    vals_s[ne:] = 1.0
    S = _csr((vals_s, (rows_s, cols_s)), shape = (p.shape[0], ne))

    dxy = p[e[:, 1]] - p[e[:, 0]]
    L = msqrt(np.sum(dxy ** 2, axis = 1))
    he = 0.5 * (h[e[:, 0]] + h[e[:, 1]])

    delta = 0.0
    # MATLAB meshfaces.boundarynodes L251 `i = zeros(size(p,1),1)` — 0 is
    # "no guess" in 1-indexed MATLAB. Use -1 for 0-indexed Python.
    i_search = np.full(p.shape[0], -1, dtype = int)
    for _iter in range(50):
        delta_old = delta

        F_factor = he / L - 1.0
        Fxy = dxy * F_factor[:, np.newaxis]
        Fp = S.dot(Fxy)
        Fp[:nnode_orig] = 0.0
        p = p + 0.2 * Fp

        dxy = p[e[:, 1]] - p[e[:, 0]]
        Lnew = msqrt(np.sum(dxy ** 2, axis = 1))
        delta = np.max(np.abs((Lnew - L) / Lnew))
        if delta < 0.02:
            break
        L = Lnew

        # MATLAB: re-interpolate size function if diverging
        if delta > delta_old:
            i_search = _mytsearch(ph[:, 0], ph[:, 1], th, p[:, 0], p[:, 1], i_search)
            h = _tinterp(ph, th, hh, p, i_search)
            he = 0.5 * (h[e[:, 0]] + h[e[:, 1]])

    return p


def _cdt(p: np.ndarray,
        node: np.ndarray,
        edge: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

    # constrained Delaunay triangulation (approximate)
    t = _mydelaunayn(p)

    # only keep triangles with internal centroids
    centroids = _tricentre(t, p)
    inside, _ = inpoly(centroids, node, edge)
    t = t[inside]

    return p, t


def meshpoly(node: np.ndarray,
        edge: np.ndarray,
        qtree: Dict[str, np.ndarray],
        p: np.ndarray,
        options: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:

    # MATLAB Mesh2d/meshpoly.m - core meshing routine
    shortedge = 0.75
    longedge = 1.5
    smalltri = 0.25
    largetri = 4.0
    qlimit = 0.5
    dt = 0.2

    # initialize mesh
    enum = findedge(p, node, edge, 1.0e-08)
    p = p[enum > 0]
    fix = np.arange(p.shape[0])

    # add internal nodes from quadtree
    inside, on_bnd = inpoly(qtree['p'], node, edge)
    internal = inside & ~on_bnd

    total_p = p.shape[0] + np.sum(internal)
    p_combined = np.empty((total_p, 2))
    p_combined[:p.shape[0]] = p
    p_combined[p.shape[0]:] = qtree['p'][internal]
    p = p_combined

    # MATLAB meshpoly.m L70: `tndx = zeros(...,1)` where 0 is MATLAB's
    # "no guess" sentinel. In Python 0-indexed, 0 is a real triangle;
    # use -1 instead so _mytsearch's `valid_guess = (i_guess >= 0)` skips
    # the initial-guess path and falls back to the inpolygon-loop (which
    # matches MATLAB's `i>0` check that always fails at initialisation).
    tndx = np.full(p.shape[0], -1, dtype = int)

    for iteration in range(options.get('maxit', 20)):
        # ensure unique node list (MATLAB: exact unique, no rounding)
        _, unique_idx, inverse_idx = np.unique(p, axis = 0, return_index = True, return_inverse = True)
        p = p[unique_idx]
        # MATLAB: fix = j(fix) where j is the inverse index — direct mapping
        # MATLAB meshpoly.m:79: fix = j(fix) (no bounds guard)
        # MATLAB meshpoly.m:80: tndx = tndx(i) (no fallback)
        fix = inverse_idx[fix]
        tndx = tndx[unique_idx]

        # constrained Delaunay
        p, t = _cdt(p, node, edge)

        if t.shape[0] == 0:
            break

        e = _getedges(t, p.shape[0])
        nume = e.shape[0]

        # sparse connectivity
        from scipy.sparse import csr_matrix as csr
        rows = np.empty(2 * nume, dtype = int)
        cols = np.empty(2 * nume, dtype = int)
        vals = np.empty(2 * nume)
        rows[:nume] = e[:, 0]
        cols[:nume] = np.arange(nume)
        vals[:nume] = 1.0
        rows[nume:] = e[:, 1]
        cols[nume:] = np.arange(nume)
        vals[nume:] = -1.0
        S = csr((vals, (rows, cols)), shape = (p.shape[0], nume))

        # size function interpolation
        tndx_full = _mytsearch(qtree['p'][:, 0], qtree['p'][:, 1], qtree['t'], p[:, 0], p[:, 1], tndx if len(tndx) == p.shape[0] else None)
        hn = _tinterp(qtree['p'], qtree['t'], qtree['h'], p, tndx_full)
        h = 0.5 * (hn[e[:, 0]] + hn[e[:, 1]])

        edgev = p[e[:, 0]] - p[e[:, 1]]
        L = np.maximum(msqrt(np.sum(edgev ** 2, axis = 1)), np.finfo(float).eps)

        # inner smoothing
        # MATLAB meshpoly.m:110 uses `for subiter = 1:(iter-1)`, i.e. skip on iteration=0
        done = False
        for subiter in range(iteration):
            L0_target = h * msqrt(np.sum(L ** 2) / np.sum(h ** 2))
            F = np.maximum(L0_target / L - 1.0, -0.1)
            Fxy = edgev * F[:, np.newaxis]
            Fp = S.dot(Fxy)

            Fp[fix] = 0.0
            p = p + dt * Fp

            edgev = p[e[:, 0]] - p[e[:, 1]]
            L0_new = np.maximum(msqrt(np.sum(edgev ** 2, axis = 1)), np.finfo(float).eps)
            move = np.max(np.abs((L0_new - L) / L))
            L = L0_new

            mlim = options.get('mlim', 0.02)
            if move < mlim:
                done = True
                break

        # re-triangulate after smoothing
        p, t = _cdt(p, node, edge)
        if t.shape[0] == 0:
            break

        e = _getedges(t, p.shape[0])
        edgev = p[e[:, 0]] - p[e[:, 1]]
        L = np.maximum(msqrt(np.sum(edgev ** 2, axis = 1)), np.finfo(float).eps)

        tndx_full = _mytsearch(qtree['p'][:, 0], qtree['p'][:, 1], qtree['t'], p[:, 0], p[:, 1])
        hn = _tinterp(qtree['p'], qtree['t'], qtree['h'], p, tndx_full)
        h = 0.5 * (hn[e[:, 0]] + hn[e[:, 1]])
        tndx = tndx_full

        r = L / np.maximum(h, np.finfo(float).eps)
        if done and np.max(r) < 3.0:
            break

        # nodal density control
        if iteration < options.get('maxit', 20) - 1:
            Ah = 0.5 * _tricentre(t, hn) ** 2
            t_area = np.abs(triarea(p, t))

            # MATLAB meshpoly.m:173-175
            small_tri = np.where(t_area < smalltri * Ah)[0]
            short_edges = np.where(r < shortedge)[0]
            # MATLAB L174: k = find(sum(abs(S),2)<2)  -- nodes with <2 edges
            S_abs_sum = np.asarray(np.abs(S).sum(axis=1)).ravel()
            low_conn = np.where(S_abs_sum < 2)[0]

            # MATLAB meshpoly.m:176-190 prob array construction
            # Order matches MATLAB L178-181: j(edges), i(triangles), k(low-conn)
            prob = np.zeros(p.shape[0], dtype = bool)
            if len(short_edges) > 0:
                prob[e[short_edges].ravel()] = True
            if len(small_tri) > 0:
                prob[t[small_tri].ravel()] = True
            if len(low_conn) > 0:
                prob[low_conn] = True
            prob[fix] = False

            pnew = p[~prob]
            tndx_new = tndx[~prob]

            # re-index fix (MATLAB meshpoly.m:184-187: j(~prob)=1; j=cumsum(j); fix=j(fix))
            remap_arr = np.zeros(p.shape[0], dtype = int)
            remap_arr[~prob] = 1
            remap_arr = np.cumsum(remap_arr) - 1
            fix = remap_arr[fix]

            # add new nodes at circumcentres of large/low-quality triangles
            # MATLAB meshpoly.m:192-204
            large_tri = t_area > largetri * Ah
            r_tri = _longest(p, t) / np.maximum(_tricentre(t, hn), np.finfo(float).eps)
            q = quality(p, t)
            low_quality = (r_tri > longedge) & (q < qlimit)

            if np.any(large_tri | low_quality):
                # MATLAB:197-198 k = find(k & ~i); i = find(i);
                i_large = np.where(large_tri)[0]
                k_lq_only = np.where(low_quality & ~large_tri)[0]
                # MATLAB:201 cc = circumcircle(p, [t(i,:); t(k,:)])
                tri_idx = np.concatenate([i_large, k_lq_only])
                cc = circumcircle(p, t[tri_idx])
                cc_points = cc[:, :2]
                cc_radii = cc[:, 2]

                # MATLAB:204 ok = [true(size(i)); false(size(k))]
                ok = np.zeros(len(cc_points), dtype = bool)
                ok[:len(i_large)] = True

                # MATLAB:205-223 skip new centres inside an already-accepted
                # circle. Note: `cc_radii` IS radius^2 (column 3 of
                # circumcircle, MATLAB `cc(:,3) = sum((p1-cc).^2)`).
                # MATLAB L215 `dx<cc(kk,3) && (dx+(y-cc(kk,2))^2)<cc(kk,3)`
                # compares squared-distances to squared-radius — NOT r^4.
                for ii in range(len(i_large), len(cc_points)):
                    x = cc_points[ii, 0]
                    y = cc_points[ii, 1]
                    is_inside = False
                    # MATLAB:211 j = find(ok) - recompute each iteration (accept set grows)
                    for kk in np.where(ok)[0]:
                        dx2 = (x - cc_points[kk, 0]) ** 2
                        r2 = cc_radii[kk]
                        if dx2 < r2:
                            if dx2 + (y - cc_points[kk, 1]) ** 2 < r2:
                                is_inside = True
                                break
                    if not is_inside:
                        ok[ii] = True

                # MATLAB:224 cc = cc(ok,:)
                cc_points = cc_points[ok]
                # MATLAB:225 cc = cc(inpoly(cc(:,1:2),node,edge),:) — inpoly AFTER dedup
                if len(cc_points) > 0:
                    inside_cc, _ = inpoly(cc_points, node, edge)
                    cc_points = cc_points[inside_cc]

                if len(cc_points) > 0:
                    total_new = pnew.shape[0] + cc_points.shape[0]
                    p_combined2 = np.empty((total_new, 2))
                    p_combined2[:pnew.shape[0]] = pnew
                    p_combined2[pnew.shape[0]:] = cc_points
                    pnew = p_combined2

                    total_tndx = tndx_new.shape[0] + cc_points.shape[0]
                    tndx_combined = np.empty(total_tndx, dtype = int)
                    tndx_combined[:tndx_new.shape[0]] = tndx_new
                    tndx_combined[tndx_new.shape[0]:] = 0
                    tndx_new = tndx_combined

            p = pnew
            tndx = tndx_new

    return p, t


def _checkgeometry(node: np.ndarray,
        edge: Optional[np.ndarray],
        face: Optional[List[np.ndarray]],
        hdata: Optional[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], Optional[Dict[str, Any]]]:

    nnode = node.shape[0]
    if edge is None:
        idx = np.arange(nnode)
        edge = np.empty((nnode, 2), dtype = int)
        edge[:nnode - 1, 0] = idx[:nnode - 1]
        edge[:nnode - 1, 1] = idx[1:nnode]
        edge[nnode - 1] = [nnode - 1, 0]

    if face is None:
        face = [np.arange(edge.shape[0])]

    # remove duplicate nodes (MATLAB checkgeometry.m L85: unique(node,'rows')
    # — no rounding)
    _, unique_idx, remap = np.unique(node, axis = 0, return_index = True, return_inverse = True)
    if len(unique_idx) < nnode:
        node = node[unique_idx]
        edge = remap[edge]

    # remove duplicate edges; remap `face` so its edge indices still refer
    # to the correct rows of the reordered `edge` array.
    nedge_old = edge.shape[0]
    e_sorted = np.sort(edge, axis = 1)
    _, unique_e_idx = np.unique(e_sorted, axis = 0, return_index = True)
    # `unique_e_idx` contains indices into the OLD edge array that survive.
    # Build a map old_idx -> new_idx (or -1 if dropped as duplicate).
    old_to_new = np.full(nedge_old, -1, dtype = int)
    # reorder unique_e_idx so that edges keep their original order as much as
    # possible — MATLAB's `[i,i,j] = unique(sort(edge,2),'rows')` returns
    # sorted indices; we preserve original insertion order to keep `face`
    # indices stable. Sort the surviving indices by their original position.
    unique_e_idx = np.sort(unique_e_idx)
    for new_i, old_i in enumerate(unique_e_idx):
        old_to_new[old_i] = new_i
    edge = edge[unique_e_idx]

    remapped_face: List[np.ndarray] = []
    for f in face:
        mapped = old_to_new[np.asarray(f, dtype = int)]
        mapped = mapped[mapped >= 0]
        remapped_face.append(mapped)
    face = remapped_face

    return node, edge, face, hdata


def _getoptions(options: Optional[Dict[str, Any]]) -> Dict[str, Any]:

    defaults = {
        'mlim': 0.02,
        'maxit': 20,
        'dhmax': 0.3,
        'output': False,
        'debug': False,
    }

    if options is None:
        return defaults

    for key in defaults:
        if key not in options:
            options[key] = defaults[key]

    if 'debug' not in options:
        options['debug'] = False

    return options


def _detect_loops(edge: np.ndarray, nnode: int) -> List[np.ndarray]:
    # Walk the edge adjacency graph to split EDGE into connected components
    # (assumed to be closed loops). Returns a list of edge-index arrays.
    nedge = edge.shape[0]
    used = np.zeros(nedge, dtype = bool)

    node_to_edges: Dict[int, List[int]] = {}
    for ei in range(nedge):
        a, b = int(edge[ei, 0]), int(edge[ei, 1])
        node_to_edges.setdefault(a, []).append(ei)
        node_to_edges.setdefault(b, []).append(ei)

    loops: List[List[int]] = []
    for start in range(nedge):
        if used[start]:
            continue
        loop = [start]
        used[start] = True
        start_node = int(edge[start, 0])
        cur_node = int(edge[start, 1])
        while cur_node != start_node:
            next_ei = -1
            nxt = -1
            for ei in node_to_edges.get(cur_node, []):
                if used[ei]:
                    continue
                a, b = int(edge[ei, 0]), int(edge[ei, 1])
                if a == cur_node:
                    next_ei = ei
                    nxt = b
                    break
                if b == cur_node:
                    next_ei = ei
                    nxt = a
                    break
            if next_ei < 0:
                break
            used[next_ei] = True
            loop.append(next_ei)
            cur_node = nxt
        loops.append(np.array(loop, dtype = int))

    return loops


def _loop_vertices(edge_indices: np.ndarray,
        edge: np.ndarray,
        node: np.ndarray) -> np.ndarray:
    # Walk edges in a loop and return the ordered vertex positions.
    if len(edge_indices) == 0:
        return np.empty((0, 2))
    a0 = int(edge[edge_indices[0], 0])
    b0 = int(edge[edge_indices[0], 1])
    pts = [node[a0], node[b0]]
    prev = b0
    for k in range(1, len(edge_indices)):
        ei = int(edge_indices[k])
        a, b = int(edge[ei, 0]), int(edge[ei, 1])
        if a == prev:
            pts.append(node[b])
            prev = b
        elif b == prev:
            pts.append(node[a])
            prev = a
        else:
            pts.append(node[a])
            prev = b
    arr = np.asarray(pts)
    if arr.shape[0] > 1 and np.allclose(arr[0], arr[-1]):
        arr = arr[:-1]
    return arr


def _loop_signed_area(edge_indices: np.ndarray,
        edge: np.ndarray,
        node: np.ndarray) -> float:
    arr = _loop_vertices(edge_indices, edge, node)
    if arr.shape[0] < 3:
        return 0.0
    x = arr[:, 0]
    y = arr[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _classify_faces(face_chk: List[np.ndarray],
        edge: np.ndarray,
        node: np.ndarray) -> Tuple[List[int], List[int], List[np.ndarray]]:
    # Split face_chk into (outer_idx, hole_idx) via containment.
    # Use the polygon CENTROID as probe (not the first vertex) — a hole's
    # vertex can lie on the outer boundary, but its centroid is always
    # strictly inside the outer loop.
    face_verts = [_loop_vertices(f, edge, node) for f in face_chk]
    is_hole = [False] * len(face_chk)
    for i, vi in enumerate(face_verts):
        if vi.shape[0] < 3:
            continue
        # centroid: robust inside-probe
        probe = vi.mean(axis = 0, keepdims = True)
        for j, vj in enumerate(face_verts):
            if i == j or vj.shape[0] < 3:
                continue
            inside_j, _ = inpoly(probe, vj)
            if bool(inside_j[0]):
                is_hole[i] = True
                break
    outer_idx = [k for k, h in enumerate(is_hole) if not h]
    hole_idx = [k for k, h in enumerate(is_hole) if h]
    # fall back to "face[0] is outer" when classification is degenerate
    if not outer_idx or (len(face_chk) > 1 and not hole_idx and all(is_hole)):
        outer_idx = [0]
        hole_idx = list(range(1, len(face_chk)))
    return outer_idx, hole_idx, face_verts


def mesh2d(node: np.ndarray,
        edge: Optional[np.ndarray] = None,
        hdata: Optional[Dict[str, Any]] = None,
        options: Optional[Dict[str, Any]] = None,
        face: Optional[List[np.ndarray]] = None) -> Tuple[np.ndarray, np.ndarray]:

    # MATLAB Mesh2d/mesh2d.m -> meshfaces -> meshpoly
    #
    # When `edge` contains multiple closed loops (e.g. an outer rectangle
    # with an inner triangular hole) the caller can pass `face` as
    # [outer_edges, hole_edges, ...] or let mesh2d auto-detect topology by
    # walking the edge adjacency graph and checking containment.
    node = np.asarray(node, dtype = float)
    if edge is not None:
        edge = np.asarray(edge, dtype = int)

    opts = _getoptions(options)

    if face is not None:
        face_in = [np.asarray(f, dtype = int) for f in face]
    else:
        face_in = None

    node, edge, face_chk, hdata = _checkgeometry(node, edge, face_in, hdata)

    # auto-detect topology when caller did not pass `face`
    auto_detected = False
    if face_in is None and edge is not None and edge.shape[0] > 0:
        loops = _detect_loops(edge, node.shape[0])
        if len(loops) > 1:
            # only split into multiple faces if at least one loop is a hole
            loop_verts_auto = [_loop_vertices(lp, edge, node) for lp in loops]
            any_hole = False
            for i, vi in enumerate(loop_verts_auto):
                if vi.shape[0] == 0:
                    continue
                probe = vi[0:1]
                for j, vj in enumerate(loop_verts_auto):
                    if i == j or vj.shape[0] < 3:
                        continue
                    inside_j, _ = inpoly(probe, vj)
                    if bool(inside_j[0]):
                        any_hole = True
                        break
                if any_hole:
                    break
            if any_hole:
                face_chk = loops
                auto_detected = True

    # quadtree decomposition
    qt_p, qt_t, qt_h = quadtree(node, edge, hdata, opts['dhmax'], opts.get('output', False))
    qt = {'p': qt_p, 't': qt_t, 'h': qt_h}

    # boundary nodes
    pbnd = _boundarynodes(qt_p, qt_t, qt_h, node, edge)

    p_all = np.empty((0, 2))
    t_all = np.empty((0, 3), dtype = int)

    # classify faces into outer / hole (either from caller or auto-detected)
    if len(face_chk) > 1 and (face_in is not None or auto_detected):
        outer_idx, hole_idx, _ = _classify_faces(face_chk, edge, node)

        hole_edges_all: List[int] = []
        for h in hole_idx:
            hole_edges_all.extend(face_chk[h].tolist())
        hole_edges_arr = np.asarray(hole_edges_all, dtype = int)

        # vertex arrays for each hole (used for explicit triangle filtering)
        hole_vert_sets = [_loop_vertices(face_chk[h], edge, node) for h in hole_idx]

        for k in outer_idx:
            if len(hole_edges_arr) > 0:
                combined = np.concatenate([face_chk[k], hole_edges_arr])
            else:
                combined = face_chk[k]
            face_edges = edge[combined]
            pnew, tnew = meshpoly(node, face_edges, qt, pbnd, opts)

            # explicit hole filter: drop triangles whose centroid lies inside
            # any hole loop. This is a belt-and-braces guard for degenerate
            # triangles that slip through meshpoly's internal inpoly() check.
            if tnew.shape[0] > 0 and len(hole_vert_sets) > 0:
                centroids = np.mean(pnew[tnew], axis = 1)
                keep = np.ones(tnew.shape[0], dtype = bool)
                for hv in hole_vert_sets:
                    if hv.shape[0] < 3:
                        continue
                    inside_h, on_h = inpoly(centroids, hv)
                    keep &= ~(inside_h & ~on_h)
                tnew = tnew[keep]

            if tnew.shape[0] > 0:
                tnew_shifted = tnew + p_all.shape[0]
                total_t = t_all.shape[0] + tnew_shifted.shape[0]
                t_combined = np.empty((total_t, 3), dtype = int)
                t_combined[:t_all.shape[0]] = t_all
                t_combined[t_all.shape[0]:] = tnew_shifted
                t_all = t_combined

                total_p = p_all.shape[0] + pnew.shape[0]
                p_combined = np.empty((total_p, 2))
                p_combined[:p_all.shape[0]] = p_all
                p_combined[p_all.shape[0]:] = pnew
                p_all = p_combined
    else:
        for k in range(len(face_chk)):
            face_edges = edge[face_chk[k]]
            pnew, tnew = meshpoly(node, face_edges, qt, pbnd, opts)

            if tnew.shape[0] > 0:
                tnew_shifted = tnew + p_all.shape[0]
                total_t = t_all.shape[0] + tnew_shifted.shape[0]
                t_combined = np.empty((total_t, 3), dtype = int)
                t_combined[:t_all.shape[0]] = t_all
                t_combined[t_all.shape[0]:] = tnew_shifted
                t_all = t_combined

                total_p = p_all.shape[0] + pnew.shape[0]
                p_combined = np.empty((total_p, 2))
                p_combined[:p_all.shape[0]] = p_all
                p_combined[p_all.shape[0]:] = pnew
                p_all = p_combined

    # fix mesh
    if p_all.shape[0] > 0 and t_all.shape[0] > 0:
        p_all, t_all, _, _ = fixmesh(p_all, t_all)

    return p_all, t_all


def meshfaces(node: np.ndarray,
        edge: np.ndarray,
        face: List[np.ndarray],
        hdata: Optional[Dict[str, Any]] = None,
        options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    2D unstructured mesh generation for multi-face polygonal geometry.

    Generates a 2D unstructured triangular mesh for a geometry with
    multiple polygonal faces. Each face can contain an arbitrary number
    of cavities. This is the multi-face counterpart to mesh2d().

    MATLAB: meshfaces.m

    Parameters
    ----------
    node : ndarray, shape (N, 2)
        XY coordinates of geometry vertices for all faces.
    edge : ndarray, shape (M, 2)
        Connectivity between nodes. Each row [n1, n2] defines an edge.
    face : list of ndarray
        Each element face[k] is an array of edge indices (0-based) that
        define the k-th polygonal face.
    hdata : dict, optional
        Element size information:
        - hdata['hmax']: maximum global element size
        - hdata['edgeh']: element sizes on specific edges, shape (K, 2)
        - hdata['fun']: user-defined size function fun(x, y, *args)
        - hdata['args']: additional arguments for hdata['fun']
    options : dict, optional
        Solver tuning parameters:
        - options['mlim']: convergence tolerance (default 0.02)
        - options['maxit']: maximum iterations (default 20)
        - options['dhmax']: max relative gradient in size function (default 0.3)
        - options['output']: display output (default False)

    Returns
    -------
    p : ndarray, shape (Np, 2)
        Nodal XY coordinates.
    t : ndarray, shape (Nt, 3)
        Triangles as indices into p (CCW ordered).
    fnum : ndarray, shape (Nt,)
        Face number (1-based) for each triangle in t.

    Examples
    --------
    >>> # Two adjacent squares sharing an edge
    >>> node = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [2, 0], [2, 1]])
    >>> edge = np.array([[0, 1], [1, 2], [2, 3], [3, 0],
    ...                  [1, 4], [4, 5], [5, 2]])
    >>> face = [np.array([0, 1, 2, 3]), np.array([4, 5, 6, 1])]
    >>> hdata = {'hmax': 0.2}
    >>> p, t, fnum = meshfaces(node, edge, face, hdata)
    """
    node = np.asarray(node, dtype = float)
    edge = np.asarray(edge, dtype = int)

    face_arrays = []
    for f in face:
        face_arrays.append(np.asarray(f, dtype = int))

    opts = _getoptions(options)
    node, edge, face_arrays, hdata = _checkgeometry(node, edge, face_arrays, hdata)

    # quadtree decomposition
    qt_p, qt_t, qt_h = quadtree(node, edge, hdata, opts['dhmax'], opts.get('output', False))
    qt = {'p': qt_p, 't': qt_t, 'h': qt_h}

    # boundary nodes
    pbnd = _boundarynodes(qt_p, qt_t, qt_h, node, edge)

    # mesh each face separately
    p_all = np.empty((0, 2))
    t_all = np.empty((0, 3), dtype = int)
    fnum_all = np.empty(0, dtype = int)

    for k in range(len(face_arrays)):
        face_edges = edge[face_arrays[k]]
        pnew, tnew = meshpoly(node, face_edges, qt, pbnd, opts)

        if tnew.shape[0] > 0:
            tnew_shifted = tnew + p_all.shape[0]

            total_t = t_all.shape[0] + tnew_shifted.shape[0]
            t_combined = np.empty((total_t, 3), dtype = int)
            t_combined[:t_all.shape[0]] = t_all
            t_combined[t_all.shape[0]:] = tnew_shifted
            t_all = t_combined

            total_p = p_all.shape[0] + pnew.shape[0]
            p_combined = np.empty((total_p, 2))
            p_combined[:p_all.shape[0]] = p_all
            p_combined[p_all.shape[0]:] = pnew
            p_all = p_combined

            # face number (1-based, matching MATLAB)
            fnum_new = np.full(tnew.shape[0], k + 1, dtype = int)
            fnum_all = np.concatenate([fnum_all, fnum_new])

    # fix mesh (preserve face numbers)
    if p_all.shape[0] > 0 and t_all.shape[0] > 0:
        p_all, t_all, _, fnum_all = fixmesh(p_all, t_all, tfun = fnum_all)

    return p_all, t_all, fnum_all


def mesh_collection(num: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Collection of meshing examples.

    Provides predefined geometries for testing and demonstration of the
    mesh generation capabilities.

    MATLAB: mesh_collection.m

    Parameters
    ----------
    num : int
        Example number (1-5):
        1 - Simple rotated rectangle (driven cavity)
        2 - Rectangle with circular hole
        3 - L-shaped domain
        4 - U-shaped domain
        5 - Simple square

    Returns
    -------
    p : ndarray, shape (N, 2)
        Nodal XY coordinates.
    t : ndarray, shape (M, 3)
        Triangle connectivity.

    Examples
    --------
    >>> p, t = mesh_collection(1)
    >>> print(p.shape, t.shape)
    """
    if num == 1:
        # Simple rotated rectangle (driven cavity)
        node = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 1.0], [0.0, 1.0]])
        # rotate by 45 degrees
        theta = np.radians(45)
        c, s = mcos(theta), msin(theta)
        rot = np.array([[c, s], [-s, c]])
        node = node @ rot
        hdata = {'hmax': 0.5}
        p, t = mesh2d(node, None, hdata)

    elif num == 2:
        # Rectangle with circular hole
        theta = mlinspace(0, 2 * np.pi, 101)[:-1]
        x = mcos(theta) / 2
        y = msin(theta) / 2

        n_circ = len(theta)
        rect_nodes = np.array([[-5.0, -5.0], [5.0, -5.0], [5.0, 15.0], [-5.0, 15.0]])
        node = np.vstack([np.column_stack([x, y]), rect_nodes])

        # circle edges
        circ_edges = np.column_stack([np.arange(n_circ), np.roll(np.arange(n_circ), -1)])
        # rectangle edges
        n = n_circ
        rect_edges = np.array([[n, n + 1], [n + 1, n + 2], [n + 2, n + 3], [n + 3, n]])

        edge = np.vstack([circ_edges, rect_edges])
        hdata = {'hmax': 0.5}
        p, t = mesh2d(node, edge, hdata)

    elif num == 3:
        # L-shaped domain
        node = np.array([
            [0.0, 0.0], [1.0, 0.0], [1.0, 0.5],
            [0.5, 0.5], [0.5, 1.0], [0.0, 1.0],
        ])
        hdata = {'hmax': 0.05}
        p, t = mesh2d(node, None, hdata)

    elif num == 4:
        # U-shaped domain
        node = np.array([
            [0.0, 0.0], [3.0, 0.0], [3.0, 3.0], [2.5, 3.0],
            [2.5, 0.5], [0.5, 0.5], [0.5, 3.0], [0.0, 3.0],
        ])
        hdata = {'hmax': 0.1}
        p, t = mesh2d(node, None, hdata)

    elif num == 5:
        # Simple unit square
        node = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        hdata = {'hmax': 0.1}
        p, t = mesh2d(node, None, hdata)

    else:
        raise ValueError(
            "Example number {} not available. Choose 1-5.".format(num)
        )

    return p, t
