"""
Coverlayer module: boundary shift and Green function refinement.

MATLAB reference: Greenfun/+coverlayer/
  shift.m, refine.m, refineret.m, refinestat.m

This module implements:
  1. shift(p1, d, **options): shift particle vertices along outward normals to
     create a thin cover-layer boundary (used for nonlocal effective-layer
     models like Luo et al., PRL 111, 093901 (2013)).
  2. refine(p, ind): return a refinement callable that the BEM solver can use
     when initializing the Green function to recompute entries for neighbour
     cover-layer elements via polar integration.
  3. refineret(obj, g, f, p, ind), refinestat(obj, g, f, p, ind): the actual
     polar-integration refinement of Green function matrices g, f.
"""

import math

from typing import Any, Callable, List, Tuple

import numpy as np


def shift(p1: Any,
        d: Any,
        **options: Any) -> Any:
    """
    Shift boundary for creation of cover layer structure.

    MATLAB: p2 = coverlayer.shift(p1, d, op, PropertyPairs)

    Parameters
    ----------
    p1 : Particle
        Source particle whose vertices will be shifted along outward normals.
    d : float or array_like
        Shift distance. Scalar: uniform shift. Array (nverts,): per-vertex.
    **options : dict
        nvec : ndarray, optional
            Precomputed vertex-space normal vectors (nverts x 3). If absent,
            face-normals are interpolated to vertices via interp_values.
        Any remaining options are forwarded to the new Particle constructor.

    Returns
    -------
    p2 : Particle
        Particle with shifted vertices and identical face topology.
    """
    from ..geometry.particle import Particle

    d = np.asarray(d, dtype = float)
    if d.size == 1:
        d = np.full(p1.nverts, float(d))

    nvec = options.pop('nvec', None)

    if p1.verts2 is None:
        if nvec is None:
            nvec, _ = p1.interp_values(p1.nvec)

        verts_round = np.round(p1.verts, 4)
        _, i1, i2 = _unique_rows(verts_round)

        nvec_verts = nvec[i1[i2], :]
        new_verts = p1.verts + nvec_verts * d[:, np.newaxis]

        return Particle(new_verts, p1.faces, interp = p1.interp, **options)

    else:
        d2 = _interp2(p1, d.reshape(-1, 1)).ravel()
        if nvec is None:
            nvec, _ = p1.interp_values(p1.nvec)
        if nvec.shape[0] != p1.verts2.shape[0]:
            nvec = _interp2(p1, nvec)

        new_verts2 = p1.verts2 + nvec * d2[:, np.newaxis]
        return Particle(new_verts2, p1.faces2, interp = p1.interp, **options)


def _unique_rows(a: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    MATLAB-compatible `[~, i1, i2] = unique(a, 'rows')`.

    Returns
    -------
    u : ndarray
        Unique rows (sorted).
    i1 : ndarray
        Indices such that a[i1] == u.
    i2 : ndarray
        Indices such that u[i2] == a.
    """
    u, i1, i2 = np.unique(a, axis = 0, return_index = True, return_inverse = True)
    return u, i1, i2


def _interp2(p: Any, v: np.ndarray) -> np.ndarray:
    """
    Interpolate vertex-space values from p.verts to p.verts2 (curved boundary).

    MATLAB: Greenfun/+coverlayer/shift.m/interp2
    """
    ind3, ind4 = p.index34()

    n_verts2 = p.verts2.shape[0]
    if v.ndim == 1:
        v = v.reshape(-1, 1)
        squeeze_out = True
    else:
        squeeze_out = False
    v2 = np.zeros((n_verts2, v.shape[1]))

    if len(ind3) > 0:
        # Triangular rows: only columns {0..2, 4..6} of faces2 / {0..2} of
        # faces are valid. Pull those out before casting to int so we do not
        # trip over NaN fillers in the quad columns.
        f2_tri = p.faces2[ind3][:, [0, 1, 2, 4, 5, 6]].astype(int)
        f_tri = p.faces[ind3][:, [0, 1, 2]].astype(int)

        i1, i2, i3 = f2_tri[:, 0], f2_tri[:, 1], f2_tri[:, 2]
        i4, i5, i6 = f2_tri[:, 3], f2_tri[:, 4], f2_tri[:, 5]
        i10, i20, i30 = f_tri[:, 0], f_tri[:, 1], f_tri[:, 2]

        v2[i1, :] = v[i10, :]
        v2[i2, :] = v[i20, :]
        v2[i3, :] = v[i30, :]
        v2[i4, :] = 0.5 * (v[i10, :] + v[i20, :])
        v2[i5, :] = 0.5 * (v[i20, :] + v[i30, :])
        v2[i6, :] = 0.5 * (v[i30, :] + v[i10, :])

    if len(ind4) > 0:
        f2 = p.faces2[ind4].astype(int)
        f = p.faces[ind4].astype(int)

        i1, i2, i3, i4 = f2[:, 0], f2[:, 1], f2[:, 2], f2[:, 3]
        i5, i6, i7, i8, i9 = f2[:, 4], f2[:, 5], f2[:, 6], f2[:, 7], f2[:, 8]
        i10, i20, i30, i40 = f[:, 0], f[:, 1], f[:, 2], f[:, 3]

        v2[i1, :] = v[i10, :]
        v2[i2, :] = v[i20, :]
        v2[i3, :] = v[i30, :]
        v2[i4, :] = v[i40, :]
        v2[i5, :] = 0.5 * (v[i10, :] + v[i20, :])
        v2[i6, :] = 0.5 * (v[i20, :] + v[i30, :])
        v2[i7, :] = 0.5 * (v[i30, :] + v[i40, :])
        v2[i8, :] = 0.5 * (v[i40, :] + v[i10, :])
        v2[i9, :] = 0.25 * (v[i10, :] + v[i20, :] + v[i30, :] + v[i40, :])

    verts2_round = np.round(p.verts2, 4)
    _, i1u, i2u = _unique_rows(verts2_round)
    v2 = v2[i1u[i2u], :]

    if squeeze_out:
        return v2.ravel()
    return v2


def refine(p: Any, ind: np.ndarray) -> Callable:
    """
    Build refinement callable for Green function initialization.

    MATLAB: fun = coverlayer.refine(p, ind)

    Green function elements for neighbour cover layer elements are refined
    through polar integration.

    Parameters
    ----------
    p : ComParticle
        Composite particle containing all sub-particles.
    ind : array_like, shape (k, 2)
        Pairs of particle indices identifying cover-layer sub-particle pairs.

    Returns
    -------
    fun : callable
        Signature: fun(obj, g, f) -> (g, f). `obj` is a greenstat or greenret
        instance; (g, f) are the Green function matrices to be refined in-place.
    """
    ind = np.asarray(ind)
    if ind.ndim == 1:
        ind = ind.reshape(1, -1)

    pairs = np.vstack([ind, ind[:, ::-1]])
    pairs = np.unique(pairs, axis = 0)

    def refun(obj: Any, g: np.ndarray, f: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        cls_name = type(obj).__name__.lower()
        if 'stat' in cls_name:
            return refinestat(obj, g, f, p, pairs)
        elif 'ret' in cls_name:
            return refineret(obj, g, f, p, pairs)
        else:
            raise ValueError('[error] Unknown Green function class: {}'.format(type(obj).__name__))

    return refun


def _select_pair_elements(p: Any,
        obj: Any,
        ind_pairs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert particle-index pairs into face indices and a mapping into obj.ind.

    MATLAB: lines 16-20 of refineret.m / refinestat.m.

    Returns
    -------
    i1, i2 : ndarray
        Global face indices for rows / columns.
    ind : ndarray
        Row positions into obj.ind (linear indices) for the retained pairs.
    """
    face_idx_of = _particle_face_indices(p)

    # MATLAB `index(p, ind(:, 1).')` concatenates sub-particle face indices
    # for each requested particle. Paired with `index(p, ind(:, 2).')`,
    # sub2ind is applied element-wise, so only (fa[k], fb[k]) pairs enter
    # the refinement -- NOT the full cross product (see refinestat.m line 16-20).
    i1_all: List[np.ndarray] = []
    i2_all: List[np.ndarray] = []
    for pa, pb in ind_pairs:
        fa = face_idx_of[int(pa) - 1]
        fb = face_idx_of[int(pb) - 1]
        n_pair = min(len(fa), len(fb))
        i1_all.append(np.asarray(fa[:n_pair]).ravel())
        i2_all.append(np.asarray(fb[:n_pair]).ravel())

    if not i1_all:
        return (np.array([], dtype = int),
                np.array([], dtype = int),
                np.array([], dtype = int))

    i1 = np.concatenate(i1_all)
    i2 = np.concatenate(i2_all)

    n = p.n if hasattr(p, 'n') else p.nfaces
    lin = i1 * n + i2

    obj_ind = np.asarray(getattr(obj, 'ind', np.array([], dtype = int)))
    if obj_ind.size == 0:
        mapping = -np.ones_like(lin)
    elif obj_ind.ndim == 2 and obj_ind.shape[1] == 2:
        lin_obj = obj_ind[:, 0].astype(int) * n + obj_ind[:, 1].astype(int)
        lookup = {int(v): k for k, v in enumerate(lin_obj)}
        mapping = np.array([lookup.get(int(x), -1) for x in lin])
    else:
        lookup = {int(v): k for k, v in enumerate(obj_ind.ravel())}
        mapping = np.array([lookup.get(int(x), -1) for x in lin])

    keep = mapping >= 0
    return i1[keep], i2[keep], mapping[keep]


def _particle_face_indices(p: Any) -> List[np.ndarray]:
    """
    Return list of global face-index arrays, one per sub-particle.

    MATLAB: i = index(p, k) returns the face indices of sub-particle k in the
    concatenated COMPARTICLE numbering.
    """
    if hasattr(p, 'index'):
        n_parts = len(getattr(p, 'p', []))
        try:
            return [np.atleast_1d(p.index(k + 1)).astype(int).ravel()
                    for k in range(n_parts)]
        except Exception:
            pass

    face_idx_of: List[np.ndarray] = []
    offset = 0
    for part in getattr(p, 'p', []):
        n_part = part.n if hasattr(part, 'n') else part.nfaces
        face_idx_of.append(np.arange(offset, offset + n_part))
        offset += n_part
    return face_idx_of


def refineret(obj: Any,
        g: np.ndarray,
        f: np.ndarray,
        p: Any,
        ind_pairs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Refine retarded Green function matrices via polar integration.

    MATLAB: Greenfun/+coverlayer/refineret.m

    The Python port differs from MATLAB in storage layout: here `g` and `f`
    are full (n1, n2) matrices (rather than per-refined-element arrays of
    Taylor coefficients) and refinement is written directly into matrix
    cells. For order > 0 we accumulate the Taylor series weighted by
    (ik)^ord -- the same summation that MATLAB's eval1.m performs when
    expanding `obj.g(:, ord+1)`.
    """
    i1, i2, _ = _select_pair_elements(p, obj, ind_pairs)
    if i1.size == 0:
        return g, f

    pc = getattr(p, 'pc', p)

    # Unique-column optimization (see refinestat for details)
    unique_i2, inv_i2 = np.unique(i2, return_inverse = True)
    pos_u, weight_u, row_u = pc.quadpol(unique_i2)

    # row_u is tri-block-then-quad-block (not sorted) when unique_i2
    # mixes triangular and quadrilateral faces; group via argsort.
    sort_idx = np.argsort(row_u, kind = 'stable')
    counts = np.bincount(row_u, minlength = len(unique_i2))
    group_offsets = np.concatenate([[0], np.cumsum(counts)])

    pair_counts = counts[inv_i2]
    total = int(pair_counts.sum())
    if total == 0:
        return g, f

    pair_key = np.repeat(np.arange(len(i2)), pair_counts)
    pos_idx = np.empty(total, dtype = int)
    cursor = 0
    for uidx in inv_i2:
        n_pts = counts[uidx]
        pos_idx[cursor:cursor + n_pts] = sort_idx[group_offsets[uidx]:group_offsets[uidx + 1]]
        cursor += n_pts

    pos_all = pos_u[pos_idx]
    weight_all = weight_u[pos_idx]

    ind1 = i1[pair_key]
    pos_src = pc.pos[ind1]
    x = pos_src[:, 0] - pos_all[:, 0]
    y = pos_src[:, 1] - pos_all[:, 1]
    z = pos_src[:, 2] - pos_all[:, 2]
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)

    vec0 = -(pc.pos[ind1, :] - pc.pos[i2[pair_key], :])
    r0 = np.sqrt(np.sum(vec0 * vec0, axis = 1))

    def quad(vals: np.ndarray) -> np.ndarray:
        return np.bincount(pair_key, weights = weight_all * vals, minlength = len(i1))

    k = getattr(obj, 'k', getattr(obj, '_k', 0.0))
    order = getattr(obj, 'order', 0)

    g_accum = np.zeros(len(i1), dtype = complex)
    for ordv in range(order + 1):
        g_accum += (1j * k) ** ordv * quad((r - r0) ** ordv / (r * math.factorial(ordv)))
    g[i1, i2] = g_accum

    nvec = pc.nvec
    inp = (x * nvec[ind1, 0] + y * nvec[ind1, 1] + z * nvec[ind1, 2])
    f_accum = -quad(inp / r ** 3).astype(complex)
    for ordv in range(1, order + 1):
        term = inp * ((r - r0) ** ordv / (r ** 3 * math.factorial(ordv))
                      + (r - r0) ** (ordv - 1) / (r ** 2 * math.factorial(ordv - 1)))
        f_accum += (1j * k) ** ordv * quad(term)
    f[i1, i2] = f_accum

    return g, f


def refinestat(obj: Any,
        g: np.ndarray,
        f: np.ndarray,
        p: Any,
        ind_pairs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Refine quasistatic Green function matrices via polar integration.

    MATLAB: Greenfun/+coverlayer/refinestat.m

    Python storage: `g` and `f` are full (n1, n2) matrices -- refinement is
    written directly via g[i1, i2] / f[i1, i2].
    """
    i1, i2, _ = _select_pair_elements(p, obj, ind_pairs)
    if i1.size == 0:
        return g, f

    pc = getattr(p, 'pc', p)

    # Call quadpol once per unique column face and expand to all pairs
    # sharing that column. This avoids regenerating the same integration
    # points when i2 contains many duplicates (as happens with cover-layer
    # pairs where every face in p1 pairs with every face in p2).
    unique_i2, inv_i2 = np.unique(i2, return_inverse = True)
    pos_u, weight_u, row_u = pc.quadpol(unique_i2)

    # quadpol emits tris first, then quads, so `row_u` is NOT sorted when
    # the two face types are interleaved in `unique_i2`. Group by row value
    # via argsort: `sort_idx[group_offsets[u]:group_offsets[u+1]]` are the
    # indices into pos_u / weight_u that belong to unique-face u.
    sort_idx = np.argsort(row_u, kind = 'stable')
    counts = np.bincount(row_u, minlength = len(unique_i2))
    group_offsets = np.concatenate([[0], np.cumsum(counts)])

    pair_counts = counts[inv_i2]
    total = int(pair_counts.sum())

    if total == 0:
        return g, f

    pair_key = np.repeat(np.arange(len(i2)), pair_counts)
    # For each pair k, gather its block of integration-point indices into
    # pos_u / weight_u via the sorted grouping.
    pos_idx = np.empty(total, dtype = int)
    cursor = 0
    for u in inv_i2:
        n_pts = counts[u]
        pos_idx[cursor:cursor + n_pts] = sort_idx[group_offsets[u]:group_offsets[u + 1]]
        cursor += n_pts

    pos_all = pos_u[pos_idx]
    weight_all = weight_u[pos_idx]

    # Source (i1) for each integration point
    pos_src = pc.pos[i1[pair_key]]
    x = pos_src[:, 0] - pos_all[:, 0]
    y = pos_src[:, 1] - pos_all[:, 1]
    z = pos_src[:, 2] - pos_all[:, 2]
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)

    def quad(vals: np.ndarray) -> np.ndarray:
        return np.bincount(pair_key, weights = weight_all * vals, minlength = len(i1))

    g[i1, i2] = quad(1.0 / r)

    fx = -quad(x / r ** 3)
    fy = -quad(y / r ** 3)
    fz = -quad(z / r ** 3)

    nvec = pc.nvec
    # Python CompGreenStat stores F as a (n1, n2) normal-derivative matrix
    # regardless of deriv mode; Cartesian derivatives live in a separate
    # _Gp_raw matrix. We therefore always collapse fx / fy / fz onto the
    # receiver's normal vector here. (MATLAB refinestat.m lines 42-44 have
    # a typo -- fx appears twice in the normal-mode fallback -- but we use
    # the corrected formula because this path always feeds the F matrix.)
    f[i1, i2] = (fx * nvec[i1, 0] + fy * nvec[i1, 1] + fz * nvec[i1, 2])

    # For deriv == 'cart' also update the Cartesian-derivative storage.
    # MATLAB refinestat.m lines 40-41 write f(ind, :) = [fx, fy, fz] into
    # the full Gp matrix; in the Python port this lives in `_Gp_raw` (with
    # refinement overrides in `_f_cart_refined`). Without this update,
    # cover-layer pair elements keep the far-field (1/r^3 * area) value in
    # Gp which differs from the polar-integrated refinement used for F.
    if getattr(obj, 'deriv', 'norm') == 'cart':
        if not hasattr(obj, '_f_cart_refined'):
            obj._f_cart_refined = []
            obj._f_cart_refined_indices = []
        if hasattr(obj, '_Gp_raw') and obj._Gp_raw is not None:
            for k in range(len(i1)):
                obj._Gp_raw[i1[k], :, i2[k]] = [fx[k], fy[k], fz[k]]
        # Also override any previously queued _f_cart_refined entry for
        # this (i1, i2) pair; otherwise append a new override.
        existing = {(int(r), int(c)): idx
                    for idx, (r, c) in enumerate(obj._f_cart_refined_indices)}
        for k in range(len(i1)):
            key = (int(i1[k]), int(i2[k]))
            cart_val = np.array([fx[k], fy[k], fz[k]])
            if key in existing:
                obj._f_cart_refined[existing[key]] = cart_val
            else:
                obj._f_cart_refined.append(cart_val)
                obj._f_cart_refined_indices.append((int(i1[k]), int(i2[k])))

    return g, f
