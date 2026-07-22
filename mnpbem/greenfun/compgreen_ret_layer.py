"""
Composite Green function for retarded approximation with layer structure.

MATLAB: Greenfun/@compgreenretlayer/

Implements structured Green function multiplication following MATLAB
matmul2.m and matmul3.m for proper polarization decomposition
(ss, hh, p, sh, hs components).
"""

import os
import sys

from typing import List, Dict, Tuple, Optional, Union, Any, Callable

import numpy as np

from .compgreen_ret import CompGreenRet
from .compgreen_stat import CompStruct
from .greenret_layer import GreenRetLayer


class _StructuredGreen(object):

    def __init__(self,
            ss: Optional[np.ndarray] = None,
            hh: Optional[np.ndarray] = None,
            p: Optional[np.ndarray] = None,
            sh: Optional[np.ndarray] = None,
            hs: Optional[np.ndarray] = None) -> None:

        self.ss = ss if ss is not None else 0
        self.hh = hh if hh is not None else 0
        self.p = p if p is not None else 0
        self.sh = sh if sh is not None else 0
        self.hs = hs if hs is not None else 0


def _safe_matmul(A, x):
    """Matrix multiply handling zero and scalar cases."""
    if isinstance(A, (int, float)):
        if A == 0:
            return 0
        return A * x
    return A @ x


def _matmul_structured(G, x, nvec, mode='sig'):
    """Structured Green function multiplication.

    Parameters
    ----------
    G : _StructuredGreen
        Structured Green function with ss, hh, p, sh, hs components.
    x : ndarray
        Surface charge (n,) for 'sig', surface current (n,3) for 'h'.
    nvec : ndarray (n,3)
        Normal vectors (unused in simple mode, kept for API consistency).
    mode : str
        'sig' for scalar-scalar, 'h' for vector-vector multiplication.
    """
    if mode == 'sig':
        return _safe_matmul(G.ss, x)
    elif mode == 'h':
        result = np.zeros_like(x, dtype=complex)
        for d in range(x.shape[1]):
            result[:, d] = _safe_matmul(G.hh, x[:, d])
        return result
    else:
        raise ValueError("mode must be 'sig' or 'h'")


def _matmul2_refl(G_comp, sig, h, mode):
    """Structured matmul2 for reflected Green function (2D components).

    MATLAB: @compgreenretlayer/private/matmul2.m

    Parameters
    ----------
    G_comp : dict
        Keys 'ss','hh','p','sh','hs', each (n1, n2) array.
    sig : ndarray
        Surface charge, shape (n2,) or (n2, npol).
    h : ndarray
        Surface current vector, shape (n2, 3) or (n2, 3, npol).
    mode : str
        'sig' for charge contribution, 'h' for current contribution.

    Returns
    -------
    ndarray
        'sig' mode: (n1,) or (n1, npol)
        'h' mode: (n1, 3) or (n1, 3, npol)
    """
    G_ss = G_comp.get('ss', 0)
    G_hh = G_comp.get('hh', 0)
    G_p = G_comp.get('p', 0)
    G_sh = G_comp.get('sh', 0)
    G_hs = G_comp.get('hs', 0)

    if mode == 'sig':
        # phi = G.ss @ sig + G.sh @ h_z
        h_z = h[:, 2] if h.ndim == 2 else h[:, 2, :]
        pot = _safe_matmul(G_ss, sig)
        sh_term = _safe_matmul(G_sh, h_z)
        if isinstance(pot, (int, float)) and pot == 0:
            return sh_term
        if isinstance(sh_term, (int, float)) and sh_term == 0:
            return pot
        return pot + sh_term

    elif mode == 'h':
        # a = [G.p @ h_x, G.p @ h_y, G.hh @ h_z + G.hs @ sig]
        h_x = h[:, 0] if h.ndim == 2 else h[:, 0, :]
        h_y = h[:, 1] if h.ndim == 2 else h[:, 1, :]
        h_z = h[:, 2] if h.ndim == 2 else h[:, 2, :]

        pot_x = _safe_matmul(G_p, h_x)
        pot_y = _safe_matmul(G_p, h_y)
        pot_z = _safe_matmul(G_hh, h_z)
        hs_term = _safe_matmul(G_hs, sig)

        if not isinstance(pot_z, (int, float)):
            if not isinstance(hs_term, (int, float)):
                pot_z = pot_z + hs_term
        elif not isinstance(hs_term, (int, float)):
            pot_z = hs_term

        # Stack into vector
        parts = [pot_x, pot_y, pot_z]
        # Determine output shape
        for part in parts:
            if not isinstance(part, (int, float)):
                ref = part
                break
        else:
            return 0

        result_parts = []
        for part in parts:
            if isinstance(part, (int, float)):
                result_parts.append(np.zeros_like(ref))
            else:
                result_parts.append(part)

        return np.stack(result_parts, axis=1)

    else:
        raise ValueError('[error] Unknown matmul2 mode: <{}>'.format(mode))


def _matmul2_refl_3d(Gp_comp, sig, h, mode):
    """Structured matmul2 for reflected Cartesian derivative (3D components).

    MATLAB: matmul2 with Gp/H1p/H2p structured arguments.

    Parameters
    ----------
    Gp_comp : dict
        Keys 'ss','hh','p','sh','hs', each (n1, n2, 3) array.
    sig : ndarray
        Surface charge, (n2,) or (n2, npol).
    h : ndarray
        Surface current, (n2, 3) or (n2, 3, npol).
    mode : str
        'sig' or 'h'.

    Returns
    -------
    ndarray
        Always (n1, 3) or (n1, 3, npol) — vector result.
    """
    if mode == 'sig':
        # grad_phi = [Gp.ss[:,:,j] @ sig + Gp.sh[:,:,j] @ h_z  for j in 0,1,2]
        h_z = h[:, 2] if h.ndim == 2 else h[:, 2, :]
        Gp_ss = Gp_comp.get('ss', 0)
        Gp_sh = Gp_comp.get('sh', 0)

        parts = []
        for j in range(3):
            ss_j = Gp_ss[:, :, j] if isinstance(Gp_ss, np.ndarray) else 0
            sh_j = Gp_sh[:, :, j] if isinstance(Gp_sh, np.ndarray) else 0
            val = _safe_matmul(ss_j, sig)
            sh_val = _safe_matmul(sh_j, h_z)
            if isinstance(val, (int, float)) and val == 0:
                val = sh_val
            elif not isinstance(sh_val, (int, float)):
                val = val + sh_val
            parts.append(val)

        # Find reference shape
        for part in parts:
            if not isinstance(part, (int, float)):
                ref = part
                break
        else:
            return 0

        result_parts = []
        for part in parts:
            if isinstance(part, (int, float)):
                result_parts.append(np.zeros_like(ref))
            else:
                result_parts.append(part)

        return np.stack(result_parts, axis=1)

    elif mode == 'h':
        # For each Cartesian direction j:
        # a_j = [Gp.p[:,:,j] @ h_x, Gp.p[:,:,j] @ h_y,
        #        Gp.hh[:,:,j] @ h_z + Gp.hs[:,:,j] @ sig]
        # But this gives (n1, 3, 3) — too complex.
        # Actually, for H-field computation we use _cross_refl_3d instead.
        raise NotImplementedError(
            'matmul2_refl_3d h mode not needed; use _cross_refl_3d')

    else:
        raise ValueError('[error] Unknown matmul2_3d mode: <{}>'.format(mode))


def _cross_refl_3d(Gp_comp, sig, h):
    """Structured cross product for magnetic field from reflected Gp.

    MATLAB: cross() in field.m using matmul3.m

    H = curl(A) where A_j = Gp(:,:,j) @ h
    H_x = Gp(:,:,2)@h_z - Gp(:,:,3)@h_y  (indices: y*z - z*y)
    H_y = Gp(:,:,3)@h_x - Gp(:,:,1)@h_z  (indices: z*x - x*z)
    H_z = Gp(:,:,1)@h_y - Gp(:,:,2)@h_x  (indices: x*y - y*x)

    With structured decomposition (matmul3.m):
    - For parallel components (i2=0,1): use G.p component
    - For z component (i2=2): use G.hh + G.hs @ sig

    Parameters
    ----------
    Gp_comp : dict
        Keys 'p','hh','hs', each (n1, n2, 3) array.
    sig : ndarray
        Surface charge, (n2,) or (n2, npol).
    h : ndarray
        Surface current, (n2, 3) or (n2, 3, npol).

    Returns
    -------
    ndarray
        (n1, 3) or (n1, 3, npol) — magnetic field contribution.
    """
    h_x = h[:, 0] if h.ndim == 2 else h[:, 0, :]
    h_y = h[:, 1] if h.ndim == 2 else h[:, 1, :]
    h_z = h[:, 2] if h.ndim == 2 else h[:, 2, :]

    Gp_p = Gp_comp.get('p', 0)
    Gp_hh = Gp_comp.get('hh', 0)
    Gp_hs = Gp_comp.get('hs', 0)

    def matmul3_comp(i1, i2):
        """Compute G(:,:,i1) @ h(:,i2,:) with structured decomposition."""
        if i2 in (0, 1):
            # Parallel component: use G.p
            G_slice = Gp_p[:, :, i1] if isinstance(Gp_p, np.ndarray) else 0
            h_slice = h[:, i2] if h.ndim == 2 else h[:, i2, :]
            return _safe_matmul(G_slice, h_slice)
        else:
            # z component: use G.hh + G.hs
            G_hh_slice = Gp_hh[:, :, i1] if isinstance(Gp_hh, np.ndarray) else 0
            G_hs_slice = Gp_hs[:, :, i1] if isinstance(Gp_hs, np.ndarray) else 0
            hh_term = _safe_matmul(G_hh_slice, h_z)
            hs_term = _safe_matmul(G_hs_slice, sig)
            if isinstance(hh_term, (int, float)) and hh_term == 0:
                return hs_term
            if isinstance(hs_term, (int, float)) and hs_term == 0:
                return hh_term
            return hh_term + hs_term

    # cross product: H = curl(G @ h)
    # H_x = matmul3(1,2) - matmul3(2,1)  (Gy@hz - Gz@hy)
    # H_y = matmul3(2,0) - matmul3(0,2)  (Gz@hx - Gx@hz)
    # H_z = matmul3(0,1) - matmul3(1,0)  (Gx@hy - Gy@hx)
    hx = _sub_safe(matmul3_comp(1, 2), matmul3_comp(2, 1))
    hy = _sub_safe(matmul3_comp(2, 0), matmul3_comp(0, 2))
    hz = _sub_safe(matmul3_comp(0, 1), matmul3_comp(1, 0))

    parts = [hx, hy, hz]
    for part in parts:
        if not isinstance(part, (int, float)):
            ref = part
            break
    else:
        return 0

    result_parts = []
    for part in parts:
        if isinstance(part, (int, float)):
            result_parts.append(np.zeros_like(ref))
        else:
            result_parts.append(part)

    return np.stack(result_parts, axis=1)


def _sub_safe(a, b):
    """Subtract handling zero cases."""
    if isinstance(a, (int, float)) and a == 0:
        if isinstance(b, (int, float)) and b == 0:
            return 0
        return -b
    if isinstance(b, (int, float)) and b == 0:
        return a
    return a - b


def _add_safe(a, b):
    """Add handling zero cases."""
    if isinstance(a, (int, float)) and a == 0:
        return b
    if isinstance(b, (int, float)) and b == 0:
        return a
    return a + b


def _infer_structured_n1(G):
    """Infer target-point count n1 from a structured Green dict."""
    if isinstance(G, dict):
        for key in ('p', 'hh', 'hs', 'ss', 'sh'):
            val = G.get(key, None)
            if isinstance(val, np.ndarray) and val.ndim >= 2:
                return int(val.shape[0])
    return None


def _zero_field_like(G, h):
    """Return a zero vector field shaped as (n1, 3[, n_pol])."""
    n1 = _infer_structured_n1(G)
    if n1 is None:
        return 0

    tail = tuple(h.shape[2:]) if isinstance(h, np.ndarray) and h.ndim > 2 else tuple()
    return np.zeros((n1, 3) + tail, dtype = complex)


def _matmul(A, x):
    # Generalized matrix multiply handling scalar/zero, 2D, and 3D cases
    if isinstance(A, (int, float)):
        if A == 0:
            return 0
        return A * x
    if isinstance(x, (int, float)):
        if x == 0:
            return 0
        return A * x
    if not isinstance(A, np.ndarray):
        return 0

    siz_a = A.shape
    siz_x = x.shape

    if len(siz_a) == 3:
        # A is (n1, 3, n2), x is (n2,) or (n2, npol)
        n1, _, n2 = siz_a
        if len(siz_x) == 1:
            return np.tensordot(A, x, axes = ([2], [0]))
        else:
            a_flat = A.reshape(n1 * 3, n2)
            x_flat = x.reshape(n2, -1)
            y_flat = a_flat @ x_flat
            return y_flat.reshape((n1, 3) + siz_x[1:])
    else:
        # Standard 2D
        if len(siz_x) == 1:
            return A @ x
        else:
            result = A @ x.reshape(siz_x[0], -1)
            return result.reshape((siz_a[0],) + siz_x[1:])


def _matmul2(G, sig, name):
    # MATLAB: @compgreenretlayer/private/matmul2.m
    # G can be plain array or structured dict
    if not isinstance(G, dict):
        # Plain Green function
        return _matmul(G, getattr(sig, name))

    # Structured Green function
    sig1 = sig.sig1
    sig2 = sig.sig2
    h1 = sig.h1
    h2 = sig.h2

    if name == 'sig1':
        # G.ss @ sig1 + G.sh @ h1(:,3,:)
        h_z = h1[:, 2] if h1.ndim == 2 else h1[:, 2, :]
        return _add_safe(
            _matmul(G['ss'], sig1),
            _matmul(G['sh'], h_z))
    elif name == 'sig2':
        # G.ss @ sig2 + G.sh @ h2(:,3,:)
        h_z = h2[:, 2] if h2.ndim == 2 else h2[:, 2, :]
        return _add_safe(
            _matmul(G['ss'], sig2),
            _matmul(G['sh'], h_z))
    elif name == 'h1':
        return _matmul2_h(G, sig1, h1)
    elif name == 'h2':
        return _matmul2_h(G, sig2, h2)
    else:
        raise ValueError('[error] Unknown matmul2 name: <{}>'.format(name))


def _matmul2_h(G, sig_charge, h):
    # [G.p @ h(:,1,:), G.p @ h(:,2,:), G.hh @ h(:,3,:) + G.hs @ sig]
    h_x = h[:, 0] if h.ndim == 2 else h[:, 0, :]
    h_y = h[:, 1] if h.ndim == 2 else h[:, 1, :]
    h_z = h[:, 2] if h.ndim == 2 else h[:, 2, :]

    pot_x = _matmul(G['p'], h_x)
    pot_y = _matmul(G['p'], h_y)
    pot_z = _add_safe(_matmul(G['hh'], h_z), _matmul(G['hs'], sig_charge))

    parts = [pot_x, pot_y, pot_z]
    # Find reference shape
    ref = None
    for part in parts:
        if not isinstance(part, (int, float)):
            ref = part
            break

    if ref is None:
        return _zero_field_like(G, h)

    # empty + slice assignment (no np.concatenate / np.stack)
    out_shape = (ref.shape[0], 3) + ref.shape[1:]
    out = np.zeros(out_shape, dtype = complex)
    for idx, part in enumerate(parts):
        if isinstance(part, (int, float)):
            pass  # already zero
        else:
            out[:, idx] = part

    return out


def _matmul3(G, sig, name, i1, i2):
    # MATLAB: @compgreenretlayer/private/matmul3.m
    # G can be plain 3D array (n1, 3, n2) or structured dict of 3D arrays
    if not isinstance(G, dict):
        # Plain Green function: G(:, i1, :) @ sig.(name)(:, i2, :)
        h = getattr(sig, name)
        siz = list(h.shape)
        siz[0:2] = [G.shape[0], 1]
        h_slice = h[:, i2] if h.ndim == 2 else h[:, i2, :]
        val = _matmul(G[:, i1, :], h_slice)
        if isinstance(val, (int, float)):
            return np.zeros(siz, dtype = complex)
        return val.reshape(siz)

    # Structured: treat parallel and perpendicular components differently
    if name == 'h1':
        sig_charge = sig.sig1
        h = sig.h1
    else:
        sig_charge = sig.sig2
        h = sig.h2

    siz = list(h.shape)
    siz[0:2] = [G['p'].shape[0], 1]

    if i2 in (0, 1):
        # Parallel: use G.p
        h_slice = h[:, i2] if h.ndim == 2 else h[:, i2, :]
        val = _matmul(G['p'][:, i1, :], h_slice)
        if isinstance(val, (int, float)):
            return np.zeros(siz, dtype = complex)
        return val.reshape(siz)
    else:
        # Perpendicular (z): use G.hh + G.hs
        h_z = h[:, 2] if h.ndim == 2 else h[:, 2, :]
        val = _add_safe(
            _matmul(G['hh'][:, i1, :], h_z),
            _matmul(G['hs'][:, i1, :], sig_charge))
        if isinstance(val, (int, float)):
            return np.zeros(siz, dtype = complex)
        return val.reshape(siz)


def _cross3(G, sig, name):
    # MATLAB: field.m cross() function
    # cross product: curl of G @ h
    if isinstance(G, (int, float)):
        return 0
    if isinstance(G, np.ndarray) and G.size == 1:
        return 0

    hx = _sub_safe(_matmul3(G, sig, name, 1, 2), _matmul3(G, sig, name, 2, 1))
    hy = _sub_safe(_matmul3(G, sig, name, 2, 0), _matmul3(G, sig, name, 0, 2))
    hz = _sub_safe(_matmul3(G, sig, name, 0, 1), _matmul3(G, sig, name, 1, 0))

    parts = [hx, hy, hz]
    ref = None
    for part in parts:
        if not isinstance(part, (int, float)):
            ref = part
            break

    if ref is None:
        h_ref = getattr(sig, name)
        return _zero_field_like(G, h_ref)

    result_parts = []
    for part in parts:
        if isinstance(part, (int, float)):
            result_parts.append(np.zeros_like(ref))
        else:
            result_parts.append(part)

    # cat(2, ...) in MATLAB — use empty + slice assignment
    out_shape = list(ref.shape)
    out_shape[1] = 3
    out = np.zeros(out_shape, dtype = complex)
    out[:, 0:1] = result_parts[0]
    out[:, 1:2] = result_parts[1]
    out[:, 2:3] = result_parts[2]
    return out


class CompGreenRetLayer(object):

    name = 'greenfunction'
    needs = {'sim': 'ret'}

    def __new__(cls, p1=None, p2=None, layer=None, **options):
        if p1 is None or p2 is None:
            return object.__new__(cls)
        if options.get('hmatrix', False) and p1 is p2:
            n_faces = getattr(p1, 'n', None)
            if n_faces is None and hasattr(p1, 'p') and len(p1.p) > 0:
                n_faces = sum(getattr(pp, 'n', 0) for pp in p1.p)
            if n_faces is not None and n_faces > 1500:
                from .aca_compgreen_ret_layer import ACACompGreenRetLayer
                hmat_opts = {k: v for k, v in options.items() if k != 'hmatrix'}
                return ACACompGreenRetLayer(p1, layer, **hmat_opts)
        return object.__new__(cls)

    def __init__(self,
            p1: Any,
            p2: Any,
            layer: Any,
            **options: Any) -> None:

        if not isinstance(self, CompGreenRetLayer):
            return
        options.pop('hmatrix', None)
        self.p1 = p1
        self.p2 = p2
        self.layer = layer
        self.deriv = options.get('deriv', 'cart')

        # Direct (free-space) Green function
        self.g = CompGreenRet(p1, p2, **options)

        # Reflected Green function
        tab = options.pop('tab', None)
        greentab_obj = options.pop('greentab_obj', None)
        if greentab_obj is not None:
            # Use pre-tabulated GreenTabLayer directly
            self.gr = GreenRetLayer(p1, p2, layer, tab=tab, deriv=self.deriv, **options)
            self.gr.tab = greentab_obj  # Replace with pre-computed table
        else:
            self.gr = GreenRetLayer(p1, p2, layer, tab=tab,
                deriv=self.deriv, **options)

        # Indices of faces connected to layer
        self._init_layer_indices()

        # Cache
        self.enei = None
        self._G_cache = {}

    def _init_layer_indices(self) -> None:
        # MATLAB Greenfun/@compgreenretlayer/private/init.m:17-21
        #   inout1 = p1.expand( num2cell( p1.inout(:, end) ) )
        #   ind1 = find( any( inout1 == layer.ind, 2 ) )
        # i.e. faces whose OUTSIDE medium index belongs to layer.ind
        def _layer_faces(p, layer_ind):
            if hasattr(p, 'expand') and hasattr(p, 'inout'):
                inout_last = np.atleast_2d(p.inout)[:, -1]
                try:
                    face_medium = p.expand([int(v) for v in inout_last])
                    face_medium = np.asarray(face_medium).ravel()
                except Exception:
                    # Fallback: use comparticle particle index
                    pos = p.pos if hasattr(p, 'pos') else p.pc.pos
                    return np.arange(pos.shape[0])
            else:
                pos = p.pos if hasattr(p, 'pos') else p.pc.pos
                return np.arange(pos.shape[0])
            layer_ind_arr = np.atleast_1d(np.asarray(layer_ind, dtype = int))
            mask = np.isin(face_medium, layer_ind_arr)
            return np.where(mask)[0]

        self.ind1 = _layer_faces(self.p1, self.layer.ind)
        self.ind2 = _layer_faces(self.p2, self.layer.ind)

    def _is_outer_surface(self, i, j):
        # MATLAB eval1.m line 33: i1 == size(obj.p1.inout, 2) && i2 == 2
        # Python 0-based: i == n_regions_p1 - 1 && j == 1
        if hasattr(self.p1, 'inout'):
            inout1 = np.atleast_2d(self.p1.inout)
            n_regions = inout1.shape[1]
        else:
            n_regions = 2
        return i == n_regions - 1 and j == 1

    def eval(self,
            i: int,
            j: int,
            key: str,
            enei: float,
            ind: Optional[np.ndarray] = None) -> Any:

        # Get direct Green function
        g_direct = self.g.eval(i, j, key, enei, ind = ind)

        # Make sure g_direct is not zero (MATLAB eval1.m lines 23-30)
        if isinstance(g_direct, (int, float)) and g_direct == 0:
            if key in ('Gp', 'H1p', 'H2p'):
                g_direct = np.zeros((self.p1.n, 3, self.p2.n), dtype = complex)
            else:
                g_direct = np.zeros((self.p1.n, self.p2.n), dtype = complex)

        # Only add reflected Green function for outer surface
        # MATLAB eval1.m line 33: if i1 == size(obj.p1.inout,2) && i2 == 2
        if not self._is_outer_surface(i, j):
            return g_direct

        # Compute reflected Green function components
        self.gr.eval_components(enei)

        # Select reflected Green function based on key
        if key == 'G':
            gr_comp = self.gr.G_comp
        elif key in ('F', 'H1', 'H2'):
            gr_comp = self.gr.F_comp
        elif key in ('Gp', 'H1p', 'H2p'):
            gr_comp = self.gr.Gp_comp
        else:
            return g_direct

        if not gr_comp:
            return g_direct

        # Assemble structured output (MATLAB eval1.m assembly() function)
        return self._assembly(g_direct, gr_comp)

    def _assembly(self, g_direct, gr_comp):
        # MATLAB eval1.m assembly() function
        # For each component name in gr_comp:
        #   'ss', 'hh', 'p' -> G_direct + G_refl (diagonal coupling)
        #   'sh', 'hs'      -> 0 + G_refl (off-diagonal coupling)
        #
        # MATLAB init.m line 25-28 selects the layer-face sub-particle before
        # building greenretlayer, so gr.G is sized (len(ind1), len(ind2)).
        # The Python GreenRetLayer is built on the full ComParticle and gives
        # a full (n_p1, n_p2) reflected matrix — for ComParticle with
        # multi-material per particle (e.g. Au@Ag core-shell on substrate)
        # only a subset of faces touches the layer (ind1/ind2 < n_total),
        # so we must restrict gr_val to the [ind1, ind2] sub-block before
        # adding to g_base.  For uniform-eps single-particle / dimer (where
        # ind1 = range(n_total)) this is a no-op.
        result = {}
        ind1 = self.ind1
        ind2 = self.ind2
        n_p1 = self.p1.n
        n_p2 = self.p2.n

        # v1.6.5 fix: when MNPBEM_GPU_LAYER is active, gr_comp values may be
        # cupy ndarrays while g_direct is a host numpy array (or vice versa
        # when both share the GPU path).  Materialise both sides on the
        # host before the in-place add to avoid cupy/numpy mix in the
        # downstream BEMRetLayer.init code path which assumes host arrays.
        from ..utils.gpu import to_host as _to_host_arr

        g_direct_host = _to_host_arr(g_direct) if not isinstance(g_direct, np.ndarray) else g_direct

        for name in gr_comp:
            if name in ('ss', 'hh', 'p'):
                g_base = g_direct_host.copy()
            else:
                g_base = g_direct_host * 0

            gr_val = gr_comp[name]

            if g_direct_host.ndim == 2:
                # gr_val expected shape: (len(ind1), len(ind2)) MATLAB-style,
                # or (n_p1, n_p2) full-Python-style.  Restrict to sub-block
                # if needed.
                if gr_val.shape == (n_p1, n_p2):
                    if len(ind1) < n_p1 or len(ind2) < n_p2:
                        gr_block = gr_val[np.ix_(ind1, ind2)]
                    else:
                        gr_block = gr_val
                else:
                    gr_block = gr_val
                gr_block = _to_host_arr(gr_block)
                g_base[np.ix_(ind1, ind2)] = g_base[np.ix_(ind1, ind2)] + gr_block
            else:
                # 3D: shape (n_p1, 3, n_p2) full or (len(ind1), 3, len(ind2)) sub
                if gr_val.shape == (n_p1, 3, n_p2):
                    if len(ind1) < n_p1 or len(ind2) < n_p2:
                        gr_block = gr_val[np.ix_(ind1, range(3), ind2)]
                    else:
                        gr_block = gr_val
                else:
                    gr_block = gr_val
                gr_block = _to_host_arr(gr_block)
                g_base[np.ix_(ind1, range(3), ind2)] = (
                    g_base[np.ix_(ind1, range(3), ind2)] + gr_block)

            result[name] = g_base

        return result

    def eval_structured(self,
            enei: float) -> _StructuredGreen:
        """Evaluate structured reflected Green function.

        MATLAB: @compgreenretlayer/private/eval2.m assembly()
        """
        self.gr.eval_components(enei)
        G_comp = self.gr.G_comp

        return _StructuredGreen(
            ss=G_comp.get('ss', 0),
            hh=G_comp.get('hh', 0),
            p=G_comp.get('p', 0),
            sh=G_comp.get('sh', 0),
            hs=G_comp.get('hs', 0)
        )

    def potential(self,
            sig: Any,
            inout: int = 1) -> CompStruct:
        # MATLAB: @compgreenretlayer/potential.m
        enei = sig.enei

        # Determine region index for p1
        n_regions_p1 = len(self.g.con)
        p1_region = min(inout - 1, n_regions_p1 - 1)

        # Set H key based on inside/outside
        H_key = 'H1' if inout == 1 else 'H2'

        # Green functions (may be plain array or structured dict)
        G1 = self.eval(p1_region, 0, 'G', enei)
        G2 = self.eval(p1_region, 1, 'G', enei)
        H1 = self.eval(p1_region, 0, H_key, enei)
        H2 = self.eval(p1_region, 1, H_key, enei)

        # Surface charges and currents
        sig1 = sig.sig1
        sig2 = sig.sig2
        h1 = sig.h1
        h2 = sig.h2

        # Potential and surface derivative via matmul2
        phi = _add_safe(
            _matmul2(G1, sig, 'sig1'),
            _matmul2(G2, sig, 'sig2'))
        phip = _add_safe(
            _matmul2(H1, sig, 'sig1'),
            _matmul2(H2, sig, 'sig2'))
        a = _add_safe(
            _matmul2(G1, sig, 'h1'),
            _matmul2(G2, sig, 'h2'))
        ap = _add_safe(
            _matmul2(H1, sig, 'h1'),
            _matmul2(H2, sig, 'h2'))

        if inout == 1:
            return CompStruct(self.p1, enei,
                phi1 = phi, phi1p = phip, a1 = a, a1p = ap)
        else:
            return CompStruct(self.p1, enei,
                phi2 = phi, phi2p = phip, a2 = a, a2p = ap)

    def field(self,
            sig: Any,
            inout: int = 1) -> CompStruct:
        # MATLAB: @compgreenretlayer/field.m
        enei = sig.enei
        k = 2 * np.pi / enei

        # Determine region index for p1
        n_regions_p1 = len(self.g.con)
        p1_region = min(inout - 1, n_regions_p1 - 1)

        # Green function for E = i*k*A
        G1 = self.eval(p1_region, 0, 'G', enei)
        G2 = self.eval(p1_region, 1, 'G', enei)

        e = 1j * k * _add_safe(
            _matmul2(G1, sig, 'h1'),
            _matmul2(G2, sig, 'h2'))

        # Derivative of Green function for grad(phi) and curl(A)
        if inout == 1:
            H1p = self.eval(p1_region, 0, 'H1p', enei)
            H2p = self.eval(p1_region, 1, 'H1p', enei)
        else:
            H1p = self.eval(p1_region, 0, 'H2p', enei)
            H2p = self.eval(p1_region, 1, 'H2p', enei)

        # Subtract gradient of scalar potential
        grad_phi = _add_safe(
            _matmul2(H1p, sig, 'sig1'),
            _matmul2(H2p, sig, 'sig2'))
        e = _sub_safe(e, grad_phi)

        # Magnetic field: H = curl(Gp @ h)
        h = _add_safe(
            _cross3(H1p, sig, 'h1'),
            _cross3(H2p, sig, 'h2'))

        # Fail fast on scalar collapse: this indicates an upstream Green
        # function assembly mismatch and should be diagnosed, not masked.
        if np.isscalar(e) or np.asarray(e).ndim == 0:
            raise RuntimeError(
                    'CompGreenRetLayer.field produced scalar E at enei={} '.format(enei)
                    + '(inout={}, p1.n={})'.format(inout, self.p1.n))
        if np.isscalar(h) or np.asarray(h).ndim == 0:
            raise RuntimeError(
                    'CompGreenRetLayer.field produced scalar H at enei={} '.format(enei)
                    + '(inout={}, p1.n={})'.format(inout, self.p1.n))

        return CompStruct(self.p1, enei, e = e, h = h)

    def setup_tabulation(self, nr = 30, nz = 20):

        self.gr.setup_tabulation(nr = nr, nz = nz)

    def __repr__(self) -> str:
        n1 = self.p1.pos.shape[0] if hasattr(self.p1, 'pos') else '?'
        n2 = self.p2.pos.shape[0] if hasattr(self.p2, 'pos') else '?'
        return 'CompGreenRetLayer(p1: {} faces, p2: {} faces)'.format(n1, n2)


def _add_to_attr(obj, attr, refl_val):
    """Add reflected value to an attribute of a CompStruct, handling zeros."""
    direct_val = getattr(obj, attr, None)
    if isinstance(refl_val, (int, float)) and refl_val == 0:
        return direct_val if direct_val is not None else 0
    if direct_val is None:
        return refl_val
    return direct_val + refl_val
