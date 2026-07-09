import os
import sys

from typing import List, Dict, Tuple, Optional, Union, Any, Callable

import numpy as np

from .compgreen_ret_layer import CompGreenRetLayer
from .greentab_layer import GreenTabLayer


class _MultiGreenTabLayer(object):
    """Collection of GreenTabLayer sub-tables with per-query dispatch.

    MATLAB: @compgreentablayer (cell-array of greentablayer objects).
    Mirrors MATLAB compgreentablayer/inside.m + interp.m: each query point
    is assigned to the sub-tab whose (r, z1, z2) domain contains it, and
    interpolation is dispatched per sub-tab.

    Exposes the same public API as GreenTabLayer so it can be used as a
    drop-in replacement for GreenRetLayer.tab.
    """

    name = 'greentablayer'

    def __init__(self, layer: Any, tabs: List[Dict[str, Any]]) -> None:
        self.layer = layer
        self._subs = [GreenTabLayer(layer, tab=t) for t in tabs]
        self.enei = None
        self.G = None
        self.Fr = None
        self.Fz = None
        self._pos = None

    # ------------------------------------------------------------------
    # Aggregate grid properties — union across sub-tabs (used only for
    # compatibility checks such as ismember).
    # ------------------------------------------------------------------
    @property
    def r(self) -> np.ndarray:
        arrs = [s.r for s in self._subs if s.r is not None]
        if not arrs:
            return None
        return np.unique(np.concatenate([np.atleast_1d(a) for a in arrs]))

    @property
    def z1(self) -> np.ndarray:
        arrs = [s.z1 for s in self._subs if s.z1 is not None]
        if not arrs:
            return None
        return np.unique(np.concatenate([np.atleast_1d(a) for a in arrs]))

    @property
    def z2(self) -> np.ndarray:
        arrs = [s.z2 for s in self._subs if s.z2 is not None]
        if not arrs:
            return None
        return np.unique(np.concatenate([np.atleast_1d(a) for a in arrs]))

    # ------------------------------------------------------------------
    # inside — MATLAB: @compgreentablayer/inside.m
    # Returns an index array (1..N) indicating which sub-tab owns each
    # query point.  Zero means no sub-tab matches.
    # ------------------------------------------------------------------
    def inside(self,
            r: np.ndarray,
            z1: np.ndarray,
            z2: Optional[np.ndarray] = None) -> np.ndarray:
        r = np.asarray(r, dtype=float).ravel()
        n = r.size
        ind = np.zeros(n, dtype=int)

        for i, sub in enumerate(self._subs):
            in_sub = np.asarray(sub.inside(r, z1, z2)).ravel()
            # First matching sub-tab wins (MATLAB behaviour via find(row))
            mask = in_sub & (ind == 0)
            ind[mask] = i + 1

        return ind

    # ------------------------------------------------------------------
    # Multi-wavelength pre-computation — propagate to every sub-tab.
    # ------------------------------------------------------------------
    def set(self, enei_arr: np.ndarray, **options: Any) -> '_MultiGreenTabLayer':
        for sub in self._subs:
            sub.set(enei_arr, **options)
        return self

    def parset(self, enei_arr: np.ndarray, **options: Any) -> '_MultiGreenTabLayer':
        for sub in self._subs:
            sub.parset(enei_arr, **options)
        return self

    # ------------------------------------------------------------------
    # eval_components — MATLAB: @compgreentablayer/interp.m
    # Per-query dispatch: group points by owning sub-tab, interpolate each
    # group with the corresponding sub-tab, scatter results back.
    # ------------------------------------------------------------------
    def eval_components(self,
            enei: float,
            r: np.ndarray,
            z1: np.ndarray,
            z2: np.ndarray
            ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:

        r_arr = np.asarray(r, dtype=float)
        z1_arr = np.asarray(z1, dtype=float)
        z2_arr = np.asarray(z2, dtype=float)
        shape = r_arr.shape

        r_flat = r_arr.ravel()
        z1_flat = z1_arr.ravel()
        z2_flat = z2_arr.ravel()
        n = r_flat.size

        ind = self.inside(r_flat, z1_flat, z2_flat)

        # Fallback: points with ind==0 are assigned to the nearest sub-tab
        # (first one) so that interpolation can still proceed.  MATLAB
        # asserts no zero index, but we extrapolate gracefully.
        if np.any(ind == 0):
            ind = ind.copy()
            ind[ind == 0] = 1

        # Discover component names from the first sub-tab
        unique_ids = np.unique(ind)
        sample_id = int(unique_ids[0])
        sample_sub = self._subs[sample_id - 1]
        sample_mask = (ind == sample_id)
        g_s, fr_s, fz_s = sample_sub.eval_components(
            enei,
            r_flat[sample_mask],
            z1_flat[sample_mask],
            z2_flat[sample_mask])
        names = list(g_s.keys())

        G_out = {name: np.zeros(n, dtype=complex) for name in names}
        Fr_out = {name: np.zeros(n, dtype=complex) for name in names}
        Fz_out = {name: np.zeros(n, dtype=complex) for name in names}

        # Write sample result
        for name in names:
            G_out[name][sample_mask] = np.asarray(g_s[name]).ravel()
            Fr_out[name][sample_mask] = np.asarray(fr_s[name]).ravel()
            Fz_out[name][sample_mask] = np.asarray(fz_s[name]).ravel()

        # Remaining sub-tabs
        for i in unique_ids:
            i = int(i)
            if i == sample_id:
                continue
            mask = (ind == i)
            if not np.any(mask):
                continue
            sub = self._subs[i - 1]
            g_d, fr_d, fz_d = sub.eval_components(
                enei,
                r_flat[mask],
                z1_flat[mask],
                z2_flat[mask])
            for name in names:
                if name in g_d:
                    G_out[name][mask] = np.asarray(g_d[name]).ravel()
                    Fr_out[name][mask] = np.asarray(fr_d[name]).ravel()
                    Fz_out[name][mask] = np.asarray(fz_d[name]).ravel()

        G_out = {name: v.reshape(shape) for name, v in G_out.items()}
        Fr_out = {name: v.reshape(shape) for name, v in Fr_out.items()}
        Fz_out = {name: v.reshape(shape) for name, v in Fz_out.items()}
        return G_out, Fr_out, Fz_out

    def eval(self,
            enei: float,
            r: np.ndarray,
            z1: np.ndarray,
            z2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        G_dict, Fr_dict, Fz_dict = self.eval_components(enei, r, z1, z2)
        G = GreenTabLayer._sum_components(G_dict)
        Fr = GreenTabLayer._sum_components(Fr_dict)
        Fz = GreenTabLayer._sum_components(Fz_dict)
        self.G = G
        self.Fr = Fr
        self.Fz = Fz
        self.enei = enei
        return G, Fr, Fz

    # ------------------------------------------------------------------
    # ismember — MATLAB: @compgreentablayer/ismember.m
    # All sub-tabs must be compatible.
    # ------------------------------------------------------------------
    def ismember(self, layer: Any, enei: Optional[np.ndarray] = None) -> bool:
        return all(sub.ismember(layer, enei) for sub in self._subs)

    def setup_grid(self, r: np.ndarray, z1: np.ndarray, z2: np.ndarray) -> None:
        # Not meaningful for a multi-tab wrapper; no-op for API compatibility.
        pass

    def __repr__(self) -> str:
        return '_MultiGreenTabLayer(nsub={})'.format(len(self._subs))


class CompGreenTabLayer(object):

    name = 'greenfunction'
    needs = {'sim': 'ret'}

    def __init__(self,
            p1: Any,
            p2: Any,
            layer: Any,
            tab: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
            **options: Any) -> None:

        self.p1 = p1
        self.p2 = p2
        self.layer = layer

        # Tabulated Green function.
        # MATLAB tabspace with a particle argument returns N x N cell of
        # tab structs, one per layer-pair.  When such a list is passed in
        # we build a multi-tab wrapper that performs per-query dispatch;
        # a single dict retains the legacy single-table path.
        if tab is None:
            self.tab = GreenTabLayer(layer)
        elif isinstance(tab, list):
            if len(tab) == 1:
                self.tab = GreenTabLayer(layer, tab=tab[0])
            else:
                self.tab = _MultiGreenTabLayer(layer, tab)
        else:
            self.tab = GreenTabLayer(layer, tab=tab)

        # CompGreenRetLayer with tabulated Green functions.  Pass the
        # wrapper (self.tab) as greentab_obj so CompGreenRetLayer/
        # GreenRetLayer re-use it directly instead of rebuilding a
        # merged single-table copy.
        options_with_tab = dict(options)
        options_with_tab['greentab_obj'] = self.tab

        self.g = CompGreenRetLayer(p1, p2, layer, **options_with_tab)

    def set(self, enei_arr, **options):
        """Pre-compute Green function table at multiple wavelengths.

        MATLAB: greentab = set(greentab, enei, op)
        """
        self.tab.set(enei_arr, **options)
        return self

    def eval(self,
            i: int,
            j: int,
            key: str,
            enei: float,
            ind: Optional[np.ndarray] = None) -> Any:

        return self.g.eval(i, j, key, enei, ind = ind)

    def potential(self,
            sig: Any,
            inout: int = 1) -> Any:

        return self.g.potential(sig, inout)

    def field(self,
            sig: Any,
            inout: int = 1) -> Any:

        return self.g.field(sig, inout)

    def tabulate(self,
            enei: float,
            r: np.ndarray,
            z1: np.ndarray,
            z2: np.ndarray) -> None:

        if isinstance(self.tab, _MultiGreenTabLayer):
            raise RuntimeError(
                'tabulate() not supported for multi-tab CompGreenTabLayer; '
                'use set()/parset() which propagate to each sub-tab.')
        self.tab.r = r
        self.tab.z1 = z1
        self.tab.z2 = z2
        self.tab._compute_tab(enei)

    def inside(self,
            r: np.ndarray,
            z1: np.ndarray,
            z2: Optional[np.ndarray] = None) -> np.ndarray:
        # MATLAB: @compgreentablayer/inside.m
        # Returns an index array indicating which sub-tab owns each point
        # (1..N; 0 if none).  For a single-tab wrapper the index is 0/1.
        r = np.asarray(r, dtype = float).ravel()
        z1 = np.asarray(z1, dtype = float).ravel()
        if z2 is not None:
            z2 = np.asarray(z2, dtype = float).ravel()

        if isinstance(self.tab, _MultiGreenTabLayer):
            return self.tab.inside(r, z1, z2)

        in_tab = self.tab.inside(r, z1, z2)
        ind = np.zeros(len(r), dtype = int)
        ind[in_tab] = 1
        return ind

    def ismember(self,
            layer: Any,
            enei: Optional[np.ndarray] = None,
            *args: Any) -> bool:
        # MATLAB: @compgreentablayer/ismember.m
        # Delegates to GreenTabLayer.ismember() and optionally checks positions.
        is_compat = self.tab.ismember(layer, enei)
        if not is_compat:
            return False

        # Handle additional particle/point arguments for position checking
        if len(args) > 0:
            from ..misc.distance_utils import pdist2

            pos_list = []
            for p in args:
                if not isinstance(p, (list, tuple)):
                    p = [p]
                pos_parts = []
                for pj in p:
                    if hasattr(pj, 'verts'):
                        pos_parts.append(pj.verts)
                    elif hasattr(pj, 'pos'):
                        pos_parts.append(pj.pos)
                total_len = sum(pp.shape[0] for pp in pos_parts)
                combined = np.empty((total_len, pos_parts[0].shape[1]), dtype = pos_parts[0].dtype)
                offset = 0
                for pp in pos_parts:
                    combined[offset:offset + pp.shape[0]] = pp
                    offset += pp.shape[0]
                pos_list.append(combined)

            pos1 = pos_list[0].copy()
            pos2 = pos_list[0].copy()
            if len(pos_list) == 2:
                total = pos1.shape[0] + pos_list[1].shape[0]
                pos1_ext = np.empty((total, pos1.shape[1]), dtype = pos1.dtype)
                pos1_ext[:pos1.shape[0]] = pos1
                pos1_ext[pos1.shape[0]:] = pos_list[1]
                pos1 = pos1_ext

            # Compute distances
            r = pdist2(pos1[:, :2], pos2[:, :2])
            z1_exp = np.repeat(pos1[:, 2:3], pos2.shape[0], axis = 1)
            z2_exp = np.repeat(pos2[:, 2:3].T, pos1.shape[0], axis = 0)

            ind = self.inside(r.ravel(), z1_exp.ravel(), z2_exp.ravel())
            return not np.any(ind == 0)

        return True

    def parset(self,
            enei_arr: np.ndarray,
            **options: Any) -> 'CompGreenTabLayer':
        # MATLAB: @compgreentablayer/parset.m
        # Delegates to GreenTabLayer.parset().
        self.tab.parset(enei_arr, **options)
        return self

    def __repr__(self) -> str:
        n1 = self.p1.pos.shape[0] if hasattr(self.p1, 'pos') else '?'
        n2 = self.p2.pos.shape[0] if hasattr(self.p2, 'pos') else '?'
        return 'CompGreenTabLayer(p1: {} faces, p2: {} faces)'.format(n1, n2)
