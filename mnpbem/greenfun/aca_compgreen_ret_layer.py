import os
import sys

from typing import List, Dict, Tuple, Optional, Union, Any, Callable

import numpy as np

from .compgreen_ret_layer import CompGreenRetLayer, _StructuredGreen
from .compgreen_stat import CompStruct
from .clustertree import ClusterTree
from .hmatrix import HMatrix, make_kaware_fadmiss


class ACACompGreenRetLayer(object):

    # MATLAB: +aca/@compgreenretlayer
    # Green function for particle and layer structure using full Maxwell's
    # equations and ACA (H-matrix acceleration).
    #
    # Wraps CompGreenRetLayer with H-matrix acceleration. Uses separate
    # H-matrices for the direct (free-space) and reflected parts.

    name = 'greenfunction'
    needs = {'sim': 'ret'}

    def __init__(self,
            p: Any,
            layer: Any,
            htol: float = 1e-6,
            kmax: int = 100,
            cleaf: int = 32,
            fadmiss: Optional[Any] = None,
            eta: float = 2.5,
            **options: Any) -> None:

        # MATLAB: compgreenretlayer.m + private/init.m
        self.p = p
        self.layer = layer
        self.htol = htol
        self.kmax = kmax

        # Initialize the dense (non-ACA) CompGreenRetLayer for function evaluation
        # MATLAB: obj.g = compgreenretlayer(p, p, varargin{:})
        self.g = CompGreenRetLayer(p, p, layer, **options)

        # Build cluster tree from particle positions
        # MATLAB: tree = clustertree(p, varargin{:})
        pos = p.pos if hasattr(p, 'pos') else p.pc.pos
        n = pos.shape[0]

        # Particle index array for cluster tree
        ipart_arr = self._build_ipart_arr(p)
        self.tree = ClusterTree(pos, cleaf = cleaf, ipart_arr = ipart_arr)

        # Template H-matrix.  ``self.hmat`` keeps the static-limit block
        # structure as a fall-back, while per-call eval rebuilds the tree
        # using a wavenumber-aware fadmiss (see ``_make_fadmiss``).
        self.eta = eta
        self._user_fadmiss = fadmiss
        default_fadmiss = (fadmiss if fadmiss is not None else
                (lambda r1, r2, d: eta * min(r1, r2) < d))
        self.hmat = HMatrix(tree = self.tree, htol = htol, kmax = kmax,
                fadmiss = default_fadmiss)

        # Compute starting cluster index for each particle
        # MATLAB: obj.ind(i) = find(ind1==i & ind2==i, 1)
        np_val = p.np if hasattr(p, 'np') else 1
        self._part_cluster_ind = np.zeros(np_val, dtype = np.int64)

        if np_val > 1 and ipart_arr is not None:
            tree = self.tree
            num_nodes = tree.son.shape[0]
            for ip in range(np_val):
                for ic in range(num_nodes):
                    cs = tree.cind[ic, 0]
                    ce = tree.cind[ic, 1]
                    face_start = tree.ind[cs, 0]
                    face_end = tree.ind[ce, 0]
                    ip_start = ipart_arr[face_start]
                    ip_end = ipart_arr[face_end]
                    if ip_start == ip and ip_end == ip:
                        self._part_cluster_ind[ip] = ic
                        break

        # Cache for dense matrices and H-matrices keyed by (i, j, key, enei)
        self._cache = {}

    def _make_fadmiss(self, enei: float) -> Callable:
        """Return wavenumber-aware admissibility for this wavelength."""
        if self._user_fadmiss is not None:
            return self._user_fadmiss
        k = 2.0 * np.pi / enei
        return make_kaware_fadmiss(k, eta0 = self.eta)

    def _new_hmat(self, enei: float) -> HMatrix:
        """Allocate a fresh H-matrix with k-aware admissibility."""
        return HMatrix(tree = self.tree, htol = self.htol, kmax = self.kmax,
                fadmiss = self._make_fadmiss(enei))

    def _build_ipart_arr(self, p: Any) -> Optional[np.ndarray]:

        if not hasattr(p, 'p') or len(p.p) <= 1:
            return None

        total_n = p.n if hasattr(p, 'n') else p.pos.shape[0]
        ipart_arr = np.empty(total_n, dtype = np.int64)
        offset = 0
        for idx, part in enumerate(p.p):
            n_part = part.n
            ipart_arr[offset:offset + n_part] = idx
            offset += n_part

        return ipart_arr

    def eval(self,
            i: int,
            j: int,
            key: str,
            enei: float) -> Any:

        # MATLAB: +aca/@compgreenretlayer/eval.m
        # Depending on i and j, the Green function interaction can be only
        # direct or additionally influenced by layer reflections.
        # MATLAB (1-based): if ~(i==2 && j==2) -> eval1, else eval2
        # Python (0-based): if not (i==1 and j==1) -> eval1, else eval2
        if not (i == 1 and j == 1):
            return self._eval1(i, j, key, enei)
        else:
            return self._eval2(i, j, key, enei)

    def _eval1(self,
            i: int,
            j: int,
            key: str,
            enei: float) -> HMatrix:

        # MATLAB: +aca/@compgreenretlayer/private/eval1.m
        # Evaluate retarded Green function for layer structure (direct part only).
        # Fills dense blocks from the underlying CompGreenRetLayer.eval(),
        # fills low-rank blocks via ACA.

        cache_key = ('eval1', i, j, key, enei)
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Get the dense matrix for element-wise access
        dense_mat = self._get_dense_mat(i, j, key, enei)

        def eval_func(row: np.ndarray, col: np.ndarray) -> np.ndarray:
            if dense_mat is None:
                return np.zeros(len(row), dtype = complex)
            if np.isscalar(dense_mat):
                return np.full(len(row), complex(dense_mat))
            return dense_mat[row, col]

        # Create fresh H-matrix with k-aware admissibility for this enei.
        hmat_result = self._new_hmat(enei)
        hmat_result.aca(eval_func)

        self._cache[cache_key] = hmat_result
        return hmat_result

    def _eval2(self,
            i: int,
            j: int,
            key: str,
            enei: float) -> Dict[str, HMatrix]:

        # MATLAB: +aca/@compgreenretlayer/private/eval2.m
        # Evaluate retarded Green function for layer structure (reflected part).
        # Returns dict of H-matrices for structured Green function components:
        # 'p', 'ss', 'hh', 'sh', 'hs'
        #
        # For each component, the full matrix = direct + reflected.
        # direct part is the same for 'p', 'ss', 'hh' (free-space Green fn).
        # reflected part varies by component.

        cache_key = ('eval2', i, j, key, enei)
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Get the direct (free-space) Green function dense matrix
        dense_direct = self._get_dense_mat(i, j, key, enei)

        # Get reflected Green function component matrices
        refl_mats = self._get_reflected_structured(key, enei)

        component_names = ['p', 'ss', 'hh', 'sh', 'hs']
        result = {}

        for name in component_names:
            refl_mat = refl_mats.get(name, None)

            # Build combined matrix for this component
            # For 'p', 'ss', 'hh': total = direct + reflected
            # For 'sh', 'hs': total = reflected only (direct is zero for cross terms)
            if name in ('p', 'ss', 'hh'):
                if dense_direct is not None and refl_mat is not None:
                    combined_mat = dense_direct + refl_mat
                elif dense_direct is not None:
                    combined_mat = dense_direct.copy()
                elif refl_mat is not None:
                    combined_mat = refl_mat.copy()
                else:
                    n = self.tree.n
                    combined_mat = np.zeros((n, n), dtype = complex)
            else:
                if refl_mat is not None:
                    combined_mat = refl_mat.copy()
                else:
                    n = self.tree.n
                    combined_mat = np.zeros((n, n), dtype = complex)

            # Build H-matrix for this component via ACA
            def make_func(mat: np.ndarray) -> Callable:
                def func(row: np.ndarray, col: np.ndarray) -> np.ndarray:
                    return mat[row, col]
                return func

            eval_func = make_func(combined_mat)

            hmat_comp = self._new_hmat(enei)
            hmat_comp.aca(eval_func)

            result[name] = hmat_comp

        self._cache[cache_key] = result
        return result

    def _get_dense_mat(self,
            i: int,
            j: int,
            key: str,
            enei: float) -> Optional[np.ndarray]:

        # Get the full dense Green function matrix from CompGreenRetLayer.
        # Caches the result.
        dense_key = ('dense', i, j, key, enei)
        if dense_key in self._cache:
            return self._cache[dense_key]

        dense_mat = self.g.eval(i, j, key, enei)

        if isinstance(dense_mat, (int, float, complex)):
            n = self.tree.n
            dense_mat = np.full((n, n), complex(dense_mat))

        self._cache[dense_key] = dense_mat
        return dense_mat

    def _get_reflected_structured(self,
            key: str,
            enei: float) -> Dict[str, Optional[np.ndarray]]:

        # Get the structured reflected Green function matrices.
        # Returns dict with keys 'p', 'ss', 'hh', 'sh', 'hs'.
        refl_key = ('refl_struct', key, enei)
        if refl_key in self._cache:
            return self._cache[refl_key]

        # Get reflected Green function from GreenRetLayer
        self.g.gr.eval(enei)
        G_refl = self.g.gr.G
        F_refl = self.g.gr.F

        n = self.tree.n
        result = {}

        if key == 'G':
            refl_base = G_refl if isinstance(G_refl, np.ndarray) else np.zeros((n, n), dtype = complex)
        elif key in ('F', 'H1', 'H2'):
            refl_base = F_refl if isinstance(F_refl, np.ndarray) else np.zeros((n, n), dtype = complex)
        else:
            refl_base = np.zeros((n, n), dtype = complex)

        # For a simple substrate geometry, the structured reflected Green function
        # has the same scalar reflected Green function for p, ss, hh components,
        # and zero for cross-coupling (sh, hs) components.
        result['p'] = refl_base
        result['ss'] = refl_base
        result['hh'] = refl_base
        result['sh'] = np.zeros((n, n), dtype = complex)
        result['hs'] = np.zeros((n, n), dtype = complex)

        self._cache[refl_key] = result
        return result

    def potential(self,
            sig: Any,
            inout: int = 1) -> CompStruct:

        # MATLAB: +aca/@compgreenretlayer/potential.m
        # Potentials and surface derivatives inside/outside of particle.

        enei = sig.enei
        p = self.p

        # Set parameters that depend on inside/outside
        H_key = 'H1' if inout == 1 else 'H2'

        # Green functions for the two region combinations
        # MATLAB: G1 = obj{inout, 1}.G(enei)
        i_region = inout - 1  # Convert to 0-based

        G1 = self.eval(i_region, 0, 'G', enei)
        G2 = self.eval(i_region, 1, 'G', enei)
        H1 = self.eval(i_region, 0, H_key, enei)
        H2 = self.eval(i_region, 1, H_key, enei)

        # Get surface charges
        sig1 = sig.sig1 if hasattr(sig, 'sig1') else (sig.sig if hasattr(sig, 'sig') else np.zeros(p.n))
        sig2 = sig.sig2 if hasattr(sig, 'sig2') else np.zeros(p.n if hasattr(p, 'n') else p.pos.shape[0])

        h1 = sig.h1 if hasattr(sig, 'h1') else None
        h2 = sig.h2 if hasattr(sig, 'h2') else None

        # Scalar potential: phi = G1*sig1 + G2*sig2
        phi = self._hmat_matmul(G1, sig1) + self._hmat_matmul(G2, sig2)
        phip = self._hmat_matmul(H1, sig1) + self._hmat_matmul(H2, sig2)

        # Vector potential: a = G1*h1 + G2*h2
        if h1 is not None and h2 is not None:
            a = self._hmat_matmul_multi(G1, h1) + self._hmat_matmul_multi(G2, h2)
            ap = self._hmat_matmul_multi(H1, h1) + self._hmat_matmul_multi(H2, h2)
        else:
            n_faces = p.n if hasattr(p, 'n') else p.pos.shape[0]
            a = np.zeros((n_faces, 3), dtype = complex)
            ap = np.zeros((n_faces, 3), dtype = complex)

        # Set output
        if inout == 1:
            return CompStruct(p, enei,
                phi1 = phi, phi1p = phip, a1 = a, a1p = ap)
        else:
            return CompStruct(p, enei,
                phi2 = phi, phi2p = phip, a2 = a, a2p = ap)

    def _hmat_matmul(self,
            hmat_or_dict: Any,
            x: np.ndarray) -> np.ndarray:

        # H-matrix times vector, handling both HMatrix and dict of HMatrix
        if isinstance(hmat_or_dict, HMatrix):
            return hmat_or_dict.mtimes_vec(x)
        elif isinstance(hmat_or_dict, dict):
            # For structured Green function, use 'ss' component for scalar fields
            if 'ss' in hmat_or_dict:
                return hmat_or_dict['ss'].mtimes_vec(x)
            elif 'p' in hmat_or_dict:
                return hmat_or_dict['p'].mtimes_vec(x)
            else:
                first_key = next(iter(hmat_or_dict))
                return hmat_or_dict[first_key].mtimes_vec(x)
        else:
            return np.zeros_like(x)

    def _hmat_matmul_multi(self,
            hmat_or_dict: Any,
            x: np.ndarray) -> np.ndarray:

        # H-matrix times multi-column array (n, 3), handling both HMatrix and dict
        if isinstance(hmat_or_dict, HMatrix):
            return hmat_or_dict.mtimes_vec(x)
        elif isinstance(hmat_or_dict, dict):
            # For vector fields, use 'hh' component
            hmat = None
            if 'hh' in hmat_or_dict:
                hmat = hmat_or_dict['hh']
            elif 'p' in hmat_or_dict:
                hmat = hmat_or_dict['p']
            else:
                first_key = next(iter(hmat_or_dict))
                hmat = hmat_or_dict[first_key]
            return hmat.mtimes_vec(x)
        else:
            return np.zeros_like(x)

    def eval_full(self,
            i: int,
            j: int,
            key: str,
            enei: float) -> Any:

        # Convert H-matrix result to full dense matrix for comparison
        hmat_result = self.eval(i, j, key, enei)
        if isinstance(hmat_result, HMatrix):
            return hmat_result.full()
        elif isinstance(hmat_result, dict):
            result = {}
            for k, v in hmat_result.items():
                if isinstance(v, HMatrix):
                    result[k] = v.full()
                else:
                    result[k] = v
            return result
        else:
            return hmat_result

    def compression(self,
            i: int = 0,
            j: int = 0,
            key: str = 'G',
            enei: float = 600.0) -> float:

        hmat = self.eval(i, j, key, enei)
        if isinstance(hmat, HMatrix):
            return hmat.compression()
        elif isinstance(hmat, dict):
            # Average compression across components
            ratios = []
            for v in hmat.values():
                if isinstance(v, HMatrix):
                    ratios.append(v.compression())
            return sum(ratios) / len(ratios) if ratios else 0.0
        return 0.0

    def clear_cache(self) -> None:

        self._cache = {}

    def __repr__(self) -> str:

        n = self.p.n if hasattr(self.p, 'n') else self.p.pos.shape[0]
        n_dense = len(self.hmat.row1)
        n_lr = len(self.hmat.row2)
        return 'ACACompGreenRetLayer(n={}, dense_blocks={}, lowrank_blocks={})'.format(
            n, n_dense, n_lr)
