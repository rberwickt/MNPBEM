import os
import sys
import numpy as np
from typing import Optional, Tuple, Any, List, Callable, Dict

from .compgreen_stat import CompGreenStat, CompStruct
from .clustertree import ClusterTree
from .hmatrix import HMatrix


class ACACompGreenStat(object):

    # MATLAB: +aca/@compgreenstat
    # Green function for particle in quasistatic approximation using ACA.
    # Wraps CompGreenStat with H-matrix acceleration for the 1/r kernel.

    def __init__(self,
            p: Any,
            htol: float = 1e-6,
            kmax: int = 100,
            cleaf: int = 32,
            fadmiss: Optional[Callable] = None,
            **options: Any) -> None:

        # p: ComParticle object
        # htol: ACA tolerance
        # kmax: maximum rank for low-rank blocks
        # cleaf: leaf size for cluster tree bisection
        # fadmiss: admissibility function (default: eta=2.5)

        self.p = p

        # Build the underlying dense CompGreenStat for element evaluation
        # MATLAB: obj.g = compgreenstat(p, p, varargin{:})
        self.g = CompGreenStat(p, p, **options)

        # Build cluster tree from particle positions
        # MATLAB: tree = clustertree(p, varargin{:})
        pos = self._get_positions(p)
        self.tree = ClusterTree(pos, cleaf = cleaf)

        # Template H-matrix
        # MATLAB: obj.hmat = hmatrix(tree, varargin{:})
        self._htol = htol
        self._kmax = kmax
        self._fadmiss = fadmiss
        self._hmat_template = HMatrix(
            tree = self.tree, htol = htol, kmax = kmax, fadmiss = fadmiss)

        # Cache for computed H-matrices: key -> HMatrix
        self._cache = {}  # type: Dict[str, HMatrix]

        # BEM solver cache
        self._enei_cache = None
        self._mat_cache = None

    def _get_positions(self, p: Any) -> np.ndarray:

        # Extract positions from a ComParticle (or similar particle object)
        if hasattr(p, 'pos'):
            return np.asarray(p.pos, dtype = np.float64)
        elif hasattr(p, 'p'):
            # ComParticle with list of particles
            parts = p.p
            if len(parts) == 1:
                return np.asarray(parts[0].pos, dtype = np.float64)
            else:
                total = sum(pp.pos.shape[0] for pp in parts)
                pos = np.empty((total, 3), dtype = np.float64)
                offset = 0
                for pp in parts:
                    n = pp.pos.shape[0]
                    pos[offset:offset + n] = pp.pos
                    offset += n
                return pos
        else:
            raise ValueError('[error] Cannot extract positions from particle object')

    def eval(self, *keys: str) -> Any:

        # MATLAB: +aca/@compgreenstat/eval.m
        # Evaluate Green function matrices as H-matrices.
        #
        # keys: 'G', 'F', 'H1', 'H2'
        # Returns: single HMatrix or tuple of HMatrix objects

        results = []
        for key in keys:
            if key == 'Gp':
                raise ValueError('[error] Gp not implemented for ACACompGreenStat')

            hmat = self._eval_single(key)
            results.append(hmat)

        if len(results) == 1:
            return results[0]
        return tuple(results)

    def _eval_single(self, key: str) -> HMatrix:

        # Return cached H-matrix if available
        if key in self._cache:
            return self._cache[key]

        # Build the kernel function for this key
        # MATLAB: fun = @(row, col) eval(obj.g, sub2ind([p.n, p.n], row, col), varargin{i})
        n = self.g.G.shape[0]

        if key == 'G':
            dense_mat = self.g.G
        elif key == 'F':
            dense_mat = self.g.F
        elif key == 'H1':
            dense_mat = self.g.eval('H1')
        elif key == 'H2':
            dense_mat = self.g.eval('H2')
        else:
            raise ValueError('[error] Unknown key <{}>'.format(key))

        # Kernel function: takes particle-ordered row/col indices, returns values
        def kernel_fun(row: np.ndarray, col: np.ndarray) -> np.ndarray:
            return dense_mat[row, col]

        # Build H-matrix using ACA
        hmat = HMatrix(
            tree = self.tree,
            htol = self._htol,
            kmax = self._kmax,
            fadmiss = self._fadmiss)
        hmat.aca(kernel_fun)

        self._cache[key] = hmat
        return hmat

    def potential(self,
            sig: CompStruct,
            inout: int = 1) -> CompStruct:

        # MATLAB: +aca/@compgreenstat/potential.m
        # Determine potentials and surface derivatives inside/outside of particle.
        #
        # sig: CompStruct with surface charges (field 'sig')
        # inout: 1 for inside, 2 for outside

        # Select H1 or H2 based on inside/outside
        H_key = 'H1' if inout == 1 else 'H2'

        # Get Green function and surface derivative H-matrices
        G_hmat, H_hmat = self.eval('G', H_key)

        # H-matrix multiply: phi = G @ sig.sig, phip = H @ sig.sig
        sig_vals = sig.sig
        phi = G_hmat @ sig_vals
        phip = H_hmat @ sig_vals

        # Set output
        if inout == 1:
            pot = CompStruct(self.p, sig.enei, phi1 = phi, phi1p = phip)
        else:
            pot = CompStruct(self.p, sig.enei, phi2 = phi, phi2p = phip)

        return pot

    def solve(self, exc: CompStruct) -> CompStruct:

        # Solve BEM equations using H-matrix accelerated Green function.
        # Computes mat = -inv(diag(lambda) + F) and returns sig = mat @ phip
        #
        # For the BEM solve we need the full F matrix since we invert it.
        # The H-matrix is used via conversion to dense for the solve,
        # but the main benefit is in the potential() calls.
        self._init_solver(exc.enei)

        sig_values = self._mat_cache @ exc.phip

        return CompStruct(self.p, exc.enei, sig = sig_values)

    def _init_solver(self, enei: float) -> None:

        # Compute BEM resolvent matrix.
        # mat = -inv(diag(lambda) + F)
        if self._enei_cache is not None and self._enei_cache == enei:
            return

        # Get dielectric functions
        eps1_vals = self.p.eps1(enei)
        eps2_vals = self.p.eps2(enei)

        # Lambda coefficient: 2*pi*(eps1+eps2)/(eps1-eps2)
        lambda_vals = 2.0 * np.pi * (eps1_vals + eps2_vals) / (eps1_vals - eps2_vals)

        # Get F as H-matrix, convert to dense for direct solve
        F_hmat = self.eval('F')
        F_dense = F_hmat.full()

        # mat = -inv(diag(lambda) + F)
        A = np.diag(lambda_vals) + F_dense
        self._mat_cache = -np.linalg.inv(A)

        self._enei_cache = enei

    def full(self, key: str = 'G') -> np.ndarray:

        # Convert H-matrix representation back to dense matrix.
        hmat = self.eval(key)
        return hmat.full()

    def compression(self, key: str = 'G') -> float:

        # Return compression ratio for a given key.
        hmat = self.eval(key)
        return hmat.compression()

    def __getattr__(self, name: str) -> Any:

        # MATLAB: +aca/@compgreenstat/subsref.m
        # Property access: obj.G, obj.F, obj.H1, obj.H2
        if name in ('G', 'F', 'H1', 'H2'):
            return self.eval(name)
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, name))

    def __repr__(self) -> str:

        n = self.tree.n
        eta_str = ''
        if 'G' in self._cache:
            eta_str = ', compression={:.3f}'.format(self._cache['G'].compression())
        return 'ACACompGreenStat(n={}{})'.format(n, eta_str)
