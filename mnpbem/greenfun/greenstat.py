"""
Standalone quasistatic Green function G = 1/r.

MATLAB: Greenfun/@greenstat/
"""

import numpy as np
from typing import Optional, Tuple, Any, List, Union

from mnpbem.utils.matlab_compat import msqrt


class GreenStat(object):
    """
    Standalone quasistatic Green function G = 1/r.

    MATLAB: @greenstat

    Computes Green function G = 1/|r - r'| and its surface derivative F
    between two particle surfaces p1 and p2, with integration refinement
    for diagonal and nearby elements.

    Properties
    ----------
    p1 : Particle
        First surface (observation points)
    p2 : Particle
        Second surface (source points)
    op : dict
        Options for calculation
    deriv : str
        'cart' for Cartesian derivatives, 'norm' for normal derivative only

    Methods
    -------
    eval(*keys, ind=None)
        Evaluate G, F, H1, H2, Gp, H1p, H2p matrices
    diag(ind, val)
        Set diagonal elements of surface derivative
    """

    def __init__(self, p1, p2, **options):
        """
        Initialize Green function in quasistatic approximation.

        MATLAB: greenstat.m + private/init.m

        Parameters
        ----------
        p1 : Particle
            Observation surface
        p2 : Particle
            Source surface
        **options : dict
            deriv : str
                'cart' or 'norm' (default: 'cart')
            AbsCutoff, RelCutoff, memsize : refinement parameters
        """
        self.p1 = p1
        self.p2 = p2
        self.op = options
        self.deriv = options.get('deriv', 'cart')

        # Private: refinement storage
        self._ind = None       # index to refined elements (linear indices into n1*n2)
        self._g = None         # refined Green function values
        self._f = None         # refined derivative values

        # Initialize
        self._init(**options)

    def _init(self, **options):
        """
        Initialize Green function with refinement.

        MATLAB: greenstat/private/init.m
        """
        import copy
        from .utils import refinematrix

        p1 = self.p1
        p2 = self.p2
        n1 = p1.pos.shape[0]
        n2 = p2.pos.shape[0]

        # Sync verts2 if needed (reuse CompGreenStat logic)
        from .compgreen_stat import CompGreenStat
        if CompGreenStat._needs_sync(p1):
            p1 = copy.deepcopy(p1)
            CompGreenStat._sync_verts2(p1)
        if p2 is not p1 and CompGreenStat._needs_sync(p2):
            p2 = copy.deepcopy(p2)
            CompGreenStat._sync_verts2(p2)

        # Store possibly-synced references for eval
        self._p1_eval = p1
        self._p2_eval = p2

        # Get refinement matrix
        refine_opts = {k: v for k, v in options.items()
                       if k in ['AbsCutoff', 'RelCutoff', 'memsize']}
        ir = refinematrix(p1, p2, **refine_opts)
        ir_dense = ir.toarray()

        # Find elements needing refinement (nonzero entries)
        nz_rows, nz_cols = np.where(ir_dense != 0)
        if len(nz_rows) == 0:
            self._ind = np.array([], dtype = int)
            self._g = np.array([])
            self._f = np.array([])
            return

        # Linear indices (row-major) into (n1, n2) matrix
        self._ind = nz_rows * n2 + nz_cols

        # Build conversion table: (row, col) -> position in ind array
        ind_sparse = np.zeros((n1, n2), dtype = int)
        for k, (r, c) in enumerate(zip(nz_rows, nz_cols)):
            ind_sparse[r, c] = k + 1  # 1-indexed (0 = no entry)

        n_refine = len(self._ind)
        self._g = np.zeros(n_refine)
        if self.deriv == 'cart':
            self._f = np.zeros((n_refine, 3))
        else:
            self._f = np.zeros(n_refine)

        # ===== Diagonal elements (ir == 2) =====
        has_quad = hasattr(p2, 'quadpol') and hasattr(p2, 'quad_integration')

        diag_mask = ir_dense == 2
        if np.any(diag_mask) and has_quad:
            diag_rows, diag_cols = np.where(diag_mask)
            # Get index into refinement array
            iface = np.array([ind_sparse[r, c] - 1 for r, c in zip(diag_rows, diag_cols)])

            pos_quad, w_quad, row_quad = p2.quadpol(diag_cols)

            # Expand face positions
            pos1_expanded = p1.pos[diag_rows[row_quad]]
            nvec1_expanded = p1.nvec[diag_rows[row_quad]]

            vec = pos1_expanded - pos_quad
            r = np.linalg.norm(vec, axis = 1)
            r = np.maximum(r, np.finfo(float).eps)

            n_diag = len(diag_rows)
            g_vals = np.bincount(row_quad, weights = w_quad / r, minlength = n_diag)[:n_diag]
            self._g[iface] = g_vals

            # Surface derivative
            n_dot_vec = np.sum(vec * nvec1_expanded, axis = 1)
            f_norm = -np.bincount(row_quad, weights = w_quad * n_dot_vec / (r ** 3), minlength = n_diag)[:n_diag]

            if self.deriv == 'norm':
                self._f[iface] = f_norm
            else:  # 'cart'
                # Normal component
                self._f[iface, 0] = f_norm

                # Tangential derivatives: for r->0 the tangential derivatives vanish
                r_safe = np.maximum(r, 1e-4 * np.max(r))

                if hasattr(p1, 'tvec1') and hasattr(p1, 'tvec2'):
                    tvec1_expanded = p1.tvec1[diag_rows[row_quad]]
                    tvec2_expanded = p1.tvec2[diag_rows[row_quad]]

                    t1_dot_vec = np.sum(vec * tvec1_expanded, axis = 1)
                    t2_dot_vec = np.sum(vec * tvec2_expanded, axis = 1)

                    f_t1 = -np.bincount(row_quad, weights = w_quad * t1_dot_vec / (r_safe ** 3), minlength = n_diag)[:n_diag]
                    f_t2 = -np.bincount(row_quad, weights = w_quad * t2_dot_vec / (r_safe ** 3), minlength = n_diag)[:n_diag]

                    # Transform from tangential/normal to Cartesian
                    nvec_face = p1.nvec[diag_rows]
                    tvec1_face = p1.tvec1[diag_rows]
                    tvec2_face = p1.tvec2[diag_rows]

                    self._f[iface, :] = (f_norm[:, np.newaxis] * nvec_face +
                                         f_t1[:, np.newaxis] * tvec1_face +
                                         f_t2[:, np.newaxis] * tvec2_face)
                else:
                    # Fallback: only normal component mapped to Cartesian
                    self._f[iface, :] = f_norm[:, np.newaxis] * p1.nvec[diag_rows]

        elif np.any(diag_mask) and not has_quad:
            # Fallback diagonal correction
            diag_rows, diag_cols = np.where(diag_mask)
            iface = np.array([ind_sparse[r, c] - 1 for r, c in zip(diag_rows, diag_cols)])
            self._g[iface] = 0.0
            if self.deriv == 'norm':
                self._f[iface] = -2.0 * np.pi
            else:
                self._f[iface, :] = -2.0 * np.pi * p1.nvec[diag_rows]

        # ===== Off-diagonal refinement elements (ir == 1) =====
        offdiag_mask = ir_dense == 1
        if np.any(offdiag_mask) and has_quad:
            _, offdiag_cols = np.where(offdiag_mask)
            unique_refine_faces = np.unique(offdiag_cols)

            pos_quad, w_sparse, _ = p2.quad_integration(unique_refine_faces)
            w_dense = w_sparse.toarray()

            pos1 = p1.pos
            nvec1 = p1.nvec

            for i, face2 in enumerate(unique_refine_faces):
                nb = np.where(offdiag_mask[:, face2])[0]
                if len(nb) == 0:
                    continue

                iface = np.array([ind_sparse[r, face2] - 1 for r in nb])

                w = w_dense[i]
                w_mask = w > 0
                pos = pos_quad[w_mask]
                w = w[w_mask]

                if len(w) == 0:
                    continue

                vec = pos1[nb, np.newaxis, :] - pos[np.newaxis, :, :]
                r = np.linalg.norm(vec, axis = 2)
                r = np.maximum(r, np.finfo(float).eps)

                self._g[iface] = (1.0 / r) @ w

                if self.deriv == 'cart':
                    # einsum to contract over quadrature points: (nb, nq, 3) x (nq,) -> (nb, 3)
                    self._f[iface, :] = -np.einsum('ijk,j->ik', vec / r[:, :, np.newaxis] ** 3, w)
                else:
                    n_dot_vec = np.sum(nvec1[nb, np.newaxis, :] * vec, axis = 2)
                    self._f[iface] = -(n_dot_vec / (r ** 3)) @ w

    def eval(self, *keys: str, ind: Optional[np.ndarray] = None) -> Any:
        """
        Evaluate Green function.

        MATLAB: greenstat/eval.m, private/eval1.m, private/eval2.m

        Parameters
        ----------
        *keys : str
            G, F, H1, H2, Gp, H1p, H2p, d
        ind : ndarray, optional
            Linear index to matrix elements to be computed.
            If None, return full matrices.

        Returns
        -------
        Single result if one key, tuple if multiple keys.
        """
        # Check if first positional arg is an index array (MATLAB compat)
        actual_keys = list(keys)
        actual_ind = ind
        if len(actual_keys) > 0 and not isinstance(actual_keys[0], str):
            actual_ind = np.asarray(actual_keys[0])
            actual_keys = actual_keys[1:]

        if len(actual_keys) == 0:
            raise ValueError('[error] At least one key must be provided')

        if actual_ind is None:
            results = self._eval_full(*actual_keys)
        else:
            results = self._eval_indexed(actual_ind, *actual_keys)

        if len(results) == 1:
            return results[0]
        return tuple(results)

    def _eval_full(self, *keys: str) -> List[np.ndarray]:
        """
        Evaluate full Green function matrices.

        MATLAB: greenstat/private/eval1.m
        """
        p1 = self._p1_eval
        p2 = self._p2_eval
        pos1 = p1.pos
        pos2 = p2.pos
        n1 = pos1.shape[0]
        n2 = pos2.shape[0]

        area = p2.area  # (n2,)

        # Difference of positions
        x = pos1[:, 0:1] - pos2[:, 0:1].T  # (n1, n2)
        y = pos1[:, 1:2] - pos2[:, 1:2].T
        z = pos1[:, 2:3] - pos2[:, 2:3].T
        d = msqrt(x ** 2 + y ** 2 + z ** 2)
        d = np.maximum(d, np.finfo(float).eps)

        # Precompute needed quantities
        need_G = 'G' in keys
        need_non_G = any(k != 'G' for k in keys)

        G = None
        if need_G:
            G = (1.0 / d) * area[np.newaxis, :]
            # Apply refinement
            if self._ind is not None and len(self._ind) > 0:
                G.ravel()[self._ind] = self._g
            G = G.reshape(n1, n2)

        Gp = None
        F_norm = None
        if need_non_G:
            if self.deriv == 'norm':
                nvec = p1.nvec
                F_norm = -(nvec[:, 0:1] * x + nvec[:, 1:2] * y + nvec[:, 2:3] * z) / (d ** 3) * area[np.newaxis, :]
                if self._ind is not None and len(self._ind) > 0:
                    F_norm.ravel()[self._ind] = self._f
                F_norm = F_norm.reshape(n1, n2)
            else:  # 'cart'
                # Cartesian derivative: Gp shape (n1, 3, n2)
                d3_area = area[np.newaxis, :] / (d ** 3)
                gp_x = -x * d3_area
                gp_y = -y * d3_area
                gp_z = -z * d3_area

                # Stack into (n1*n2, 3) for refinement, then reshape
                Gp_flat = np.empty((n1 * n2, 3))
                Gp_flat[:, 0] = gp_x.ravel()
                Gp_flat[:, 1] = gp_y.ravel()
                Gp_flat[:, 2] = gp_z.ravel()

                if self._ind is not None and len(self._ind) > 0:
                    Gp_flat[self._ind, :] = self._f

                # Reshape to (n1, 3, n2) matching MATLAB: permute(reshape(Gp, n1, n2, 3), [1,3,2])
                Gp = Gp_flat.reshape(n1, n2, 3).transpose(0, 2, 1)

        # Reset diagonal elements of d
        d_orig = d.copy()
        d_orig[d_orig <= np.finfo(float).eps] = 0

        results = []
        for key in keys:
            if key == 'G':
                results.append(G)
            elif key == 'F':
                if self.deriv == 'norm':
                    results.append(F_norm)
                else:
                    # F = nvec . Gp (inner product along axis=1 of Gp)
                    nvec = p1.nvec
                    F_val = np.sum(nvec[:, :, np.newaxis] * Gp, axis = 1)
                    results.append(F_val)
            elif key == 'H1':
                if self.deriv == 'norm':
                    results.append(F_norm + 2 * np.pi * (d_orig == 0))
                else:
                    nvec = p1.nvec
                    F_val = np.sum(nvec[:, :, np.newaxis] * Gp, axis = 1)
                    results.append(F_val + 2 * np.pi * (d_orig == 0))
            elif key == 'H2':
                if self.deriv == 'norm':
                    results.append(F_norm - 2 * np.pi * (d_orig == 0))
                else:
                    nvec = p1.nvec
                    F_val = np.sum(nvec[:, :, np.newaxis] * Gp, axis = 1)
                    results.append(F_val - 2 * np.pi * (d_orig == 0))
            elif key == 'Gp':
                if self.deriv == 'norm':
                    raise ValueError('[error] Only surface derivative computed (deriv=norm)')
                results.append(Gp)
            elif key == 'H1p':
                if self.deriv == 'norm':
                    raise ValueError('[error] Only surface derivative computed (deriv=norm)')
                results.append(Gp + 2 * np.pi * self._outer(p1.nvec, d_orig == 0))
            elif key == 'H2p':
                if self.deriv == 'norm':
                    raise ValueError('[error] Only surface derivative computed (deriv=norm)')
                results.append(Gp - 2 * np.pi * self._outer(p1.nvec, d_orig == 0))
            elif key == 'd':
                results.append(d_orig)
            else:
                raise ValueError('[error] Unknown key <{}>'.format(key))

        return results

    def _eval_indexed(self, ind: np.ndarray, *keys: str) -> List[np.ndarray]:
        """
        Evaluate Green function at specific linear indices.

        MATLAB: greenstat/private/eval2.m
        """
        p1 = self._p1_eval
        p2 = self._p2_eval
        n1 = p1.pos.shape[0]
        n2 = p2.pos.shape[0]

        ind = np.asarray(ind)
        row = ind // n2
        col = ind % n2

        pos1 = p1.pos[row, :]
        pos2 = p2.pos[col, :]
        area = p2.area[col]

        d = np.linalg.norm(pos1 - pos2, axis = 1)
        d = np.maximum(d, np.finfo(float).eps)

        # Find refinement elements
        _, ind1, ind2 = np.intersect1d(self._ind, ind, return_indices = True)

        need_G = 'G' in keys
        need_non_G = any(k != 'G' for k in keys)

        G = None
        if need_G:
            G = (1.0 / d) * area
            if len(ind1) > 0:
                G[ind2] = self._g[ind1]

        F_val = None
        Gp = None
        if need_non_G:
            if self.deriv == 'norm':
                F_val = -np.sum(p1.nvec[row, :] * (pos1 - pos2), axis = 1) / (d ** 3) * area
                if len(ind1) > 0:
                    F_val[ind2] = self._f[ind1]
            else:  # 'cart'
                need_Gp = any(k in ('Gp', 'H1p', 'H2p') for k in keys)
                if need_Gp:
                    diff = pos1 - pos2
                    gp1 = -(diff[:, 0]) / (d ** 3) * area
                    gp2 = -(diff[:, 1]) / (d ** 3) * area
                    gp3 = -(diff[:, 2]) / (d ** 3) * area
                    if len(ind1) > 0:
                        gp1[ind2] = self._f[ind1, 0]
                        gp2[ind2] = self._f[ind1, 1]
                        gp3[ind2] = self._f[ind1, 2]
                    Gp = np.column_stack([gp1, gp2, gp3])

                    if any(k in ('F', 'H1', 'H2') for k in keys):
                        F_val = np.sum(p1.nvec[row, :] * Gp, axis = 1)
                else:
                    F_val = -np.sum(p1.nvec[row, :] * (pos1 - pos2), axis = 1) / (d ** 3) * area
                    if len(ind1) > 0:
                        F_val[ind2] = np.sum(p1.nvec[row[ind2], :] * self._f[ind1, :], axis = 1)

        # Reset diagonal
        d_orig = d.copy()
        d_orig[d_orig <= np.finfo(float).eps] = 0

        results = []
        for key in keys:
            if key == 'G':
                results.append(G)
            elif key == 'F':
                results.append(F_val)
            elif key == 'H1':
                results.append(F_val + 2 * np.pi * (d_orig == 0))
            elif key == 'H2':
                results.append(F_val - 2 * np.pi * (d_orig == 0))
            elif key == 'Gp':
                if self.deriv == 'norm':
                    raise ValueError('[error] Only surface derivative computed (deriv=norm)')
                results.append(Gp)
            elif key == 'H1p':
                if self.deriv == 'norm':
                    raise ValueError('[error] Only surface derivative computed (deriv=norm)')
                nvec_diag = np.zeros_like(Gp)
                nvec_diag[d_orig == 0, :] = p1.nvec[row[d_orig == 0], :]
                results.append(Gp + 2 * np.pi * nvec_diag)
            elif key == 'H2p':
                if self.deriv == 'norm':
                    raise ValueError('[error] Only surface derivative computed (deriv=norm)')
                nvec_diag = np.zeros_like(Gp)
                nvec_diag[d_orig == 0, :] = p1.nvec[row[d_orig == 0], :]
                results.append(Gp - 2 * np.pi * nvec_diag)
            elif key == 'd':
                results.append(d_orig)
            else:
                raise ValueError('[error] Unknown key <{}>'.format(key))

        return results

    def diag(self, ind: np.ndarray, f: np.ndarray) -> 'GreenStat':
        """
        Set diagonal elements of surface derivative of Green function.

        MATLAB: greenstat/diag.m

        Parameters
        ----------
        ind : ndarray
            Index to diagonal elements (face indices, 0-indexed)
        f : ndarray
            Value of diagonal element for surface derivative

        Returns
        -------
        self : GreenStat
        """
        n1 = self.p1.pos.shape[0]
        n2 = self.p2.pos.shape[0]

        # Convert face indices to linear indices: sub2ind(n1, n2, ind, ind)
        linear_ind = ind * n2 + ind

        if self._ind is None or len(self._ind) == 0:
            # First time setting diagonal
            self._ind = linear_ind
            self._g = np.zeros(len(linear_ind))
            if self.deriv == 'cart':
                self._f = np.asarray(f).copy()
            else:
                self._f = np.asarray(f).copy()
        else:
            # Find intersection of existing and new indices
            common_mask = np.isin(self._ind, linear_ind)
            if np.any(common_mask):
                # Build mapping: for each existing index that matches, find position in new ind
                existing_positions = np.where(common_mask)[0]
                for ep in existing_positions:
                    new_pos = np.where(linear_ind == self._ind[ep])[0]
                    if len(new_pos) > 0:
                        if self._f.ndim > 1:
                            self._f[ep, :] = self._f[ep, :] + f[new_pos[0]]
                        else:
                            self._f[ep] = self._f[ep] + f[new_pos[0]]

        return self

    @staticmethod
    def _outer(nvec: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Outer product of normal vectors with mask.

        MATLAB: outer(nvec, d == 0)

        Returns shape (n1, 3, n2) where result[i, :, j] = nvec[i, :] * mask[i, j]
        """
        # nvec: (n1, 3), mask: (n1, n2)
        return nvec[:, :, np.newaxis] * mask[:, np.newaxis, :]

    @property
    def G(self) -> np.ndarray:
        return self.eval('G')

    @property
    def F(self) -> np.ndarray:
        return self.eval('F')

    @property
    def H1(self) -> np.ndarray:
        return self.eval('H1')

    @property
    def H2(self) -> np.ndarray:
        return self.eval('H2')

    @property
    def Gp(self) -> np.ndarray:
        return self.eval('Gp')

    def __repr__(self) -> str:
        return 'GreenStat(p1: {} faces, p2: {} faces)'.format(
            self.p1.pos.shape[0], self.p2.pos.shape[0])
