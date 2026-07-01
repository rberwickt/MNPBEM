import os
import sys

from typing import List, Dict, Tuple, Optional, Union, Any, Callable

import numpy as np
from scipy.sparse import csr_matrix, coo_matrix

from mnpbem.utils.matlab_compat import mlinspace, msqrt

from .greentab_layer import GreenTabLayer
from .refine_utils import refinematrixlayer


class GreenRetLayer(object):

    name = 'greenretlayer'

    def __init__(self,
            p1: Any,
            p2: Any,
            layer: Any,
            tab: Optional[Dict[str, Any]] = None,
            deriv: str = 'cart',
            **options: Any) -> None:

        self.p1 = p1
        self.p2 = p2
        self.layer = layer
        self.deriv = deriv
        self.enei = None

        self.G = None
        self.F = None
        self.Gp = None

        # Per-component reflected Green caches (filled by eval_components).
        # Initialised here so the wavelength cache guard in eval_components
        # can test them without an AttributeError on the first call.
        self.G_comp = None
        self.F_comp = None
        self.Gp_comp = None

        # MATLAB init.m: offdiag='full' -> skip shape-function sparse precompute
        # (use slow but exact full boundary element integration at eval time).
        # Default (anything else, including absent) -> shape-function singularity
        # subtraction via sparse matrices (fast + accurate near-surface).
        self._offdiag_mode = options.get('offdiag', 'default')

        # Tabulated Green function
        if tab is not None:
            self.tab = GreenTabLayer(layer, tab = tab)
        else:
            self.tab = GreenTabLayer(layer)

        # Compute positions and distances for reflected Green function
        self._init_positions()

        # Compute refinement indices
        # MATLAB: init.m -> refinematrixlayer, then init1/init2 for off-diagonal
        self._init_refinement(**options)

        # Build sparse singularity-subtraction matrices (MATLAB init1.m / init2.m).
        # Skipped when offdiag='full' (fallback to slow full integration path).
        self._init_refine_sparse()

    def _init_positions(self) -> None:

        pos1 = self.p1.pos
        pos2 = self.p2.pos

        n1 = pos1.shape[0]
        n2 = pos2.shape[0]

        # Radial distance between face pairs
        dx = pos1[:, 0:1] - pos2[:, 0:1].T  # (n1, n2)
        dy = pos1[:, 1:2] - pos2[:, 1:2].T  # (n1, n2)

        self._r = msqrt(dx ** 2 + dy ** 2)  # (n1, n2)
        self._z1 = pos1[:, 2]  # (n1,)
        self._z2 = pos2[:, 2]  # (n2,)
        self._dx = dx
        self._dy = dy

    def _init_refinement(self, **options) -> None:
        """Identify diagonal and off-diagonal elements needing refinement.

        MATLAB: @greenretlayer/private/init.m
        """
        RelCutoff = options.get('RelCutoff', 3)
        AbsCutoff = options.get('AbsCutoff', 0)

        # MATLAB: always call refinematrixlayer; defaults (0,0) still select
        # near-surface elements (d2<0 or id<0 in effective comparison).
        # DO NOT short-circuit — was causing 752% error for d<1nm substrates.

        # Compute refinement matrix
        ir = refinematrixlayer(self.p1, self.p2, self.layer,
                AbsCutoff = AbsCutoff, RelCutoff = RelCutoff)

        ir_array = ir.toarray()

        # Linear indices (row-major) of diagonal elements (ir==2)
        diag_rows, diag_cols = np.where(ir_array == 2)
        if len(diag_rows) > 0:
            self._diag_id = np.ravel_multi_index(
                (diag_rows, diag_cols),
                (self.p1.n, self.p2.n))
            self._diag_faces = diag_rows  # Face indices for diagonal elements
        else:
            self._diag_id = np.array([], dtype = int)
            self._diag_faces = np.array([], dtype = int)

        # Linear indices of off-diagonal refinement elements (ir==1)
        offdiag_rows, offdiag_cols = np.where(ir_array == 1)
        if len(offdiag_rows) > 0:
            self._offdiag_ind = np.ravel_multi_index(
                (offdiag_rows, offdiag_cols),
                (self.p1.n, self.p2.n))
            self._offdiag_rows = offdiag_rows
            self._offdiag_cols = offdiag_cols
        else:
            self._offdiag_ind = np.array([], dtype = int)
            self._offdiag_rows = np.array([], dtype = int)
            self._offdiag_cols = np.array([], dtype = int)

    # -----------------------------------------------------------------
    #  Sparse init (MATLAB init1.m / init2.m)
    # -----------------------------------------------------------------
    def _init_refine_sparse(self) -> None:
        """Precompute sparse singularity-subtraction matrices.

        MATLAB reference:
          init1.m  -- normal-derivative case (ig, ifr, ifz)
          init2.m  -- Cartesian-derivative case (ig, if1, if2, ifz)

        The algorithm evaluates static ``1/d`` kernels at face vertices,
        weights them by shape functions, and stores the result in sparse
        matrices. At eval time the tabulated Green function is sampled at
        those same vertex (r0, z1, z2) coordinates, and a single sparse
        matmul accumulates the refined off-diagonal entries.

        Attributes set
        --------------
        _refine_ready : bool
            True if sparse matrices were built and should be used.
        _refine_ir    : (nvertices,) ndarray — vertex radii
        _refine_iz    : (nvertices, 2) ndarray — vertex z-values (z1, z2)
        _refine_ig    : sparse (N_offdiag, nvertices)
        _refine_ifr   : sparse (N_offdiag, nvertices)   (norm path)
        _refine_ifz   : sparse (N_offdiag, nvertices)
        _refine_if1   : sparse (N_offdiag, nvertices)   (cart path)
        _refine_if2   : sparse (N_offdiag, nvertices)   (cart path)
        """
        self._refine_ready = False

        # Skip when user requested the full integration path
        if self._offdiag_mode == 'full':
            return
        if len(self._offdiag_ind) == 0:
            return

        p1 = self.p1
        p2 = self.p2
        layer = self.layer

        # Build mapping ind(row, col) -> sparse column index (1-based in MATLAB).
        # We use 0-based here: ind_map[row, col] = k  (0 <= k < N_offdiag)
        N_off = len(self._offdiag_rows)
        ind_map = -np.ones((p1.n, p2.n), dtype = np.int64)
        # Preserve the same ordering used by self._offdiag_ind so callers that
        # consume ig*g etc. slot entries back into the flat index correctly.
        ind_map[self._offdiag_rows, self._offdiag_cols] = np.arange(N_off)

        # Face columns that need refinement
        reface = np.unique(self._offdiag_cols)

        # Accumulators (single pass)
        ir_list = []       # vertex radii (r0) - per face slab
        iz1_list = []      # vertex z1
        iz2_list = []      # vertex z2
        i1_all = []        # sparse row indices
        j_all = []         # sparse col indices into vertex stream
        g_all = []
        fr_all = []        # norm path
        fz_norm_all = []   # norm path: includes nvec_z factor
        f1_all = []        # cart path
        f2_all = []
        fz_cart_all = []   # cart path: pure z/d^3

        face_slab_offset = 0

        for face in reface:
            # rows (neighbour faces) that need refinement in this column
            nb = np.where(ind_map[:, face] >= 0)[0]
            if len(nb) == 0:
                continue
            iface = ind_map[nb, face]  # sparse-row index

            # Quadrature points + weights for this face (single-column)
            pos_q, w_sparse, _ = _particle_quad(p2, np.array([face]))
            _, nz_cols, w_vals = _sparse_find(w_sparse)
            pos = pos_q[nz_cols]
            w = np.asarray(w_vals).reshape(-1)
            if len(w) == 0:
                continue

            # Vertices and shape function for this face
            s, verts = _shape_and_verts(p2, face)

            nvec = p1.nvec[nb]

            # Distances centroids -> vertices (d0, r0, z0)
            r0, z0, d0, _, _, _ = _refine_dist(
                layer, p1.pos[nb], verts, nvec)
            # Distances centroids -> quadrature points (r, z, d, in, x, y)
            r, z, d, in_prod, x, y = _refine_dist(
                layer, p1.pos[nb], pos, nvec)

            # z-values of centroids and vertices (rounded to layer)
            z1_round_arr = layer.round_z(p1.pos[nb, 2])[0]
            z2_round_arr = layer.round_z(verts[:, 2])[0]
            n_nb = len(nb)
            n_verts = verts.shape[0]

            # MATLAB: obj.ir = [obj.ir; r0(:)] -- column-major (Fortran) flatten.
            # For fixed vertex v, neighbour k: column index = v*n_nb + k.
            ir_list.append(r0.ravel(order = 'F'))
            z1_grid = np.broadcast_to(z1_round_arr[:, np.newaxis],
                    (n_nb, n_verts)).ravel(order = 'F')
            z2_grid = np.broadcast_to(z2_round_arr[np.newaxis, :],
                    (n_nb, n_verts)).ravel(order = 'F')
            iz1_list.append(z1_grid)
            iz2_list.append(z2_grid)

            # For each shape mode (== vertex index)
            n_shape = s.shape[1]
            d_safe = np.maximum(d, 1e-30)

            for i in range(n_shape):
                d0_i = d0[:, i:i + 1]                   # (n_nb, 1)
                r0_i = np.maximum(r0[:, i:i + 1], 1e-30)
                z0_i = z0[:, i:i + 1]
                s_i = s[:, i].reshape(1, -1)            # (1, m)

                # g0 = d0(:,i) * s(:,i)^T -> (n_nb, m)
                g0 = d0_i * s_i
                fr0 = (d0_i ** 3 / r0_i) * s_i
                # z0 may be zero when points are exactly on layer; protect.
                # In MATLAB z0 is computed by dist() as sum of mindist(z) -
                # it is always >= 0, and the face is assumed off-layer, so
                # z0 > 0 in practice. 1e-30 guard is purely defensive.
                z0_safe = np.where(np.abs(z0_i) < 1e-30, 1e-30, z0_i)
                fz0 = (d0_i ** 3 / z0_safe) * s_i

                # norm path
                g_ent = (g0 / d_safe) @ w
                fr_ent = (fr0 * in_prod * r / d_safe ** 3) @ w
                fz_ent_norm = (fz0 * nvec[:, 2:3] * z / d_safe ** 3) @ w
                # cart path
                f1_ent = (fr0 * x / d_safe ** 3) @ w
                f2_ent = (fr0 * y / d_safe ** 3) @ w
                fz_ent_cart = (fz0 * z / d_safe ** 3) @ w

                # Column indices: Fortran-order flatten of r0 (n_nb, n_verts)
                # Shape mode i maps to vertex v=i: cols = i*n_nb + (0..n_nb-1)
                j = face_slab_offset + i * n_nb + np.arange(n_nb)
                i1_all.append(iface)
                j_all.append(j)
                g_all.append(g_ent)
                fr_all.append(fr_ent)
                fz_norm_all.append(fz_ent_norm)
                f1_all.append(f1_ent)
                f2_all.append(f2_ent)
                fz_cart_all.append(fz_ent_cart)

            face_slab_offset += n_nb * n_verts

        if not i1_all:
            return

        # Stack vertex coords (size = face_slab_offset = sum over faces)
        self._refine_ir = np.concatenate(ir_list)
        self._refine_iz = np.column_stack(
                [np.concatenate(iz1_list), np.concatenate(iz2_list)])
        nverts_total = len(self._refine_ir)

        i1_flat = np.concatenate(i1_all)
        j_flat = np.concatenate(j_all)

        shape = (N_off, nverts_total)
        self._refine_ig = csr_matrix(
                (np.concatenate(g_all), (i1_flat, j_flat)),
                shape = shape, dtype = complex)
        self._refine_ifr = csr_matrix(
                (np.concatenate(fr_all), (i1_flat, j_flat)),
                shape = shape, dtype = complex)
        self._refine_ifz_norm = csr_matrix(
                (np.concatenate(fz_norm_all), (i1_flat, j_flat)),
                shape = shape, dtype = complex)
        self._refine_if1 = csr_matrix(
                (np.concatenate(f1_all), (i1_flat, j_flat)),
                shape = shape, dtype = complex)
        self._refine_if2 = csr_matrix(
                (np.concatenate(f2_all), (i1_flat, j_flat)),
                shape = shape, dtype = complex)
        self._refine_ifz_cart = csr_matrix(
                (np.concatenate(fz_cart_all), (i1_flat, j_flat)),
                shape = shape, dtype = complex)

        self._refine_ready = True

    # -----------------------------------------------------------------
    #  Helper: interpolate tab + return r_rounded and zmin
    # -----------------------------------------------------------------
    def _interp_components_with_pos(self,
            r: np.ndarray,
            z1: np.ndarray,
            z2: np.ndarray
            ) -> Tuple[Dict, Dict, Dict, np.ndarray, np.ndarray]:
        """Per-component interpolation returning also r_rounded and zmin.

        Mirrors MATLAB greentablayer/interp.m 5-output mode:
        [G, Fr, Fz, r, zmin].
        """
        layer = self.layer

        r = np.maximum(r, layer.rmin)
        z1_r, z2_r = layer.round_z(z1, z2)

        zmin1, _ = layer.mindist(z1_r)
        zmin2, _ = layer.mindist(z2_r)
        zmin = zmin1 + zmin2

        G_dict, Fr_dict, Fz_dict = self.tab.eval_components(
            self.enei, r, z1_r, z2_r)

        return G_dict, Fr_dict, Fz_dict, r, zmin

    # -----------------------------------------------------------------
    #  Divergent term coefficient (Waxenegger et al. Eq. 17)
    # -----------------------------------------------------------------
    def _compute_divergent_coeff(self,
            face_z: np.ndarray
            ) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """Compute the divergent term coefficient f for faces in the layer.

        MATLAB initrefl1.m lines 89-98:
          [~, ind] = mindist(layer, p.pos(id(lin), 3));
          z0 = layer.z(ind);
          dir = sign(p.pos(id(lin), 3) - z0);
          [~, ~, f0, r0, z0] = interp(tab, 0*z0, z0+1e-10*dir, z0+1e-10*dir);
          f = f0 .* (r0.^2 + z0.^2).^1.5 ./ z0;

        Parameters
        ----------
        face_z : ndarray
            z-coordinates of faces located in the layer.

        Returns
        -------
        f_dict : dict of ndarray
            Per-component divergent coefficient.
        zmin0 : ndarray
            zmin values at the probe points.
        """
        layer = self.layer

        _, ind = layer.mindist(face_z)
        z0_layer = layer.z[ind - 1]  # 1-based ind -> 0-based

        direction = np.sign(face_z - z0_layer)
        direction[direction == 0] = 1.0

        z_probe = z0_layer + 1e-10 * direction
        r_zero = np.zeros_like(z_probe)

        # Get per-component values near the singularity
        _, _, Fz0_dict, r0, zmin0 = self._interp_components_with_pos(
            r_zero, z_probe, z_probe)

        d0 = msqrt(r0 ** 2 + zmin0 ** 2)

        f_dict = {}
        for name in Fz0_dict:
            # f = f0 * d^3 / zmin  (extract normalized coefficient)
            f_dict[name] = Fz0_dict[name] * d0 ** 3 / np.maximum(np.abs(zmin0), 1e-30)

        return f_dict, zmin0

    # -----------------------------------------------------------------
    #  Diagonal refinement (polar integration)
    # -----------------------------------------------------------------
    def _refine_diagonal_norm(self,
            G_dict: Dict[str, np.ndarray],
            F_dict: Dict[str, np.ndarray]
            ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Refine diagonal elements for deriv='norm'.

        MATLAB: initrefl1.m lines 44-131
        """
        if len(self._diag_id) == 0:
            return G_dict, F_dict

        n1 = self.p1.n
        n2 = self.p2.n
        p = self.p1
        id_linear = self._diag_id
        id_faces = self._diag_faces

        # Polar integration points and weights
        pos_quad, weight_quad, row_quad = p.quadpol(id_faces)

        # Centroids expanded to integration point size
        pos_centroid = p.pos[id_faces[row_quad]]

        # Difference vectors
        dx = pos_quad[:, 0] - pos_centroid[:, 0]
        dy = pos_quad[:, 1] - pos_centroid[:, 1]
        r_quad = msqrt(dx ** 2 + dy ** 2)
        z1_quad = pos_quad[:, 2]
        z2_quad = pos_centroid[:, 2]

        # Normal vectors
        nvec_quad = p.nvec[id_faces[row_quad]]
        r_safe = np.maximum(r_quad, 1e-10)
        in_product = (dx * nvec_quad[:, 0] + dy * nvec_quad[:, 1]) / r_safe

        # Interpolate Green function at quadrature points
        g_dict, fr_dict, fz_dict, r_rounded, zmin_quad = \
            self._interp_components_with_pos(r_quad, z1_quad, z2_quad)
        rr_quad = msqrt(r_rounded ** 2 + zmin_quad ** 2)

        # Identify faces located in the layer (close to interface)
        _, lin = self.layer.indlayer(p.pos[id_faces, 2])

        # Expand to integration point size
        lin_row = lin[row_quad]

        # Compute divergent coefficient for in-layer faces
        if np.any(lin):
            lin_faces_z = p.pos[id_faces[lin], 2]
            f_coeff, _ = self._compute_divergent_coeff(lin_faces_z)

            lin_map = np.full(len(id_faces), -1, dtype = int)
            lin_map[lin] = np.arange(np.sum(lin))
            irow = lin_map[row_quad[lin_row]]

            # Linear index for in-layer diagonal matrix entries
            lin2_row = id_faces[lin]
            lin2_col = id_faces[lin]
            lin2_linear = np.ravel_multi_index((lin2_row, lin2_col), (n1, n2))
        else:
            f_coeff = None

        names = list(g_dict.keys())
        n_faces_diag = len(id_faces)

        for name in names:
            g_vals = g_dict[name]
            fr_vals = fr_dict[name]
            fz_vals = fz_dict[name].copy()

            # Integrate G using polar quadrature
            G_refined = np.zeros(n_faces_diag, dtype = complex)
            np.add.at(G_refined, row_quad, weight_quad * g_vals)
            G_flat = G_dict[name].ravel()
            G_flat[id_linear] = G_refined
            G_dict[name] = G_flat.reshape(n1, n2)

            # Subtract divergent term from fz for in-layer faces
            if np.any(lin) and f_coeff is not None:
                f = f_coeff[name]
                rr_safe = np.maximum(rr_quad[lin_row], 1e-30)
                fz_vals[lin_row] -= f[irow] * zmin_quad[lin_row] / rr_safe ** 3

            # Integrate F using polar quadrature
            f_integrand = fr_vals * in_product + fz_vals * nvec_quad[:, 2]
            F_refined = np.zeros(n_faces_diag, dtype = complex)
            np.add.at(F_refined, row_quad, weight_quad * f_integrand)

            F_flat = F_dict[name].ravel()
            F_flat[id_linear] = F_refined

            # Add back divergent term
            if np.any(lin) and f_coeff is not None:
                f = f_coeff[name]
                F_flat[lin2_linear] += 2 * np.pi * f * p.nvec[id_faces[lin], 2]

            F_dict[name] = F_flat.reshape(n1, n2)

        return G_dict, F_dict

    def _refine_diagonal_cart_gp_only(self,
            Gp_dict: Dict[str, np.ndarray]) -> None:
        """Refine only the Gp (Cartesian derivative) diagonal elements.

        MATLAB: initrefl2.m lines 43-129, Gp parts only.
        """
        if len(self._diag_id) == 0:
            return

        n1 = self.p1.n
        n2 = self.p2.n
        p = self.p1
        id_faces = self._diag_faces

        pos_quad, weight_quad, row_quad = p.quadpol(id_faces)
        pos_centroid = p.pos[id_faces[row_quad]]

        dx = pos_quad[:, 0] - pos_centroid[:, 0]
        dy = pos_quad[:, 1] - pos_centroid[:, 1]
        r_quad = msqrt(dx ** 2 + dy ** 2)
        z1_quad = pos_quad[:, 2]
        z2_quad = pos_centroid[:, 2]

        _, fr_dict, fz_dict, r_rounded, zmin_quad = \
            self._interp_components_with_pos(r_quad, z1_quad, z2_quad)
        rr_quad = msqrt(r_rounded ** 2 + zmin_quad ** 2)
        r_safe = np.maximum(r_rounded, 1e-10)

        _, lin = self.layer.indlayer(p.pos[id_faces, 2])
        lin_row = lin[row_quad]

        if np.any(lin):
            lin_faces_z = p.pos[id_faces[lin], 2]
            f_coeff, _ = self._compute_divergent_coeff(lin_faces_z)
            lin_map = np.full(len(id_faces), -1, dtype = int)
            lin_map[lin] = np.arange(np.sum(lin))
            irow = lin_map[row_quad[lin_row]]
        else:
            f_coeff = None

        names = list(Gp_dict.keys())
        n_faces_diag = len(id_faces)

        for name in names:
            fr_vals = fr_dict[name]
            fz_vals = fz_dict[name].copy()

            # Subtract divergent term
            if np.any(lin) and f_coeff is not None:
                f = f_coeff[name]
                rr_safe = np.maximum(rr_quad[lin_row], 1e-30)
                fz_vals[lin_row] -= f[irow] * zmin_quad[lin_row] / rr_safe ** 3

            Gp_x_refined = np.zeros(n_faces_diag, dtype = complex)
            Gp_y_refined = np.zeros(n_faces_diag, dtype = complex)
            Gp_z_refined = np.zeros(n_faces_diag, dtype = complex)
            np.add.at(Gp_x_refined, row_quad, weight_quad * fr_vals * dx / r_safe)
            np.add.at(Gp_y_refined, row_quad, weight_quad * fr_vals * dy / r_safe)
            np.add.at(Gp_z_refined, row_quad, weight_quad * fz_vals)

            Gp = Gp_dict[name]
            Gp[id_faces, 0, id_faces] = Gp_x_refined
            Gp[id_faces, 1, id_faces] = Gp_y_refined
            Gp[id_faces, 2, id_faces] = Gp_z_refined

            # Add back divergent term for z-component
            if np.any(lin) and f_coeff is not None:
                f = f_coeff[name]
                lin_faces_idx = id_faces[lin]
                Gp[lin_faces_idx, 2, lin_faces_idx] += 2 * np.pi * f

            Gp_dict[name] = Gp

    # -----------------------------------------------------------------
    #  Off-diagonal refinement (boundary element integration)
    # -----------------------------------------------------------------
    def _refine_offdiagonal_components(self) -> None:
        """Refine off-diagonal elements for all components (G, F, Gp).

        MATLAB: initrefl1.m lines 134-186 + initrefl2.m lines 132-190
        Uses full boundary element integration over source faces.
        """
        if len(self._offdiag_ind) == 0:
            return

        n1 = self.p1.n
        n2 = self.p2.n
        names = list(self.G_comp.keys())

        ind_matrix = np.zeros((n1, n2), dtype = int)
        offdiag_rows, offdiag_cols = np.unravel_index(
            self._offdiag_ind, (n1, n2))
        ind_matrix[offdiag_rows, offdiag_cols] = 1

        columns_with_refine = np.unique(offdiag_cols)

        for col in columns_with_refine:
            rows = np.where(ind_matrix[:, col])[0]
            if len(rows) == 0:
                continue

            pos1 = self.p1.pos[rows]
            pos_quad, w_sparse, _ = _particle_quad(self.p2, np.array([col]))

            _, nz_cols, nz_vals = _sparse_find(w_sparse)
            pos2 = pos_quad[nz_cols]
            w = nz_vals

            if len(w) == 0:
                continue

            x = pos1[:, 0:1] - pos2[:, 0].T
            y = pos1[:, 1:2] - pos2[:, 1].T
            r = msqrt(x ** 2 + y ** 2)

            z1 = np.tile(pos1[:, 2:3], (1, len(w)))
            z2 = np.tile(pos2[:, 2].T, (len(rows), 1))

            nvec = self.p1.nvec[rows]
            r_safe = np.maximum(r, 1e-10)
            in_product = (nvec[:, 0:1] * x + nvec[:, 1:2] * y) / r_safe

            g_dict_q, fr_dict_q, fz_dict_q = self.tab.eval_components(
                self.enei,
                r.ravel(),
                z1.ravel(),
                z2.ravel())

            for name in names:
                g_q = g_dict_q[name].reshape(r.shape)
                fr_q = fr_dict_q[name].reshape(r.shape)
                fz_q = fz_dict_q[name].reshape(r.shape)

                # Refine G
                self.G_comp[name][rows, col] = g_q @ w

                # Refine F (normal derivative)
                f_integrand = fr_q * in_product + fz_q * nvec[:, 2:3]
                self.F_comp[name][rows, col] = f_integrand @ w

                # Refine Gp (Cartesian derivative)
                self.Gp_comp[name][rows, 0, col] = (fr_q * x / r_safe) @ w
                self.Gp_comp[name][rows, 1, col] = (fr_q * y / r_safe) @ w
                self.Gp_comp[name][rows, 2, col] = fz_q @ w

    # -----------------------------------------------------------------
    #  Off-diagonal refinement (sparse, fast path)
    # -----------------------------------------------------------------
    def _refine_offdiagonal_sparse(self) -> None:
        """Fast sparse off-diagonal refinement using singularity subtraction.

        MATLAB: initrefl1.m lines 135-143 + initrefl2.m lines 133-141.

        At init time we precomputed ``ig``, ``ifr``, ``ifz``, ``if1``, ``if2``
        which encode shape-function-weighted static-kernel integrals.  At
        eval time we sample the tabulated Green function at the vertex
        points stored in ``_refine_ir``/``_refine_iz`` and accumulate via
        sparse matmul.
        """
        if not self._refine_ready:
            return

        # Interpolate tabulated Green function at vertex points.
        g_dict, fr_dict, fz_dict = self.tab.eval_components(
                self.enei,
                self._refine_ir,
                self._refine_iz[:, 0],
                self._refine_iz[:, 1])

        names = list(self.G_comp.keys())
        n1 = self.p1.n
        n2 = self.p2.n

        # Rows/cols of off-diagonal entries (flat -> 2D).
        rows = self._offdiag_rows
        cols = self._offdiag_cols

        for name in names:
            g_vec = np.asarray(g_dict[name]).ravel()
            fr_vec = np.asarray(fr_dict[name]).ravel()
            fz_vec = np.asarray(fz_dict[name]).ravel()

            # Sparse matmul: (N_off, nverts) @ (nverts,) = (N_off,)
            g_ref = np.asarray(self._refine_ig @ g_vec).ravel()
            # norm-path: F = ifr*fr + ifz_norm*fz  (ifz_norm already has nvec_z)
            f_ref = (np.asarray(self._refine_ifr @ fr_vec).ravel()
                   + np.asarray(self._refine_ifz_norm @ fz_vec).ravel())
            # cart-path (Gp)
            gp_x = np.asarray(self._refine_if1 @ fr_vec).ravel()
            gp_y = np.asarray(self._refine_if2 @ fr_vec).ravel()
            gp_z = np.asarray(self._refine_ifz_cart @ fz_vec).ravel()

            # Slot back into per-component matrices.
            self.G_comp[name][rows, cols] = g_ref
            self.F_comp[name][rows, cols] = f_ref
            self.Gp_comp[name][rows, 0, cols] = gp_x
            self.Gp_comp[name][rows, 1, cols] = gp_y
            self.Gp_comp[name][rows, 2, cols] = gp_z

    # -----------------------------------------------------------------
    #  Apply refinement to per-component Green functions
    # -----------------------------------------------------------------
    def _apply_refinement_components(self) -> None:
        """Apply diagonal and off-diagonal refinement to per-component Green functions.

        Handles both norm and cart derivatives simultaneously since
        eval_components always needs both F_comp and Gp_comp.
        """
        # Diagonal refinement
        if len(self._diag_id) > 0:
            # Refine G_comp and F_comp (norm-style: initrefl1.m)
            self.G_comp, self.F_comp = self._refine_diagonal_norm(
                self.G_comp, self.F_comp)
            # Refine Gp_comp (cart-style: initrefl2.m)
            self._refine_diagonal_cart_gp_only(self.Gp_comp)
            # MATLAB compgreenretlayer routes 'F'/'H1'/'H2' via gr.F, which is
            # filled by initrefl2.m: F = inner(nvec, Gp) AFTER cart refinement.
            # Recompute F_comp[diag] from refined Gp_comp[diag] so we match
            # MATLAB's cart-style result (rim faces with quad points near the
            # centroid otherwise diverge between the norm/cart formulas due to
            # the rmin-clamp in `interp` only being applied on the cart path).
            nvec1 = self.p1.nvec
            for nm in self.F_comp.keys():
                Gp_diag = self.Gp_comp[nm]
                F_einsum = np.einsum('ik,ikj->ij', nvec1, Gp_diag)
                F_flat = self.F_comp[nm].ravel()
                F_flat[self._diag_id] = F_einsum.ravel()[self._diag_id]
                self.F_comp[nm] = F_flat.reshape(self.F_comp[nm].shape)

        # Off-diagonal refinement: prefer sparse (shape-function) path when
        # available, fall back to full integration otherwise.
        if len(self._offdiag_ind) > 0:
            if getattr(self, '_refine_ready', False):
                self._refine_offdiagonal_sparse()
            else:
                self._refine_offdiagonal_components()

    # -----------------------------------------------------------------
    #  Main evaluation methods
    # -----------------------------------------------------------------
    def eval(self,
            enei: float) -> None:

        if self.enei is not None and np.isclose(self.enei, enei):
            return

        self.enei = enei

        n1 = self.p1.pos.shape[0]
        n2 = self.p2.pos.shape[0]

        # Round z-values to avoid being too close to layer interface
        z1, z2 = self.layer.round_z(self._z1, self._z2)

        r_flat = self._r.ravel()
        z1_flat = np.repeat(z1, n2)
        z2_flat = np.tile(z2, n1)

        # Enforce minimum radial distance
        r_flat = np.maximum(r_flat, self.layer.rmin)

        # Compute reflected Green function
        G, Fr, Fz = self.tab.eval(enei, r_flat, z1_flat, z2_flat)

        G = G.reshape(n1, n2)
        Fr = Fr.reshape(n1, n2)
        Fz = Fz.reshape(n1, n2)

        # Multiply G with p2.area (MATLAB initrefl1.m line 36, initrefl2.m line 35).
        # Fr/Fz keep raw (no-area) form; area is applied inside
        # _compute_F_norm/_compute_F_cart to match MATLAB's multiplication
        # order (see Wave 17 A).
        area2 = self.p2.area  # (n2,)
        self.G = G * area2[np.newaxis, :]

        # Compute surface derivative
        if self.deriv == 'norm':
            self._compute_F_norm(self.G, Fr, Fz, area2)
        else:
            self._compute_F_cart(self.G, Fr, Fz, area2)

    def eval_components(self,
            enei: float) -> None:
        """Compute per-component reflected Green function (G, F, Gp).

        Always computes both normal and Cartesian derivatives regardless
        of self.deriv, since field() needs Gp and potential() needs F.

        After calling, results are stored in:
          self.G_comp  : dict of (n1, n2) arrays
          self.F_comp  : dict of (n1, n2) arrays (normal derivative)
          self.Gp_comp : dict of (n1, n2, 3) arrays (Cartesian derivative)
        """
        _gprof = os.environ.get('MNPBEM_INIT_PROFILE', '0') == '1'
        if _gprof:
            import time as _t
            _gt0 = _t.perf_counter()

        # The per-component reflected Green function (G_comp/F_comp/Gp_comp)
        # depends only on ``enei`` and the fixed geometry, never on which
        # CompGreenRetLayer.eval() key requested it.  BEMRetLayer.init()
        # issues four outer-surface evals per wavelength (G22, G12, H22,
        # H12), each of which previously re-ran the full O(n^2) tabulated
        # interpolation + refinement here — a 4x redundancy that dominates
        # the substrate BEM build (≈19s per call on a 2696-face dimer).
        # Cache by wavelength so the heavy work runs once per enei; mirrors
        # the guard already present in eval() above.
        if (self.enei is not None and np.isclose(self.enei, enei)
                and self.G_comp is not None
                and self.F_comp is not None
                and self.Gp_comp is not None):
            if _gprof:
                print('[gr-prof]   eval_components CACHED (enei={:.3f})'.format(
                    enei), flush=True)
            return

        n1 = self.p1.pos.shape[0]
        n2 = self.p2.pos.shape[0]

        z1, z2 = self.layer.round_z(self._z1, self._z2)

        r_flat = self._r.ravel()
        z1_flat = np.repeat(z1, n2)
        z2_flat = np.tile(z2, n1)
        r_flat = np.maximum(r_flat, self.layer.rmin)

        G_dict, Fr_dict, Fz_dict = self.tab.eval_components(
            enei, r_flat, z1_flat, z2_flat)
        if _gprof:
            print('[gr-prof]   tab.eval_components(full) {:.3f}s'.format(
                _t.perf_counter() - _gt0), flush=True); _gt0 = _t.perf_counter()

        self.enei = enei
        self.G_comp = {}
        self.F_comp = {}
        self.Gp_comp = {}

        nvec1 = self.p1.nvec
        r_safe = np.maximum(self._r, np.finfo(float).eps)
        area2 = self.p2.area  # (n2,)

        # Precompute the area-weighted Cartesian factors once.  At 5768
        # faces each (n,3,n) c128 Gp block is ~1.6 GB; the old form
        # allocated it via np.zeros, filled the slices, then re-allocated
        # a second full copy for ``Gp * area2`` (peak 2 blocks) and finally
        # routed the (n,3,n) reduction through einsum.  Folding area2 into
        # the per-slice expressions, writing into a single np.empty buffer,
        # and computing F directly from the three slices halves the
        # allocation traffic and the einsum overhead — numerically identical
        # to the previous form (associativity of the elementwise products).
        area_row = area2[np.newaxis, :]                  # (1, n2)
        dx_over_r = self._dx / r_safe                    # (n1, n2)
        dy_over_r = self._dy / r_safe
        nvx = nvec1[:, 0:1]
        nvy = nvec1[:, 1:2]
        nvz = nvec1[:, 2:3]

        for name in G_dict:
            G = G_dict[name].reshape(n1, n2)
            Fr = Fr_dict[name].reshape(n1, n2)
            Fz = Fz_dict[name].reshape(n1, n2)

            # MATLAB initrefl2.m line 35-40: Gp uses raw Fr/Fz, then area'
            # is applied via fun(...); G itself is multiplied by area' in place.
            self.G_comp[name] = G * area_row

            Fr_area = Fr * area_row
            gp_x = Fr_area * dx_over_r
            gp_y = Fr_area * dy_over_r
            gp_z = Fz * area_row

            Gp = np.empty((n1, 3, n2), dtype = complex)
            Gp[:, 0, :] = gp_x
            Gp[:, 1, :] = gp_y
            Gp[:, 2, :] = gp_z
            self.Gp_comp[name] = Gp

            # Normal derivative: F = inner(nvec, Gp) (MATLAB initrefl2.m
            # line 197).  Compute directly from the slices to skip einsum's
            # (n,3,n) intermediate; identical to np.einsum('ik,ikj->ij').
            self.F_comp[name] = gp_x * nvx + gp_y * nvy + gp_z * nvz

        if _gprof:
            print('[gr-prof]   build G/F/Gp comp        {:.3f}s'.format(
                _t.perf_counter() - _gt0), flush=True); _gt0 = _t.perf_counter()

        # Apply refinement if configured
        has_refinement = (len(self._diag_id) > 0 or len(self._offdiag_ind) > 0)
        if has_refinement:
            self._apply_refinement_components()
        if _gprof:
            print('[gr-prof]   apply_refinement          {:.3f}s'.format(
                _t.perf_counter() - _gt0), flush=True)

    # -----------------------------------------------------------------
    #  Surface derivative computation (unrefined, for eval())
    # -----------------------------------------------------------------
    def _compute_F_norm(self,
            G: np.ndarray,
            Fr: np.ndarray,
            Fz: np.ndarray,
            area2: Optional[np.ndarray] = None) -> None:

        nvec1 = self.p1.nvec
        n1 = nvec1.shape[0]
        n2 = self.p2.pos.shape[0]

        r_safe = np.maximum(self._r, np.finfo(float).eps)

        # MATLAB initrefl1.m line 27-28, 38-40:
        #   in = (x*nvec_x + y*nvec_y) / max(r, 1e-10)
        #   F  = (Fr .* in + Fz .* nvec_z) .* area'
        in_prod = (self._dx * nvec1[:, 0:1] + self._dy * nvec1[:, 1:2]) / r_safe
        F = Fr * in_prod + Fz * nvec1[:, 2:3]
        if area2 is not None:
            F = F * area2[np.newaxis, :]

        self.F = F

    def _compute_F_cart(self,
            G: np.ndarray,
            Fr: np.ndarray,
            Fz: np.ndarray,
            area2: Optional[np.ndarray] = None) -> None:
        """Compute Cartesian derivative Gp and normal derivative F.

        MATLAB: initrefl2.m
        - Gp is the 3D Cartesian derivative, shape (n1, 3, n2)
        - F is the 2D normal derivative: F[i,j] = nvec[i] . Gp[i,:,j]
        """

        nvec1 = self.p1.nvec
        n1 = self.p1.pos.shape[0]
        n2 = self.p2.pos.shape[0]

        r_safe = np.maximum(self._r, np.finfo(float).eps)

        # MATLAB initrefl2.m line 37-40:
        #   Gp(:,1,:) = (Fr .* x ./ max(r,1e-10)) .* area'
        #   Gp(:,2,:) = (Fr .* y ./ max(r,1e-10)) .* area'
        #   Gp(:,3,:) =  Fz                       .* area'
        Gp = np.zeros((n1, 3, n2), dtype = complex)
        Gp[:, 0, :] = Fr * self._dx / r_safe
        Gp[:, 1, :] = Fr * self._dy / r_safe
        Gp[:, 2, :] = Fz
        if area2 is not None:
            Gp = Gp * area2[np.newaxis, np.newaxis, :]

        # Normal derivative: F = inner(nvec, Gp) (MATLAB initrefl2.m line 197)
        # F[i,j] = nvec[i,0]*Gp[i,0,j] + nvec[i,1]*Gp[i,1,j] + nvec[i,2]*Gp[i,2,j]
        F = np.einsum('ik,ikj->ij', nvec1, Gp)

        self.Gp = Gp
        self.F = F

    def setup_tabulation(self, nr = 30, nz = 20):

        z1, z2 = self.layer.round_z(self._z1, self._z2)

        # r: logarithmic (rmin -> max radial distance)
        r_max = max(self._r.max(), self.layer.rmin * 10)
        r_grid = np.geomspace(self.layer.rmin, r_max, nr)

        # z1, z2: linear (face z-coordinate range)
        z_all = np.concatenate([z1, z2])
        z_min, z_max = z_all.min(), z_all.max()
        if np.isclose(z_min, z_max):
            z_max = z_min + 1.0
        z1_grid = mlinspace(z_min, z_max, nz)
        z2_grid = mlinspace(z_min, z_max, nz)

        self.tab.setup_grid(r_grid, z1_grid, z2_grid)

    def __repr__(self) -> str:
        n_diag = len(self._diag_id) if hasattr(self, '_diag_id') else 0
        n_offdiag = len(self._offdiag_ind) if hasattr(self, '_offdiag_ind') else 0
        return 'GreenRetLayer(n1={}, n2={}, deriv={}, diag={}, offdiag={})'.format(
            self.p1.pos.shape[0], self.p2.pos.shape[0], self.deriv,
            n_diag, n_offdiag)


# =====================================================================
#  Module-level helpers
# =====================================================================

def _sparse_find(sparse_matrix):
    """Extract non-zero entries from a sparse matrix.

    Returns (rows, cols, values) like MATLAB's find().
    """
    from scipy.sparse import find as sp_find
    rows, cols, vals = sp_find(sparse_matrix)
    return rows, cols, vals


def _particle_quad(p, ind):
    """Call particle.quad() method avoiding name conflict with quad attribute.

    The Particle class has both a `quad` attribute (QuadFace data) and a
    `quad()` method.  Because instance attributes shadow methods, we call
    the private implementations directly.

    For ComParticle: use the concatenated `pc` (Particle) to match MATLAB
    @compound/quad() delegation behavior.

    Returns (pos, w_sparse, iface) matching MATLAB quad().
    """
    # ComParticle handling: delegate to concatenated Particle (pc)
    if hasattr(p, 'pc') and p.pc is not None and not hasattr(p, 'interp'):
        p = p.pc
    if p.interp == 'flat':
        return p._quad_flat(ind)
    else:
        return p._quad_curv(ind)


def _underlying_particle(p):
    """Return the concrete Particle object (handles ComParticle wrapper)."""
    if hasattr(p, 'pc') and p.pc is not None and not hasattr(p, 'interp'):
        return p.pc
    return p


def _shape_and_verts(p, face_idx):
    """Return (shape_function_values, vertex_positions) for one face.

    Mirrors MATLAB ``@greenretlayer/private/shapefunction.m`` combined with
    ``vertices(p, face)``.

    For a triangle face (3 vertices):
        s(q, :) = [x_q, y_q, 1 - x_q - y_q]      -> (m, 3)
        pos_q  = m quadrature points

    For a quadrilateral face (4 vertices):
        MATLAB maps the triangular Dunavant rule onto two sub-triangles of
        the canonical [-1, 1]^2 square (v1=(-1,-1), v2=(1,-1), v3=(1,1),
        v4=(-1,1)).  Two triangles -> 2m quadrature points.
        s(q, :) = 0.25 * [(1-u)(1-v), (1+u)(1-v), (1+u)(1+v), (1-u)(1+v)]
                  evaluated at the same (u, v) sample locations.
    """
    p = _underlying_particle(p)
    quad_table = p.quad

    faces = p.faces
    is_triangle = (faces.shape[1] < 4) or np.isnan(faces[face_idx, 3])

    if is_triangle:
        x = np.asarray(quad_table.x).ravel()
        y = np.asarray(quad_table.y).ravel()
        s = np.column_stack([x, y, 1.0 - x - y])
        vert_ids = faces[face_idx, :3].astype(int)
        verts = p.verts[vert_ids]
        return s, verts

    # Quadrilateral: reproduce MATLAB @quadface/quad([-1,-1,0], [1,-1,0],
    # [1,1,0], [-1,1,0]) which stitches together two adapted triangles.
    # Triangle 1 uses corners (v1, v2, v3) = ((-1,-1), (1,-1), (1,1)):
    #   u = x*(-1) + y*(1) + (1-x-y)*(1)  = 1 - 2x
    #   v = x*(-1) + y*(-1) + (1-x-y)*(1) = 1 - 2x - 2y   ... wait this is wrong
    # Actually adaptrule: pos = x*v1 + y*v2 + (1-x-y)*v3.
    # With v1=(-1,-1), v2=(1,-1), v3=(1,1):
    #   u = x*(-1) + y*(1)  + (1-x-y)*(1)  = 1 - 2x
    #   v = x*(-1) + y*(-1) + (1-x-y)*(1)  = 1 - 2x
    # -> u == v; that collapses.  The correct reading is:
    #   v1 -> maps x=1 (other two=0) to v1; v2 -> y=1 to v2; v3 -> (1-x-y)=1 to v3
    # With x=1 -> point is v1; y=1 -> v2; x=y=0 -> v3.  So:
    #   u = 1*x + 1*y + 1*(1-x-y)  when v1=(v1u), v2=(v2u), v3=(v3u)
    # Substituting v1=(-1,-1), v2=(1,-1), v3=(1,1):
    #   u = -1*x + 1*y + 1*(1-x-y)       = 1 - 2x
    #   v = -1*x + -1*y + 1*(1-x-y)      = 1 - 2x - 2y + x - y... recomputing:
    #   u_1 = (-1)*x + (1)*y + (1)*(1 - x - y) = -x + y + 1 - x - y = 1 - 2x
    #   v_1 = (-1)*x + (-1)*y + (1)*(1 - x - y) = -x - y + 1 - x - y = 1 - 2x - 2y
    # Triangle 2 uses (v3, v4, v1) = ((1,1), (-1,1), (-1,-1)):
    #   u_2 = (1)*x + (-1)*y + (-1)*(1 - x - y) = x - y - 1 + x + y = 2x - 1
    #   v_2 = (1)*x + (1)*y + (-1)*(1 - x - y) = x + y - 1 + x + y = 2x + 2y - 1
    x = np.asarray(quad_table.x).ravel()
    y = np.asarray(quad_table.y).ravel()
    u1 = 1.0 - 2.0 * x
    v1 = 1.0 - 2.0 * x - 2.0 * y
    u2 = 2.0 * x - 1.0
    v2 = 2.0 * x + 2.0 * y - 1.0
    u = np.concatenate([u1, u2])
    v = np.concatenate([v1, v2])

    s = 0.25 * np.column_stack([
            (1.0 - u) * (1.0 - v),
            (1.0 + u) * (1.0 - v),
            (1.0 + u) * (1.0 + v),
            (1.0 - u) * (1.0 + v),
    ])
    vert_ids = faces[face_idx, :4].astype(int)
    verts = p.verts[vert_ids]
    return s, verts


def _refine_dist(layer, pos1, pos2, nvec):
    """Compute distances used by init1.m/init2.m.

    Mirrors the private ``dist`` helpers.

    Returns
    -------
    r  : (n1, n2) radial (xy) distance, floored at layer.rmin
    z  : (n1, n2) sum of mindist-to-layer for z1 and z2 (always >= 0)
    d  : (n1, n2) sqrt(r^2 + z^2)
    in_prod : (n1, n2) inner product (x*nvec_x + y*nvec_y) / max(r, 1e-10)
    x  : (n1, n2) x-difference pos1 - pos2
    y  : (n1, n2) y-difference pos1 - pos2
    """
    pos1 = np.asarray(pos1, dtype = float)
    pos2 = np.asarray(pos2, dtype = float)

    x = pos1[:, 0:1] - pos2[:, 0].reshape(1, -1)
    y = pos1[:, 1:2] - pos2[:, 1].reshape(1, -1)

    # MATLAB init1.m/init2.m line 102-103:
    #   round( obj.layer, mindist( obj.layer, pos(:,3) ) )
    # first compute mindist (scalar distance to nearest layer), then apply
    # layer.round to enforce the minimum clearance (obj.zmin).
    zmin1, _ = layer.mindist(pos1[:, 2])
    zmin2, _ = layer.mindist(pos2[:, 2])
    zmin1 = layer.round_z(zmin1)[0]
    zmin2 = layer.round_z(zmin2)[0]
    z = zmin1[:, np.newaxis] + zmin2[np.newaxis, :]

    r = msqrt(x ** 2 + y ** 2)
    r = np.maximum(r, layer.rmin)
    d = msqrt(r ** 2 + z ** 2)

    r_safe = np.maximum(r, 1e-10)
    in_prod = (x * nvec[:, 0:1] + y * nvec[:, 1:2]) / r_safe

    return r, z, d, in_prod, x, y
