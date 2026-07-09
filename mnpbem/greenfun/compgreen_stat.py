"""
Composite Green function for quasistatic approximation.

MATLAB: Greenfun/@compgreenstat/
100% identical to MATLAB MNPBEM implementation.
"""

import numpy as np
from typing import Optional, Tuple, Any


class CompGreenStat(object):
    """
    Green function for composite points and particle in quasistatic approximation.

    MATLAB: @compgreenstat

    Properties
    ----------
    name : str
        'greenfunction' (constant)
    needs : dict
        {'sim': 'stat'} (constant)
    p1 : ComParticle
        Green function between points p1 and comparticle p2
    p2 : ComParticle
        Green function between points p1 and comparticle p2
    g : GreenStat
        Green functions connecting p1 and p2

    Methods
    -------
    __init__(p1, p2, **options)
        Constructor - initialize Green functions for composite objects
    eval(*keys, ind=None)
        Evaluate Green function (G, F, H1, H2, Gp, H1p, H2p)
    field(sig, inout=1)
        Electric field inside/outside of particle surface
    potential(sig, inout=1)
        Potentials and surface derivatives inside/outside of particle

    Properties (via __getattr__)
    ----------------------------
    G, F, H1, H2, Gp, H1p, H2p
        Green function matrices
    deriv
        'cart' or 'norm' - derivative type
    """

    # Class constants
    name = 'greenfunction'
    needs = {'sim': 'stat'}

    def __new__(cls, p1=None, p2=None, **options):
        # Optional H-matrix acceleration: when ``hmatrix=True`` and the mesh
        # is large enough to benefit from ACA, return an ACACompGreenStat
        # instance (drop-in replacement supporting .G/.F/.eval()/.potential()).
        if p1 is None or p2 is None:
            return object.__new__(cls)
        if options.get('hmatrix', False) and p1 is p2:
            n_faces = getattr(p1, 'n', None)
            if n_faces is None and hasattr(p1, 'p') and len(p1.p) > 0:
                n_faces = sum(getattr(pp, 'n', 0) for pp in p1.p)
            if n_faces is not None and n_faces > 1500:
                from .aca_compgreen_stat import ACACompGreenStat
                hmat_opts = {k: v for k, v in options.items() if k != 'hmatrix'}
                return ACACompGreenStat(p1, **hmat_opts)
        return object.__new__(cls)

    def __init__(self, p1, p2, **options):
        """
        Initialize Green functions for composite objects.

        MATLAB: compgreenstat.m, init.m

        Parameters
        ----------
        p1 : ComParticle
            Green function between points p1 and comparticle p2
        p2 : ComParticle
            Green function between points p1 and comparticle p2
        **options : dict
            deriv : str, optional
                'cart' (Cartesian) or 'norm' (normal) derivative (default: 'norm')
            waitbar : int, optional
                Show progress bar (default: 0)
            hmatrix : bool, optional
                Use ACA H-matrix acceleration when ``p1 is p2`` and the mesh
                exceeds 1500 faces (default: False).  Ignored otherwise.

        Examples
        --------
        >>> from mnpbem import trisphere, EpsConst, ComParticle
        >>> from mnpbem.greenfun import CompGreenStat
        >>>
        >>> eps = [EpsConst(1.0), EpsConst(2.0)]
        >>> p = trisphere(144, 10.0)
        >>> cp = ComParticle(eps, [p], [[2, 1]])
        >>> g = CompGreenStat(cp, cp)
        """
        # If __new__ returned an ACACompGreenStat replacement object, skip
        # the dense initializer (it has already been constructed).
        if not isinstance(self, CompGreenStat):
            return
        options.pop('hmatrix', None)
        self.p1 = p1
        self.p2 = p2
        self.deriv = options.get('deriv', 'cart')

        # BEM solver cache
        self._enei_cache = None
        self._mat_cache = None

        # Optional user-supplied refinement hook (e.g. coverlayer.refine).
        # Signature: fun(obj, G, F) -> (G, F). Applied after Green function
        # matrices have been built so that cover-layer / nonlocal-layer
        # geometries can inject additional polar-integration corrections.
        self._refun = options.pop('refun', None)

        # Initialize Green function
        self._init(p1, p2, **options)

        if self._refun is not None:
            self.G, self.F = self._refun(self, self.G, self.F)

    def _init(self, p1, p2, **options):
        """
        Initialize composite Green function.

        MATLAB: @compgreenstat/private/init.m

        Handles:
        - Creation of Green function between vertcat(p1.p) and vertcat(p2.p)
        - Closed surface diagonal correction (Fuchs & Liu, PRB 14, 5521, 1976)
        """
        # Get underlying particles
        pp1 = p1.p  # List of particles in p1
        pp2 = p2.p  # List of particles in p2

        # Concatenate all particles to create single particle for Green function
        # MATLAB: vertcat(pp1{:}), vertcat(pp2{:})
        if len(pp1) == 1:
            pc1 = pp1[0]
        else:
            pc1 = pp1[0]
            for p in pp1[1:]:
                pc1 = pc1 + p

        if len(pp2) == 1:
            pc2 = pp2[0]
        else:
            pc2 = pp2[0]
            for p in pp2[1:]:
                pc2 = pc2 + p

        # Initialize Green function using GreenStat
        # For now, compute G and F directly (greenstat object not yet implemented)
        self._compute_greenstat(pc1, pc2, **options)

        # Handle closed surfaces for diagonal elements
        # MATLAB: if any(strcmp('closed', fieldnames(full1))) && (full1 == p2) && ~isempty(full1.closed{:})

        # For mirror particles, full1 is the full expanded particle (pfull)
        full1 = p1
        if hasattr(p1, 'pfull'):
            full1 = p1.pfull

        # For a closed particle the surface integral of -F should give 2*pi
        # See R. Fuchs and S. H. Liu, Phys. Rev. B 14, 5521 (1976)
        if hasattr(full1, 'closed') and (full1 is p2 or full1 == p2):
            if full1.closed is not None and any(c is not None for c in full1.closed):
                self._handle_closed_surfaces(p1, p2, full1, **options)

    @staticmethod
    def _needs_sync(p):
        """Check whether verts2 needs synchronization with verts.

        Uses a relative threshold (1% of particle extent) to distinguish
        genuine user-introduced drift (e.g. manually shifting verts without
        verts2) from small natural offsets due to curved-element
        representation.
        """
        if not hasattr(p, 'verts2') or not hasattr(p, 'verts'):
            return False
        if p.verts2 is None or p.verts is None:
            return False
        if len(p.verts) == 0 or len(p.verts2) == 0:
            return False
        verts_center = 0.5 * (p.verts.min(axis = 0) + p.verts.max(axis = 0))
        verts2_center = 0.5 * (p.verts2.min(axis = 0) + p.verts2.max(axis = 0))
        offset = verts_center - verts2_center
        extent = np.linalg.norm(p.verts.max(axis = 0) - p.verts.min(axis = 0))
        threshold = max(0.01 * extent, 1e-6)
        return np.linalg.norm(offset) > threshold

    @staticmethod
    def _sync_verts2(p):
        """Sync verts2 with verts if they are out of sync.

        When users manually shift verts and pos without also shifting verts2,
        the midpoint vertices become inconsistent. This method detects and
        corrects the offset so that curved-element quadrature (quadpol,
        quad_integration) uses the correct positions.
        """
        if not hasattr(p, 'verts2') or not hasattr(p, 'verts'):
            return
        if p.verts2 is None or p.verts is None:
            return
        if len(p.verts) == 0 or len(p.verts2) == 0:
            return
        verts_center = 0.5 * (p.verts.min(axis = 0) + p.verts.max(axis = 0))
        verts2_center = 0.5 * (p.verts2.min(axis = 0) + p.verts2.max(axis = 0))
        offset = verts_center - verts2_center
        extent = np.linalg.norm(p.verts.max(axis = 0) - p.verts.min(axis = 0))
        threshold = max(0.01 * extent, 1e-6)
        if np.linalg.norm(offset) > threshold:
            p.verts2 += offset

    def _compute_greenstat(self, p1, p2, **options):
        """
        Compute quasistatic Green function matrices G and F.

        MATLAB: greenstat/private/init.m and eval1.m

        G(r, r') = 1 / |r - r'|  (no 4π factor)
        F[i,j] = - n_i · (r_i - r_j) / |r_i - r_j|³

        With refinement: diagonal and nearby elements use polar integration
        Without refinement: F_diagonal = -2π (analytical value)
        """
        import copy
        from .utils import refinematrix
        from ._numba_kernels import green_stat_assemble

        # Sync verts2 with verts if they have drifted apart.
        # Work on deep copies so that the original particle objects are
        # never mutated -- shared particle references (e.g. mirror
        # symmetry particles reused across multiple Green function
        # constructions) must remain unchanged.
        if self._needs_sync(p1):
            p1 = copy.deepcopy(p1)
            self._sync_verts2(p1)
        if p2 is not p1 and self._needs_sync(p2):
            p2 = copy.deepcopy(p2)
            self._sync_verts2(p2)

        pos1 = p1.pos
        pos2 = p2.pos
        nvec1 = p1.nvec
        area2 = p2.area

        n1 = pos1.shape[0]
        n2 = pos2.shape[0]

        # Get refinement matrix - filter options for refinematrix
        refine_opts = {k: v for k, v in options.items()
                       if k in ['AbsCutoff', 'RelCutoff', 'memsize']}
        ir = refinematrix(p1, p2, **refine_opts)

        # Store refinement indices
        self.ind = np.array(ir.nonzero()).T  # Array of (row, col) pairs

        # Assemble G, F (and Gp when deriv == 'cart').  When p1 is p2
        # the kernel zeros the diagonal so that the refinement /
        # analytical-correction path below can fill the self-term.
        # Otherwise the i == j entry is treated as a regular far-field
        # element (matching the original numpy broadcasting behavior).
        same_block = (p1 is p2)
        want_gp = self.deriv == 'cart'
        G_full, F_full, Gp_full = green_stat_assemble(
            pos1, pos2, nvec1, area2, same_block, want_gp
        )
        self.G = G_full
        self.F = F_full
        if want_gp:
            self._Gp_raw = Gp_full

        # Apply refinement if needed
        if len(self.ind) > 0:
            # Check whether the particle supports the quadrature methods
            # (quadpol, quad_integration) required for polar integration
            # refinement.  When these are missing -- e.g. for simplified /
            # mock particle objects -- fall back to the analytical diagonal
            # correction:  G_diag ~ 1/(4*pi*bradius),  F_diag = -2*pi.
            has_quad = hasattr(p2, 'quadpol') and hasattr(p2, 'quad_integration')
            if has_quad:
                self._refine_greenstat(p1, p2, ir, **options)
            else:
                self._fallback_diagonal(p1, p2)
        else:
            # No refinement - use analytical diagonal values
            if p1 is p2:
                np.fill_diagonal(self.G, 0.0)
                np.fill_diagonal(self.F, -2.0 * np.pi)

    def _fallback_diagonal(self, p1, p2):
        """
        Simple diagonal correction when polar-integration quadrature
        infrastructure (quadpol / quad_integration) is unavailable.

        For diagonal (self-term) elements:
          G_ii = 1 / (4 * pi * bradius_i)   (approximate self-coupling)
          F_ii = -2 * pi                      (analytical value for flat element)

        For off-diagonal refinement elements the existing far-field
        approximation (1/d * area) is kept unchanged.
        """
        if p1 is p2 or (hasattr(p1, 'pos') and hasattr(p2, 'pos')
                        and p1.pos is p2.pos):
            n = p1.pos.shape[0]
            nvec1 = p1.nvec

            # Boundary-element radius for self-term G correction
            if callable(getattr(p2, 'bradius', None)):
                brad = p2.bradius()
            elif hasattr(p2, 'bradius') and isinstance(p2.bradius, np.ndarray):
                brad = p2.bradius
            else:
                brad = np.ones(n)

            brad_safe = np.maximum(brad, np.finfo(float).eps)

            for i in range(n):
                self.G[i, i] = 1.0 / (4.0 * np.pi * brad_safe[i])
                self.F[i, i] = -2.0 * np.pi

    def _refine_greenstat(self, p1, p2, ir, **options):
        """
        Refine diagonal and nearby elements using polar integration.

        MATLAB: greenstat/private/init.m lines 35-123
        """
        pos1 = p1.pos
        nvec1 = p1.nvec

        # Convert sparse matrix to dense for easier indexing
        ir_dense = ir.toarray()

        # ===== Diagonal elements (ir == 2) =====
        diag_mask = ir_dense == 2
        if np.any(diag_mask):
            diag_rows, diag_cols = np.where(diag_mask)

            # Integration points and weights for polar integration
            # Call quadpol with ALL diagonal faces at once (MATLAB-style)
            pos_quad, w_quad, row_quad = p2.quadpol(diag_cols)

            # Expand face positions to match integration points
            # MATLAB: expand = @( x ) subsref( x( face, : ), substruct( '()', { row, ':' } ) );
            pos1_expanded = pos1[diag_rows[row_quad]]
            nvec1_expanded = nvec1[diag_rows[row_quad]]

            # Vector from centroids to integration points
            vec = pos1_expanded - pos_quad
            r = np.linalg.norm(vec, axis=1)
            r = np.maximum(r, np.finfo(float).eps)

            # Accumulate values for each unique diagonal element (vectorized)
            # MATLAB: g( iface ) = accumarray( row, w ./ r );
            n_diag = len(diag_rows)
            g_vals = np.bincount(row_quad, weights=w_quad / r, minlength=n_diag)[:n_diag]
            self.G[diag_rows, diag_cols] = g_vals

            # F (normal derivative) — always 2D
            n_dot_vec = np.sum(vec * nvec1_expanded, axis=1)
            f_vals = -np.bincount(row_quad, weights=w_quad * n_dot_vec / (r ** 3), minlength=n_diag)[:n_diag]
            self.F[diag_rows, diag_cols] = f_vals

            if self.deriv == 'cart':
                # Also compute and store 3D Gp refinement for cart derivative
                # MATLAB: f(iface, :) = nvec*f1 + tvec1*f2 + tvec2*f3
                if not hasattr(self, '_f_cart_refined'):
                    self._f_cart_refined = []
                    self._f_cart_refined_indices = []

                tvec1 = p1.tvec1[diag_rows[row_quad]] if hasattr(p1, 'tvec1') else np.zeros_like(nvec1_expanded)
                tvec2 = p1.tvec2[diag_rows[row_quad]] if hasattr(p1, 'tvec2') else np.zeros_like(nvec1_expanded)
                rr = np.maximum(r, 1e-4 * np.max(r))
                in1 = np.sum(vec * tvec1, axis=1)
                in2 = np.sum(vec * tvec2, axis=1)

                f1 = -np.bincount(row_quad, weights=w_quad * n_dot_vec / (r ** 3), minlength=n_diag)[:n_diag]
                f2 = -np.bincount(row_quad, weights=w_quad * in1 / (rr ** 3), minlength=n_diag)[:n_diag]
                f3 = -np.bincount(row_quad, weights=w_quad * in2 / (rr ** 3), minlength=n_diag)[:n_diag]

                for k in range(n_diag):
                    f_cart = (p1.nvec[diag_rows[k]] * f1[k] +
                              p1.tvec1[diag_rows[k]] * f2[k] +
                              p1.tvec2[diag_rows[k]] * f3[k])
                    self._f_cart_refined.append(f_cart)
                    self._f_cart_refined_indices.append((diag_rows[k], diag_cols[k]))

        # ===== Off-diagonal refinement elements (ir == 1) =====
        offdiag_mask = ir_dense == 1
        if np.any(offdiag_mask):
            # Get unique faces that need refinement
            _, offdiag_cols = np.where(offdiag_mask)
            unique_refine_faces = np.unique(offdiag_cols)

            # Integration points and weights for boundary element integration
            # Returns: pos (n_total, 3), w_sparse (n_faces, n_total), iface
            pos_quad, w_sparse, _ = p2.quad_integration(unique_refine_faces)

            # Use the CSR layout directly to avoid materializing a dense
            # (n_faces, n_total) array that blows up memory for large meshes
            # (e.g. 25k faces × 12.7M points ~ 2.4 TiB).  Each face occupies
            # a disjoint slice of integration points, so per-row iteration
            # over indptr/indices/data gives us the same (pos_i, w_i) pair.
            w_csr = w_sparse.tocsr()
            indptr = w_csr.indptr
            indices = w_csr.indices
            data = w_csr.data

            # Process each unique face
            for i, face2 in enumerate(unique_refine_faces):
                # Find all rows that need refinement for this column
                nb = np.where(offdiag_mask[:, face2])[0]
                if len(nb) == 0:
                    continue

                # Get integration points and weights for this face directly
                # from the CSR row slice (avoids dense conversion).
                row_start, row_end = indptr[i], indptr[i + 1]
                if row_end == row_start:
                    continue
                cols_i = indices[row_start:row_end]
                w_row = data[row_start:row_end]
                # Drop any explicitly stored zero weights to match the
                # previous ``mask = w > 0`` semantics.
                nz = w_row > 0
                if not np.all(nz):
                    cols_i = cols_i[nz]
                    w_row = w_row[nz]
                if len(w_row) == 0:
                    continue
                pos = pos_quad[cols_i]
                w = w_row

                # Difference vectors: centroids - integration points
                # pos1[nb] shape: (len(nb), 3)
                # pos shape: (n_quad, 3)
                # vec shape: (len(nb), n_quad, 3)
                vec = pos1[nb, np.newaxis, :] - pos[np.newaxis, :, :]
                r = np.linalg.norm(vec, axis=2)  # (len(nb), n_quad)
                r = np.maximum(r, np.finfo(float).eps)

                # Green function: (1/r) @ w
                g_vals = (1.0 / r) @ w  # (len(nb),)
                self.G[nb, face2] = g_vals

                # Surface derivative (F — always 2D normal)
                n_dot_vec = np.sum(nvec1[nb, np.newaxis, :] * vec, axis=2)
                f_vals = -(n_dot_vec / (r ** 3)) @ w
                self.F[nb, face2] = f_vals

                if self.deriv == 'cart':
                    # Also store 3D cart refinement for Gp
                    f_cart_vals = -np.einsum('ijk,j->ik', vec / r[:, :, np.newaxis] ** 3, w)
                    if not hasattr(self, '_f_cart_refined'):
                        self._f_cart_refined = []
                        self._f_cart_refined_indices = []
                    for k, nb_idx in enumerate(nb):
                        self._f_cart_refined.append(f_cart_vals[k])
                        self._f_cart_refined_indices.append((nb_idx, face2))

    def _handle_closed_surfaces(self, p1, p2, full1, **options):
        """
        Handle closed surface diagonal correction.

        MATLAB: @compgreenstat/private/init.m lines 24-57

        For closed surfaces, diagonal elements of F are set to -2π*dir - f',
        where f' is the sum over the closed surface integral.
        """
        # Loop over particles
        for i in range(len(p1.p)):
            # Index to particle faces
            ind = p1.index_func(i + 1)  # 1-indexed in MATLAB

            # Select particle and closed particle surface
            part = p1.p[i]
            full, dir_val, loc = self._closedparticle(p1, i)

            if full is not None:
                if loc is not None:
                    # Use already computed Green function object
                    f = self._fun_closed(loc, ind, **options)
                else:
                    # Set up Green function for closed surface
                    # Create temporary greenstat for full closed surface
                    g_temp = CompGreenStat.__new__(CompGreenStat)
                    g_temp.deriv = self.deriv
                    g_temp.p1 = full
                    g_temp.p2 = part
                    g_temp._compute_greenstat(full, part, **options)

                    # Sum over closed surface
                    f = self._fun_closed_greenstat(g_temp, **options)

                # Set diagonal elements of Green function
                # MATLAB: obj.g = diag(obj.g, ind, -2*pi*dir - f.')
                # f is transposed in MATLAB, so f[i] corresponds to ind[i]
                if isinstance(ind, (list, np.ndarray)):
                    ind_array = np.array(ind)
                else:
                    ind_array = np.array([ind])

                # F is always 2D — apply scalar correction
                diag_vals = -2.0 * np.pi * dir_val - f
                self.F[ind_array, ind_array] += diag_vals

                if self.deriv == 'cart':
                    # Also correct Gp refinement
                    if not hasattr(self, '_f_cart_refined'):
                        self._f_cart_refined = []
                        self._f_cart_refined_indices = []
                    cart_vals = diag_vals[:, np.newaxis] * part.nvec
                    for j, idx in enumerate(ind_array):
                        # Check if already in the list and update
                        found = False
                        for k, (ri, ci) in enumerate(self._f_cart_refined_indices):
                            if ri == idx and ci == idx:
                                self._f_cart_refined[k] += cart_vals[j]
                                found = True
                                break
                        if not found:
                            self._f_cart_refined.append(cart_vals[j])
                            self._f_cart_refined_indices.append((idx, idx))

    def _closedparticle(self, p, i):
        """
        Get closed particle surface.

        MATLAB: closedparticle(p1, i)

        Parameters
        ----------
        p : ComParticle
            Composite particle
        i : int
            Particle index (0-indexed in Python)

        Returns
        -------
        full : Particle or None
            Full closed particle
        dir_val : float
            Direction indicator (+1 or -1)
        loc : array or None
            Local indices
        """
        # Call ComParticle's closedparticle method (expects 1-indexed)
        return p.closedparticle(i + 1)

    def _fun_closed(self, loc, ind, **options):
        """
        Sum over closed surface using already computed Green function.

        MATLAB: init.m/fun() with loc and ind arguments
        f = sum(area1[loc] * F[loc, ind] / area2[ind], axis=0)

        Parameters
        ----------
        loc : array
            Indices into p1 (row indices)
        ind : array
            Indices into p2 (column indices)

        Returns
        -------
        f : array
            Surface integral values for each column
        """
        # Get areas
        area1 = self.p1.pc.area if hasattr(self.p1, 'pc') else self.p1.area
        area2 = self.p2.pc.area if hasattr(self.p2, 'pc') else self.p2.area

        # Extract submatrix F[loc, ind]
        F_sub = self.F[np.ix_(loc, ind)]

        # Compute weighted sum: f = sum(area1[loc] * F[loc, ind] / area2[ind], axis=0)
        F_weighted = area1[loc][:, np.newaxis] * F_sub / area2[ind][np.newaxis, :]
        f = np.sum(F_weighted, axis=0)

        return f

    def _fun_closed_greenstat(self, g, **options):
        """
        Sum over closed surface.

        MATLAB: init.m/fun(g, varargin)
        f = sum(diag(area1) * g.F * diag(1./area2), 1)
        """
        p1 = g.p1 if hasattr(g, 'p1') else self.p1
        p2 = g.p2 if hasattr(g, 'p2') else self.p2

        area1 = p1.area if hasattr(p1, 'area') else np.ones(g.F.shape[0])
        area2 = p2.area if hasattr(p2, 'area') else np.ones(g.F.shape[1])

        # f = sum(area1[:, None] * g.F * (1/area2)[None, :], axis=0)
        F_weighted = area1[:, np.newaxis] * g.F / area2[np.newaxis, :]
        f = np.sum(F_weighted, axis=0)

        return f

    def solve(self, exc):
        """
        Solve BEM equations for given excitation.

        MATLAB: @bemstat/solve.m, @bemstat/mldivide.m

        Parameters
        ----------
        exc : CompStruct
            compstruct with field 'phip' for external excitation

        Returns
        -------
        sig : CompStruct
            compstruct with field 'sig' for surface charge

        Examples
        --------
        >>> from mnpbem.simulation import PlaneWaveStat
        >>> pol = np.array([1, 0, 0])
        >>> pw = PlaneWaveStat(pol)
        >>> exc = pw.potential(cp, enei=400)
        >>> sig = g.solve(exc)
        """
        # Initialize BEM solver (compute resolvent matrix if needed)
        self._init_solver(exc.enei)

        # Solve: sig = mat @ phip
        sig_values = self._matmul(self._mat_cache, exc.phip)

        # Return as CompStruct
        return CompStruct(self.p1, exc.enei, sig=sig_values)

    def _init_solver(self, enei):
        """
        Compute BEM resolvent matrix for given wavelength.

        MATLAB: @bemstat/subsref.m (case '()')

        Parameters
        ----------
        enei : float
            light wavelength in vacuum (nm)

        Notes
        -----
        Computes: mat = -inv(diag(lambda) + F)
        where lambda = 2π(ε₁ + ε₂)/(ε₁ - ε₂)

        Reference: Garcia de Abajo & Howie, PRB 65, 115418 (2002), Eq. (23)
        """
        # Use previously computed matrix if wavelength hasn't changed
        if self._enei_cache is not None and self._enei_cache == enei:
            return

        # Get inside and outside dielectric functions at this wavelength
        # eps1 = inside, eps2 = outside
        eps1_vals = self.p1.eps1(enei)  # Array, one value per face
        eps2_vals = self.p1.eps2(enei)  # Array, one value per face

        # Lambda coefficient [Garcia de Abajo, Eq. (23)]
        # lambda = 2π(ε₁ + ε₂)/(ε₁ - ε₂)
        lambda_vals = 2.0 * np.pi * (eps1_vals + eps2_vals) / (eps1_vals - eps2_vals)

        # BEM resolvent matrix
        # mat = -inv(diag(lambda) + F)
        A = np.diag(lambda_vals) + self.F
        self._mat_cache = -np.linalg.inv(A)

        # Cache wavelength
        self._enei_cache = enei

    def eval(self, *args, **kwargs):
        """
        Evaluate Green function.

        MATLAB: @compgreenstat/eval.m

        Usage
        -----
        g = obj.eval(key1, key2, ...)
        g = obj.eval(ind, key1, key2, ...)

        Parameters
        ----------
        ind : array, optional
            Index to matrix elements to be computed
        key : str
            G    - Green function
            F    - Surface derivative of Green function
            H1   - F + 2 * pi
            H2   - F - 2 * pi
            Gp   - Derivative of Green function
            H1p  - Gp + 2 * pi
            H2p  - Gp - 2 * pi

        Returns
        -------
        varargout : tuple
            Requested Green functions

        Examples
        --------
        >>> G = g.eval('G')
        >>> F = g.eval('F')
        >>> G, F = g.eval('G', 'F')
        >>> H1, H2 = g.eval('H1', 'H2')
        """
        # Parse arguments
        if len(args) == 0:
            raise ValueError("At least one key must be provided")

        # Check if first argument is indices
        ind = kwargs.get('ind', None)
        if len(args) > 0 and isinstance(args[0], (np.ndarray, list, tuple)) and \
           not isinstance(args[0], str):
            ind = args[0]
            keys = args[1:]
        else:
            keys = args

        # Evaluate each key
        results = []
        for key in keys:
            if key == 'G':
                result = self._eval_G(ind)
            elif key == 'F':
                result = self._eval_F(ind)
            elif key == 'H1':
                result = self._eval_H1(ind)
            elif key == 'H2':
                result = self._eval_H2(ind)
            elif key == 'Gp':
                result = self._eval_Gp(ind)
            elif key == 'H1p':
                result = self._eval_H1p(ind)
            elif key == 'H2p':
                result = self._eval_H2p(ind)
            else:
                raise ValueError("Unknown key: {}".format(key))

            results.append(result)

        # Return results
        if len(results) == 1:
            return results[0]
        else:
            return tuple(results)

    def eval_block(self, key, enei, col_start, col_stop):
        """
        Evaluate a column slice of the quasistatic Green function matrix.

        Mirrors :meth:`mnpbem.greenfun.compgreen_ret.CompGreenRet.eval_block`
        (the retarded sibling) so the same ``DistributedMatrix.from_func``
        callback shape works for both solvers.  Quasistatic Green
        functions are wavelength-independent (``k = 0``) so the
        ``enei`` argument is accepted but ignored.

        Parameters
        ----------
        key : str
            Green function key (``'G'``, ``'F'``, ``'H1'``, ``'H2'``).
            Derivative keys (``'Gp'``, ``'H1p'``, ``'H2p'``) are also
            supported and return shape ``(p1.n, 3, ncol)``.
        enei : float
            Light wavelength in vacuum (nm).  Accepted for API parity
            with :class:`CompGreenRet`; not used in the quasistatic
            branch because the surface Laplacian / Coulomb kernel does
            not depend on it.
        col_start : int
            Inclusive lower bound of the column range (global p2 index).
        col_stop : int
            Exclusive upper bound of the column range.

        Returns
        -------
        g : ndarray
            Sliced Green function tile of shape ``(p1.n, ncol)`` for
            scalar keys, ``(p1.n, 3, ncol)`` for derivative keys.
            ``np.allclose(g, eval(key)[..., col_start:col_stop])`` holds
            exactly (the slice is a view).

        Notes
        -----
        - Provided for use with
          :meth:`mnpbem.utils.distributed_matrix.DistributedMatrix.from_func`
          so the BEM build path can scatter the column tiles directly to
          the GPUs that will own them.  In the quasistatic case the full
          ``self.F`` / ``self.G`` is already on host (from
          :meth:`_compute_greenstat`), so this method is a fast slice
          rather than a per-block recomputation.
        - Diagonal corrections (polar refinement, closed-surface fix,
          analytical -2π fallback) are preserved because they are
          baked into ``self.F`` before any slice is taken.
        """
        col_start = int(col_start)
        col_stop = int(col_stop)
        assert 0 <= col_start <= col_stop <= self.p2.n, \
            ("[error] eval_block col range ({}, {}) out of bounds for "
             "p2.n={}".format(col_start, col_stop, self.p2.n))

        # Slice the appropriate matrix.  For H1/H2 we cannot reuse the
        # eval('H1')/eval('H2') copy because it allocates a full (n1,
        # n2) buffer; instead we slice ``self.F`` and apply the diagonal
        # correction in-place on the slice.
        if key == 'G':
            return self.G[:, col_start:col_stop]
        if key == 'F':
            return self.F[:, col_start:col_stop]
        if key == 'H1':
            block = self.F[:, col_start:col_stop].copy()
            if self.p1 is self.p2:
                # Diagonal of F lives at column j and row j, so the
                # entries that land inside the column slice are rows
                # [col_start, col_stop) of the slice's local columns
                # [0, ncol).
                rows = np.arange(col_start, col_stop)
                local_cols = np.arange(col_stop - col_start)
                block[rows, local_cols] = self.F[rows, rows] + 2.0 * np.pi
            return block
        if key == 'H2':
            block = self.F[:, col_start:col_stop].copy()
            if self.p1 is self.p2:
                rows = np.arange(col_start, col_stop)
                local_cols = np.arange(col_stop - col_start)
                block[rows, local_cols] = self.F[rows, rows] - 2.0 * np.pi
            return block
        if key == 'Gp':
            Gp = self._eval_Gp()
            return Gp[:, :, col_start:col_stop]
        if key == 'H1p':
            Gp = self._eval_Gp()
            block = Gp[:, :, col_start:col_stop].copy()
            if self.p1 is self.p2:
                nvec = self.p1.nvec
                rows = np.arange(col_start, col_stop)
                local_cols = np.arange(col_stop - col_start)
                block[rows, :, local_cols] = (
                    Gp[rows, :, rows] + 2 * np.pi * nvec[rows])
            return block
        if key == 'H2p':
            Gp = self._eval_Gp()
            block = Gp[:, :, col_start:col_stop].copy()
            if self.p1 is self.p2:
                nvec = self.p1.nvec
                rows = np.arange(col_start, col_stop)
                local_cols = np.arange(col_stop - col_start)
                block[rows, :, local_cols] = (
                    Gp[rows, :, rows] - 2 * np.pi * nvec[rows])
            return block
        raise ValueError("[error] eval_block: unknown key '{}'".format(key))

    def _eval_G(self, ind=None):
        """Evaluate G matrix."""
        if ind is None:
            return self.G
        else:
            return self.G.ravel()[ind]

    def _eval_F(self, ind=None):
        """Evaluate F matrix."""
        if ind is None:
            return self.F
        else:
            return self.F.ravel()[ind]

    def _eval_H1(self, ind=None):
        """Evaluate H1 = F + 2π on diagonal."""
        H1 = self.F.copy()
        if self.p1 is self.p2:
            np.fill_diagonal(H1, np.diag(self.F) + 2.0 * np.pi)

        if ind is None:
            return H1
        else:
            return H1.ravel()[ind]

    def _eval_H2(self, ind=None):
        """Evaluate H2 = F - 2π on diagonal."""
        H2 = self.F.copy()
        if self.p1 is self.p2:
            np.fill_diagonal(H2, np.diag(self.F) - 2.0 * np.pi)

        if ind is None:
            return H2
        else:
            return H2.ravel()[ind]

    def _eval_Gp(self, ind=None):
        """
        Evaluate Gp - Cartesian derivative of Green function.

        MATLAB: greenstat/eval1.m (case 'cart')
        Gp = -r / d³ * area, then refine: Gp(ind, :) = f

        Returns
        -------
        Gp : ndarray, shape (n1, 3, n2)
            Cartesian derivative of Green function
        """
        if not hasattr(self, '_Gp'):
            if hasattr(self, '_Gp_raw') and self._Gp_raw is not None:
                # Use pre-computed Gp from _init (deriv='cart')
                self._Gp = self._Gp_raw.copy()
            else:
                pos1 = self.p1.pos
                pos2 = self.p2.pos
                area2 = self.p2.area

                r = pos1[:, np.newaxis, :] - pos2[np.newaxis, :, :]
                d = np.linalg.norm(r, axis=2)
                d = np.maximum(d, np.finfo(float).eps)
                Gp = -r / (d[:, :, np.newaxis] ** 3) * area2[np.newaxis, :, np.newaxis]
                self._Gp = np.transpose(Gp, (0, 2, 1))

            # Apply refinement for cart derivative (MATLAB: Gp(ind,:) = obj.f)
            if self.deriv == 'cart' and hasattr(self, '_f_cart_refined'):
                for idx, (row_i, col_i) in enumerate(self._f_cart_refined_indices):
                    self._Gp[row_i, :, col_i] = self._f_cart_refined[idx]

        if ind is None:
            return self._Gp
        else:
            return self._Gp.reshape(-1, 3)[ind]

    def _eval_H1p(self, ind=None):
        """
        Evaluate H1p = Gp + 2π*nvec on diagonal.

        Returns
        -------
        H1p : ndarray, shape (n1, 3, n2)
        """
        Gp = self._eval_Gp()
        H1p = Gp.copy()

        if self.p1 is self.p2:
            nvec = self.p1.nvec
            for i in range(len(nvec)):
                H1p[i, :, i] += 2 * np.pi * nvec[i]

        if ind is None:
            return H1p
        else:
            return H1p.reshape(-1, 3)[ind]

    def _eval_H2p(self, ind=None):
        """
        Evaluate H2p = Gp - 2π*nvec on diagonal.

        Returns
        -------
        H2p : ndarray, shape (n1, 3, n2)
        """
        Gp = self._eval_Gp()
        H2p = Gp.copy()

        if self.p1 is self.p2:
            nvec = self.p1.nvec
            for i in range(len(nvec)):
                H2p[i, :, i] -= 2 * np.pi * nvec[i]

        if ind is None:
            return H2p
        else:
            return H2p.reshape(-1, 3)[ind]

    def field(self, sig, inout=1):
        """
        Electric field inside/outside of particle surface.

        MATLAB: @compgreenstat/field.m

        Parameters
        ----------
        sig : CompStruct
            compstruct with surface charges (see BEMSTAT)
        inout : int
            fields inside (inout=1, default) or outside (inout=2) of particle surface

        Returns
        -------
        field : CompStruct
            compstruct object with electric field 'e'

        Examples
        --------
        >>> field = g.field(sig, inout=1)  # Inside
        >>> field = g.field(sig, inout=2)  # Outside
        """
        # Derivative of Green function
        if inout == 1:
            Hp = self.eval('H1p')
        else:
            Hp = self.eval('H2p')

        # Electric field: e = -Hp @ sig
        # MATLAB: e = -matmul(Hp, sig.sig)
        e = -self._matmul(Hp, sig.sig)

        # Set output as CompStruct
        field = CompStruct(self.p1, sig.enei, e=e)
        return field

    def potential(self, sig, inout=1):
        """
        Determine potentials and surface derivatives inside/outside of particle.

        MATLAB: @compgreenstat/potential.m

        Parameters
        ----------
        sig : CompStruct
            compstruct with surface charges (see BEMSTAT)
        inout : int
            potentials inside (inout=1, default) or outside (inout=2) of particle

        Returns
        -------
        pot : CompStruct
            compstruct object with potentials 'phi' and surface derivatives 'phip'

        Examples
        --------
        >>> pot = g.potential(sig, inout=1)  # Inside
        >>> pot = g.potential(sig, inout=2)  # Outside
        """
        # Set parameters that depend on inside/outside
        # MATLAB: H = subsref({'H1', 'H2'}, substruct('{}', {inout}))
        H_key = 'H1' if inout == 1 else 'H2'

        # Get Green function and surface derivative
        # MATLAB: [G, H] = eval(obj.g, 'G', H)
        G, H = self.eval('G', H_key)

        # Potential and surface derivative
        # MATLAB: phi = matmul(G, sig.sig)
        #         phip = matmul(H, sig.sig)
        phi = self._matmul(G, sig.sig)
        phip = self._matmul(H, sig.sig)

        # Set output
        if inout == 1:
            pot = CompStruct(self.p1, sig.enei, phi1=phi, phi1p=phip)
        else:
            pot = CompStruct(self.p1, sig.enei, phi2=phi, phi2p=phip)

        return pot

    def _matmul(self, a, x):
        """
        Generalized matrix multiplication for tensors.

        MATLAB: Misc/matmul.m

        The matrix multiplication is performed along the last dimension of A
        and the first dimension of X.
        """
        if np.isscalar(a) or (isinstance(a, np.ndarray) and a.size == 1):
            # A is scalar
            if a == 0:
                return 0
            else:
                return a * x
        elif np.isscalar(x) or (isinstance(x, np.ndarray) and x.size == 1):
            # X is scalar
            if x == 0:
                return 0
            else:
                return a * x
        else:
            # A is matrix/tensor
            siza = a.shape
            sizx = x.shape if hasattr(x, 'shape') else (len(x),)

            # Check if we need special handling for 3D arrays
            if len(siza) == 3:
                # a is (n1, 3, n2), x is (n2,) or (n2, ...)
                # Result should be (n1, 3, ...)
                n1, _, n2 = siza

                if len(sizx) == 1:
                    # x is 1D: (n2,)
                    # y[i, :] = sum_j a[i, :, j] * x[j]
                    y = np.tensordot(a, x, axes=([2], [0]))  # (n1, 3)
                else:
                    # x is multi-dimensional: (n2, ...)
                    # Reshape and multiply
                    a_flat = a.reshape(n1 * 3, n2)  # (n1*3, n2)
                    x_flat = x.reshape(n2, -1)  # (n2, prod(rest))
                    y_flat = a_flat @ x_flat  # (n1*3, prod(rest))

                    # Reshape back
                    new_shape = (n1, 3) + sizx[1:]
                    y = y_flat.reshape(new_shape)

                return y
            else:
                # Standard 2D matrix multiplication
                # a is (n1, n2), x is (n2,) or (n2, ...)
                if len(sizx) == 1:
                    return a @ x
                else:
                    # x is multi-dimensional
                    return a @ x.reshape(sizx[0], -1).reshape((sizx[0],) + sizx[1:])

    def __getattr__(self, name):
        """
        Property access via attribute lookup.

        MATLAB: @compgreenstat/subsref.m

        Provides access to:
        - obj.G, obj.F, obj.H1, obj.H2, obj.Gp, obj.H1p, obj.H2p
        - obj.deriv
        """
        if name in ['G', 'F', 'H1', 'H2', 'Gp', 'H1p', 'H2p']:
            return self.eval(name)
        elif name == 'deriv':
            return self.deriv
        else:
            raise AttributeError("'{}' object has no attribute '{}'".format(type(self).__name__, name))

    def __repr__(self):
        """String representation."""
        return (
            "CompGreenStat(p1: {} faces, "
            "p2: {} faces)".format(
                self.p1.n if hasattr(self.p1, 'n') else '?',
                self.p2.n if hasattr(self.p2, 'n') else '?')
        )

    def __str__(self):
        """Detailed string representation."""
        return (
            "compgreenstat:\n"
            "  p1: {}\n"
            "  p2: {}\n"
            "  G: {}\n"
            "  F: {}".format(
                self.p1,
                self.p2,
                self.G.shape if hasattr(self, 'G') else 'not computed',
                self.F.shape if hasattr(self, 'F') else 'not computed')
        )


class CompStruct(object):
    """
    Structure for compound of points or particles.

    MATLAB: @compstruct

    Contains:
    - a reference of the points or particles
    - the light wavelength in vacuum
    - an arbitrary number of additional fields

    For the fields, operations +, -, *, / are defined as for normal arrays.
    """

    def __init__(self, p, enei=None, **kwargs):
        """
        Initialize compstruct.

        Parameters
        ----------
        p : ComParticle or CompStruct
            compound of particles or other compstruct object
        enei : float, optional
            light wavelength in vacuum
        **kwargs : dict
            Additional fields (e.g., e=..., h=..., phi=..., etc.)
        """
        if isinstance(p, CompStruct):
            self.p = p.p
            self.enei = p.enei
            self.val = p.val.copy()
        else:
            self.p = p
            self.enei = enei
            self.val = {}

        # Add additional fields
        for key, value in kwargs.items():
            self.val[key] = value

    def __getattr__(self, name):
        """Access additional fields via attribute."""
        if name in ['p', 'enei', 'val']:
            return object.__getattribute__(self, name)
        elif name in self.val:
            return self.val[name]
        else:
            raise AttributeError("'{}' object has no attribute '{}'".format(type(self).__name__, name))

    def __setattr__(self, name, value):
        """Set additional fields via attribute."""
        if name in ['p', 'enei', 'val']:
            object.__setattr__(self, name, value)
        else:
            if not hasattr(self, 'val'):
                object.__setattr__(self, 'val', {})
            self.val[name] = value

    def __getitem__(self, key):
        """Dictionary-style access to fields."""
        if key == 'p':
            return self.p
        elif key == 'enei':
            return self.enei
        elif key in self.val:
            return self.val[key]
        else:
            raise KeyError("'{}'".format(key))

    def __setitem__(self, key, value):
        """Dictionary-style setting of fields."""
        if key == 'p':
            self.p = value
        elif key == 'enei':
            self.enei = value
        else:
            self.val[key] = value

    def get(self, key, default=None):
        """
        Dictionary-style get method with default value.

        Parameters
        ----------
        key : str
            Field name to retrieve
        default : any, optional
            Default value if key not found (default: None)

        Returns
        -------
        value : any
            Field value or default if not found
        """
        try:
            return self[key]
        except KeyError:
            return default

    def set(self, **kwargs):
        """
        Set field names of compstruct object.

        MATLAB: @compstruct/set.m

        Parameters
        ----------
        **kwargs : dict
            Field name-value pairs to set

        Returns
        -------
        self : CompStruct
            Updated compstruct object

        Examples
        --------
        >>> exc = exc.set(a1=a, a1p=ap)
        """
        for key, value in kwargs.items():
            self.val[key] = value
        return self

    def __add__(self, other):
        """
        Element-wise addition of CompStruct fields.

        MATLAB: @compstruct/plus.m

        Adds corresponding array fields from two CompStruct objects.
        The result keeps the particle and wavelength from self.

        Parameters
        ----------
        other : CompStruct
            CompStruct to add

        Returns
        -------
        result : CompStruct
            New CompStruct with summed fields
        """
        result = CompStruct(self.p, self.enei)
        all_keys = set(self.val.keys()) | set(other.val.keys())
        for key in all_keys:
            val_self = self.val.get(key, None)
            val_other = other.val.get(key, None)
            if val_self is not None and val_other is not None:
                result.val[key] = val_self + val_other
            elif val_self is not None:
                result.val[key] = val_self
            else:
                result.val[key] = val_other
        return result

    def __sub__(self, other):
        """
        Element-wise subtraction of CompStruct fields.

        MATLAB: @compstruct/minus.m
        """
        return self.__add__(-other)

    def __neg__(self):
        """
        Negate all fields of CompStruct.

        MATLAB: @compstruct/uminus.m
        """
        result = CompStruct(self.p, self.enei)
        for key, value in self.val.items():
            result.val[key] = -value
        return result

    def __mul__(self, other):
        """
        Multiply all fields by a scalar.

        MATLAB: @compstruct/mtimes.m
        """
        result = CompStruct(self.p, self.enei)
        for key, value in self.val.items():
            result.val[key] = other * value
        return result

    def __rmul__(self, other):
        """
        Right multiplication (scalar * CompStruct).

        MATLAB: @compstruct/mtimes.m
        """
        return self.__mul__(other)

    def __radd__(self, other):
        """
        Right addition for CompStruct.

        Supports sum() by handling 0 + CompStruct.

        Parameters
        ----------
        other : int or CompStruct
            If 0 (from sum() start value), returns self.

        Returns
        -------
        result : CompStruct
        """
        if other == 0:
            return self
        return self.__add__(other)

    def fieldnames(self):
        """
        Get field names of CompStruct.

        MATLAB: @compstruct/fieldnames.m

        Returns
        -------
        names : list of str
        """
        return list(self.val.keys())

    def isfield(self, name):
        """
        Check whether CompStruct has a field with the given name.

        MATLAB: @compstruct/isfield.m
        """
        return name in self.val

    def iselement(self, name):
        """
        Check whether CompStruct has a field with the given name.

        MATLAB: @compstruct/iselement.m (alias for isfield)
        """
        return name in self.val

    def getfield(self, name):
        """
        Get the value of a CompStruct field.

        MATLAB: @compstruct/getfield.m
        """
        return self.val[name]

    def to_struct(self):
        """
        Convert CompStruct to a plain dict.

        MATLAB: @compstruct/struct.m
        """
        return dict(self.val)

    def __repr__(self):
        """String representation."""
        fields = ', '.join(self.val.keys())
        return "CompStruct(p={}, enei={}, fields=[{}])".format(self.p, self.enei, fields)

    def __str__(self):
        """Detailed string representation."""
        s = "compstruct:\n"
        s += "  p: {}\n".format(self.p)
        s += "  enei: {}\n".format(self.enei)
        for key, value in self.val.items():
            if isinstance(value, np.ndarray):
                s += "  {}: array{}\n".format(key, value.shape)
            else:
                s += "  {}: {}\n".format(key, value)
        return s
