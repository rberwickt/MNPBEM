import os
import sys
import copy

from typing import List, Dict, Tuple, Optional, Union, Any, Callable

import numpy as np

from mnpbem.utils.matlab_compat import msqrt

from .compgreen_stat import CompGreenStat, CompStruct


class CompGreenStatLayer(object):

    name = 'greenfunction'
    needs = {'sim': 'stat'}

    def __init__(self,
            p1: Any,
            p2: Any,
            layer: Any,
            **options: Any) -> None:

        self.p1 = p1
        self.p2 = p2
        self.layer = layer
        self.deriv = options.get('deriv', 'norm')

        # BEM solver cache
        self._enei_cache = None
        self._mat_cache = None

        # Initialize direct and reflected Green functions
        self._init(p1, p2, layer, **options)

    def _init(self,
            p1: Any,
            p2: Any,
            layer: Any,
            **options: Any) -> None:

        # MATLAB: @compgreenstatlayer/init.m uses image charge method
        # Only works for single layer (substrate) with particle in upper medium
        assert layer.n == 1, 'compgreenstatlayer requires a single interface'

        # Direct Green function (free-space)
        self.g = CompGreenStat(p1, p2, **options)

        z_layer = layer.z[0]
        self._z_layer = z_layer

        # MATLAB indices (1-based translated to 0-based):
        #   [ind, indl] = indlayer(layer, p2.pos(:,3))
        #   indl = find(indl)         faces of p2 LOCATED in the layer
        #   ind2 = setdiff(1:n, indl) faces of p2 NOT in the layer
        #   ind1 = find(indlayer(p1.pos(:,3)) == layer.ind(1))  p1 in upper medium
        ind_p2, in_p2 = layer.indlayer(p2.pos[:, 2])
        self._indl = np.where(in_p2)[0]
        self._ind2 = np.setdiff1d(np.arange(p2.pos.shape[0]), self._indl)

        ind_p1, _ = layer.indlayer(p1.pos[:, 2])
        self._ind1 = np.where(ind_p1 == layer.ind[0])[0]

        # All p2 positions must be in upper medium (assertion from MATLAB)
        assert np.all(ind_p2 == layer.ind[0]), \
            'compgreenstatlayer: p2 must be in upper medium'

        # Create reflected particle by mirroring ONLY ind2 faces of p2
        # MATLAB: p2r = shift(flip(select(shift(p2, vec), 'index', ind2), 3), -vec)
        self._create_reflected_green(p1, p2, z_layer, **options)

    def _create_reflected_green(self,
            p1: Any,
            p2: Any,
            z_layer: float,
            **options: Any) -> None:

        # Get the underlying particles from ComParticle
        if hasattr(p2, 'p'):
            pp2_list = p2.p
        else:
            pp2_list = [p2]

        if hasattr(p1, 'p'):
            pp1_list = p1.p
        else:
            pp1_list = [p1]

        # Concatenate p1 particles
        if len(pp1_list) == 1:
            pc1 = pp1_list[0]
        else:
            pc1 = pp1_list[0]
            for p in pp1_list[1:]:
                pc1 = pc1 + p

        # Create reflected version of each p2 particle
        reflected_particles = []
        for pp in pp2_list:
            pp_r = self._mirror_particle(pp, z_layer)
            reflected_particles.append(pp_r)

        # Concatenate reflected particles
        if len(reflected_particles) == 1:
            pc2r = reflected_particles[0]
        else:
            pc2r = reflected_particles[0]
            for p in reflected_particles[1:]:
                pc2r = pc2r + p

        # Compute reflected Green function with refinement
        # Pass waitbar=0 to suppress progress bars
        refl_options = {k: v for k, v in options.items()
                        if k in ['AbsCutoff', 'RelCutoff', 'memsize', 'deriv']}
        refl_options['waitbar'] = 0

        # Store reflected positions for backward compatibility
        self._pos2r = pc2r.pos.copy()

        # Use the same CompGreenStat logic for the reflected part
        self._gr = _ReflectedGreenStat(pc1, pc2r, **refl_options)

    def _mirror_particle(self,
            particle: Any,
            z_layer: float) -> Any:

        # Create a mirror of the particle across z_layer
        # MATLAB: shift(flip(shift(p, [0,0,-z_layer]), 3), [0,0,z_layer])
        # Equivalent to: z' = 2*z_layer - z, then flip normals

        pr = copy.deepcopy(particle)

        # Check if this is a proper Particle with full geometry
        has_full_geometry = (hasattr(pr, 'verts') and hasattr(pr, 'faces')
                            and hasattr(pr, '_norm'))

        if has_full_geometry:
            # Sync verts2 with verts if they are out of sync
            # (happens when user manually shifts verts/pos but not verts2)
            if hasattr(pr, 'verts2') and pr.verts2 is not None:
                verts_z_center = 0.5 * (pr.verts[:, 2].min() + pr.verts[:, 2].max())
                verts2_z_center = 0.5 * (pr.verts2[:, 2].min() + pr.verts2[:, 2].max())
                z_offset = verts_z_center - verts2_z_center
                if np.abs(z_offset) > 1e-6:
                    pr.verts2[:, 2] += z_offset

            # Mirror verts
            pr.verts[:, 2] = 2 * z_layer - pr.verts[:, 2]

            # Mirror verts2 if present
            if hasattr(pr, 'verts2') and pr.verts2 is not None:
                pr.verts2[:, 2] = 2 * z_layer - pr.verts2[:, 2]

            # Flip face ordering
            if hasattr(pr, 'index34'):
                ind3, ind4 = pr.index34()
            else:
                ind3 = np.arange(pr.faces.shape[0])
                ind4 = np.array([], dtype = int)

            if len(ind3) > 0:
                pr.faces[ind3, :3] = pr.faces[ind3, :3][:, ::-1]
            if len(ind4) > 0:
                pr.faces[ind4, :4] = pr.faces[ind4, :4][:, ::-1]

            # Flip faces2 if present
            if hasattr(pr, 'faces2') and pr.faces2 is not None:
                if len(ind3) > 0:
                    cols_src = [2, 1, 0, 5, 4, 6]
                    cols_dst = [0, 1, 2, 4, 5, 6]
                    temp = pr.faces2[ind3][:, cols_src].copy()
                    pr.faces2[np.ix_(ind3, cols_dst)] = temp
                if len(ind4) > 0:
                    cols_src = [3, 2, 1, 0, 6, 5, 4, 7]
                    cols_dst = [0, 1, 2, 3, 4, 5, 6, 7]
                    temp = pr.faces2[ind4][:, cols_src].copy()
                    pr.faces2[np.ix_(ind4, cols_dst)] = temp

            # Recompute normals and positions
            pr._norm()
        else:
            # For mock/simplified particles, mirror pos and nvec directly
            pr.pos = particle.pos.copy()
            pr.pos[:, 2] = 2 * z_layer - pr.pos[:, 2]
            if hasattr(pr, 'nvec'):
                pr.nvec = particle.nvec.copy()
                pr.nvec[:, 2] = -pr.nvec[:, 2]

        return pr

    def _image_factors(self,
            enei: float) -> Tuple[complex, complex, complex]:

        layer = self.layer

        # Get dielectric functions of upper and lower layers
        eps1_val, _ = layer.eps[0](enei)
        eps2_val, _ = layer.eps[1](enei)

        # Image charge factors (Jackson Eq. 4.45)
        f1 = 2 * eps1_val / (eps2_val + eps1_val)
        f2 = -(eps2_val - eps1_val) / (eps2_val + eps1_val)
        fl = eps1_val / eps2_val * f1

        return f1, f2, fl

    @property
    def G(self) -> np.ndarray:
        return self.g.G

    @property
    def F(self) -> np.ndarray:
        return self.g.F

    def eval(self,
            enei: float,
            key: str = 'G') -> np.ndarray:

        # MATLAB: @compgreenstatlayer/eval.m
        layer = self.layer
        eps1_val, _ = layer.eps[0](enei)
        eps2_val, _ = layer.eps[1](enei)

        # image charge factor in upper medium
        f2 = -(eps2_val - eps1_val) / (eps2_val + eps1_val)

        n1 = self.p1.pos.shape[0]
        ind1 = self._ind1
        indl = self._indl
        ind2 = self._ind2

        # MATLAB: accumarray(ind1, 1, [n,1], [], 2*eps1/(eps1+eps2))
        # Places val=1 at ind1 indices; fill others with 2*eps1/(eps1+eps2).
        f1 = np.full(n1, 2 * eps1_val / (eps1_val + eps2_val), dtype = complex)
        f1[ind1] = 1.0
        # MATLAB: 2 * accumarray(ind1, eps1, [n,1], [], eps2) / (eps1+eps2)
        # ind1 -> eps1; others -> eps2; then *2/(eps1+eps2).
        fl = np.full(n1, 2 * eps2_val / (eps1_val + eps2_val), dtype = complex)
        fl[ind1] = 2 * eps1_val / (eps1_val + eps2_val)

        same_p = (self.p1 is self.p2) or getattr(self, 'p1', None) is getattr(self, 'p2', None)

        if key == 'G':
            G = self.g.G * f1[:, np.newaxis]
            # layer correction
            if len(indl) > 0:
                G[:, indl] = self.g.G[:, indl] * fl[:, np.newaxis]
            # reflected part (only for upper-medium p1 x not-in-layer p2)
            if len(ind1) > 0 and len(ind2) > 0:
                gr_F = self._gr.G
                G[np.ix_(ind1, ind2)] += f2 * gr_F[np.ix_(ind1, ind2)]
            return G

        if key in ('F', 'H1', 'H2'):
            F = self.g.F * f1[:, np.newaxis]
            # layer correction: (ind1 \ indl, indl) *= (1 + f2) / f1_prev_factor
            # MATLAB: F(ind, indl) = F(ind, indl) * (1 + f2), where F already had
            # f1 applied. Since f1[ind]=2*eps1/(eps1+eps2) and we want to end up
            # with (1+f2), we re-scale from f1 to (1+f2) by multiplying by (1+f2)/f1.
            # Simpler: recompute from raw g.F * (1 + f2).
            ind_notin = np.setdiff1d(ind1, indl)
            if len(ind_notin) > 0 and len(indl) > 0:
                F[np.ix_(ind_notin, indl)] = self.g.F[np.ix_(ind_notin, indl)] * (1 + f2)
            # reflected part
            if len(ind1) > 0 and len(ind2) > 0:
                F[np.ix_(ind1, ind2)] += f2 * self._gr.F[np.ix_(ind1, ind2)]
            # Same-particle: zero out in-layer self-block + diagonal f
            if same_p and len(indl) > 0:
                F[np.ix_(indl, indl)] = 0
                f_diag = np.zeros(n1, dtype = complex)
                f_diag[indl] = f2
                if key == 'H1':
                    F = F + 2 * np.pi * (np.diag(f_diag) + np.eye(n1))
                elif key == 'H2':
                    F = F + 2 * np.pi * (np.diag(f_diag) - np.eye(n1))
            elif same_p:
                if key == 'H1':
                    F = F + 2 * np.pi * np.eye(n1)
                elif key == 'H2':
                    F = F - 2 * np.pi * np.eye(n1)
            return F

        if key == 'Gp':
            Gp = self.g._eval_Gp()
            Gp_out = Gp * f1[:, np.newaxis, np.newaxis]
            if len(indl) > 0:
                Gp_out[:, :, indl] = Gp[:, :, indl] * fl[:, np.newaxis, np.newaxis]
            if len(ind1) > 0 and len(ind2) > 0:
                Gp_out[np.ix_(ind1, np.arange(3), ind2)] += f2 * self._gr.Gp[np.ix_(ind1, np.arange(3), ind2)]
            return Gp_out

        if key in ('H1p', 'H2p'):
            # MATLAB eval.m handles Gp (not H1p/H2p directly). H1p/H2p add
            # 2π·nvec only for self-term (p1 is p2). Apply the same layer
            # corrections as Gp.
            Hp_raw = self.g._eval_H1p() if key == 'H1p' else self.g._eval_H2p()
            Hp = Hp_raw * f1[:, np.newaxis, np.newaxis]
            if len(indl) > 0:
                Hp[:, :, indl] = Hp_raw[:, :, indl] * fl[:, np.newaxis, np.newaxis]
            if len(ind1) > 0 and len(ind2) > 0:
                Hp[np.ix_(ind1, np.arange(3), ind2)] += f2 * self._gr.Gp[np.ix_(ind1, np.arange(3), ind2)]
            return Hp

        raise ValueError('[error] Unknown Green function key: <{}>'.format(key))

    def eval_multi(self,
            enei: float,
            *keys: str) -> Tuple[np.ndarray, ...]:

        results = []
        for key in keys:
            results.append(self.eval(enei, key))

        if len(results) == 1:
            return results[0]
        return tuple(results)

    def potential(self,
            sig: Any,
            inout: int = 1) -> CompStruct:

        enei = sig.enei

        H_key = 'H1' if inout == 1 else 'H2'

        G = self.eval(enei, 'G')
        H = self.eval(enei, H_key)

        phi = G @ sig.sig if hasattr(sig, 'sig') else np.zeros(self.p1.pos.shape[0])
        phip = H @ sig.sig if hasattr(sig, 'sig') else np.zeros(self.p1.pos.shape[0])

        if inout == 1:
            return CompStruct(self.p1, enei, phi1 = phi, phi1p = phip)
        else:
            return CompStruct(self.p1, enei, phi2 = phi, phi2p = phip)

    @staticmethod
    def _matmul(M, x):
        """Multiply 3D Green function matrix M (n1,3,n2) with vector/matrix x.

        Matches CompGreenStat._matmul logic.
        """
        if x.ndim == 1:
            return np.einsum('ijk,k->ij', M, x)
        elif x.ndim == 2:
            return np.einsum('ijk,kl->ijl', M, x)
        else:
            return np.einsum('ijk,k...->ij...', M, x)

    def field(self,
            sig: Any,
            inout: int = 1) -> CompStruct:

        enei = sig.enei
        Hp_key = 'H1p' if inout == 1 else 'H2p'

        # Use Cartesian derivative (H1p/H2p) for correct 3D electric field
        Hp = self.eval(enei, Hp_key)
        e = -self._matmul(Hp, sig.sig)

        return CompStruct(self.p1, enei, e = e)

    def __repr__(self) -> str:
        n1 = self.p1.pos.shape[0] if hasattr(self.p1, 'pos') else '?'
        n2 = self.p2.pos.shape[0] if hasattr(self.p2, 'pos') else '?'
        return 'CompGreenStatLayer(p1: {} faces, p2: {} faces)'.format(n1, n2)


class _ReflectedGreenStat(object):
    """
    Green function between p1 and the reflected (image) particle p2r.

    This is a simplified version of CompGreenStat that handles the case
    where p1 and p2r are different particles (no closed surface correction).
    Uses integration refinement for nearby elements.
    """

    def __init__(self, p1, p2r, **options):
        self.deriv = options.get('deriv', 'norm')
        self._compute(p1, p2r, **options)

    def _compute(self, p1, p2r, **options):

        pos1 = p1.pos
        pos2 = p2r.pos
        nvec1 = p1.nvec
        area2 = p2r.area

        n1 = pos1.shape[0]
        n2 = pos2.shape[0]

        # Compute distances
        r = pos1[:, np.newaxis, :] - pos2[np.newaxis, :, :]  # (n1, n2, 3)
        d = np.linalg.norm(r, axis = 2)  # (n1, n2)
        d_safe = np.maximum(d, np.finfo(float).eps)

        # G matrix: G[i,j] = 1/d * area[j]
        self.G = (1.0 / d_safe) * area2[np.newaxis, :]

        # Gp matrix (Cartesian derivative): Gp[i,:,j] = -r[i,j,:] / d^3 * area[j]
        # Always compute for field() support (shape: n1, 3, n2)
        Gp_raw = -r / (d_safe[:, :, np.newaxis] ** 3) * area2[np.newaxis, :, np.newaxis]
        self.Gp = np.transpose(Gp_raw, (0, 2, 1))  # (n1, 3, n2)

        # F matrix: F[i,j] = - nvec1[i].r[i,j] / d^3 * area[j]
        n_dot_r = np.sum(nvec1[:, np.newaxis, :] * r, axis = 2)
        if self.deriv == 'norm':
            self.F = -n_dot_r / (d_safe ** 3) * area2[np.newaxis, :]
        else:
            self.F = -r / (d_safe[:, :, np.newaxis] ** 3) * area2[np.newaxis, :, np.newaxis]

        # Apply refinement for nearby elements
        # The reflected particle may have elements close to p1 (especially for
        # particles near the layer interface)
        has_quad = (hasattr(p2r, 'quadpol') and hasattr(p2r, 'quad_integration')
                    and hasattr(p2r, 'quad'))
        if has_quad:
            self._refine(p1, p2r, **options)

    def _refine(self, p1, p2r, **options):
        """Apply integration refinement for nearby reflected elements.

        MATLAB: greenstat/private/init.m (off-diagonal refinement section)
        """
        from .utils import refinematrix

        pos1 = p1.pos
        nvec1 = p1.nvec

        n1 = pos1.shape[0]
        n2 = p2r.pos.shape[0]

        # Get refinement matrix (marks nearby elements that need refinement)
        refine_opts = {k: v for k, v in options.items()
                       if k in ['AbsCutoff', 'RelCutoff', 'memsize']}
        try:
            ir = refinematrix(p1, p2r, **refine_opts)
        except Exception:
            return

        ir_dense = ir.toarray() if hasattr(ir, 'toarray') else np.asarray(ir)

        # For reflected particles, there should be no diagonal refinement (ir==2)
        # since p1 != p2r. Only off-diagonal refinement (ir==1).
        # Also include ir==2 in case refinematrix marks some as diagonal
        offdiag_mask = (ir_dense == 1) | (ir_dense == 2)
        if not np.any(offdiag_mask):
            return

        # Get faces that need refinement
        reface_cols = np.where(np.any(offdiag_mask, axis = 0))[0]
        if len(reface_cols) == 0:
            return

        # Use quad_integration to get integration points and weights
        # MATLAB: [postab, wtab] = quad(p2, reface)
        try:
            postab_all, wtab_sparse, iface_all = p2r.quad_integration(reface_cols)
        except Exception:
            return

        if postab_all is None:
            return

        # wtab_sparse is (n_reface, n_points) sparse matrix
        # For each face in reface_cols, extract the integration points and weights
        for idx, face in enumerate(reface_cols):
            # Rows of p1 that need refinement for this face
            nb = np.where(offdiag_mask[:, face])[0]
            if len(nb) == 0:
                continue

            # Get weights for this face from sparse matrix
            w_row = wtab_sparse[idx, :].toarray().flatten()
            w_mask = w_row != 0
            if not np.any(w_mask):
                continue

            w = w_row[w_mask]
            pos = postab_all[w_mask]

            # Compute refined Green function and surface derivative
            # r = pos1[nb] - integration_points
            x = pos1[nb, 0:1] - pos[:, 0:1].T  # (nnb, nquad)
            y = pos1[nb, 1:2] - pos[:, 1:2].T
            z = pos1[nb, 2:3] - pos[:, 2:3].T

            r_dist = msqrt(x ** 2 + y ** 2 + z ** 2)
            r_dist = np.maximum(r_dist, np.finfo(float).eps)

            # G
            self.G[nb, face] = (1.0 / r_dist) @ w

            # Gp (Cartesian derivative, always updated)
            self.Gp[nb, 0, face] = -(x / r_dist ** 3) @ w
            self.Gp[nb, 1, face] = -(y / r_dist ** 3) @ w
            self.Gp[nb, 2, face] = -(z / r_dist ** 3) @ w

            # F (surface derivative)
            if self.deriv == 'norm':
                ndotr = (nvec1[nb, 0:1] * x + nvec1[nb, 1:2] * y
                         + nvec1[nb, 2:3] * z)
                self.F[nb, face] = -(ndotr / r_dist ** 3) @ w
            else:
                self.F[nb, face, 0] = -(x / r_dist ** 3) @ w
                self.F[nb, face, 1] = -(y / r_dist ** 3) @ w
                self.F[nb, face, 2] = -(z / r_dist ** 3) @ w
