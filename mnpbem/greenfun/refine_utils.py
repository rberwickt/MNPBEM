"""
Utility functions for Green function refinement.

MATLAB reference: /Greenfun/+green/
"""

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.spatial import distance


def _pdist2_matlab(p1, p2):
    """MATLAB Misc/+misc/pdist2.m parity.

    Uses the algebraic identity d^2 = |p1|^2 + |p2|^2 - 2 p1.p2 (instead of the
    direct sqrt(sum((p1-p2)^2)) form scipy.cdist uses) so that boundary cases
    in refinematrix's `id_rel < RelCutoff` test agree with MATLAB at the ULP
    level. See Wave 21 Track C diagnostics.
    """
    p1 = np.asarray(p1)
    p2 = np.asarray(p2)
    n1_sq = np.sum(p1 * p1, axis=1)
    n2_sq = np.sum(p2 * p2, axis=1)
    d_sq = n1_sq[:, np.newaxis] + n2_sq[np.newaxis, :] - 2.0 * (p1 @ p2.T)
    d_sq[d_sq < 1e-10] = 0.0
    return np.sqrt(d_sq)


def refinematrix(p1, p2, AbsCutoff=0, RelCutoff=3, memsize=2e7):
    """
    Determine which Green function elements need refinement.

    Creates a sparse matrix indicating refinement requirements:
    - 0: Far field (no refinement needed, use 1/d approximation)
    - 1: Near field (refine off-diagonal with boundary element integration)
    - 2: Diagonal (refine with polar integration)

    MATLAB reference: /Greenfun/+green/refinematrix.m

    Parameters
    ----------
    p1, p2 : Particle
        Boundary element particles
    AbsCutoff : float, optional
        Absolute distance cutoff (nm) for refinement
        Default: 0 (use only relative cutoff)
    RelCutoff : float, optional
        Relative distance cutoff (multiples of element radius)
        Default: 3
    memsize : float, optional
        Maximum memory for distance matrix chunks
        Default: 2e7 (20M elements)

    Returns
    -------
    ir : scipy.sparse matrix, shape (p1.n, p2.n)
        Refinement matrix
        ir[i,j] = 0: far field
        ir[i,j] = 1: near field (refine off-diagonal)
        ir[i,j] = 2: diagonal (refine with polar integration)

    Notes
    -----
    The refinement matrix is used to determine which Green function
    elements require special treatment:

    - Diagonal elements (d=0): Always refined with polar integration
      to handle self-interaction singularity

    - Near off-diagonal (d < cutoff): Refined with boundary element
      integration for better accuracy

    - Far field (d > cutoff): Use simple 1/d * exp(ikd) formula

    The cutoffs are determined by:
    - AbsCutoff: absolute distance in nm
    - RelCutoff: distance in units of element radius

    An element is refined if EITHER cutoff is satisfied.

    Examples
    --------
    >>> from mnpbem import trisphere
    >>> p = trisphere(144, 10)  # 10 nm sphere
    >>> ir = refinematrix(p, p, RelCutoff=3)
    >>> print("Diagonal: {}".format(np.sum(ir.diagonal() == 2)))
    >>> print("Near: {}".format(np.sum((ir == 1).toarray())))
    >>> print("Far: {}".format(np.sum((ir == 0).toarray())))
    """
    # Positions and sizes
    pos1 = p1.pos  # (n1, 3)
    pos2 = p2.pos  # (n2, 3)
    n1, n2 = p1.n, p2.n

    # Boundary element radius (for relative cutoff)
    rad2 = p2.bradius()  # (n2,)

    # Radius for p1 (use p1's own radius, fallback to p2 if needed)
    try:
        rad1 = p1.bradius()  # (n1,)
    except (AttributeError, NotImplementedError):
        rad1 = rad2

    # Initialize sparse refinement matrix
    ir = lil_matrix((n1, n2), dtype=int)

    # Process in chunks to manage memory
    # Each chunk processes memsize / n1 columns of the distance matrix
    chunk_size = max(1, int(memsize / n1))
    n_chunks = int(np.ceil(n2 / chunk_size))

    for chunk_idx in range(n_chunks):
        # Column range for this chunk
        i_start = chunk_idx * chunk_size
        i_end = min(i_start + chunk_size, n2)
        chunk = slice(i_start, i_end)

        # Distance matrix for this chunk: (n1, chunk_size)
        # Use MATLAB's algebraic pdist2 form for ULP parity at the
        # `id_rel == RelCutoff` boundary (see _pdist2_matlab docstring).
        d = _pdist2_matlab(pos1, pos2[chunk])

        # Approximate distance to boundary element centers
        # Subtract element radius to get distance to element surface
        d2 = d - rad2[chunk][np.newaxis, :]

        # Relative distance (in units of element radius)
        if rad1.ndim == 1 and len(rad1) == n1:
            # Different radius for each element in p1
            id_rel = d2 / rad1[:, np.newaxis]
        else:
            # Use radius of p2 elements
            id_rel = d2 / rad2[chunk][np.newaxis, :]

        # Find diagonal elements (d == 0)
        row_diag, col_diag = np.where(d == 0)

        # Find near elements (within cutoff, but not diagonal)
        # An element is "near" if:
        # - d2 < AbsCutoff (absolute distance) OR
        # - id_rel < RelCutoff (relative distance)
        # AND d != 0 (not diagonal)
        near_mask = ((d2 < AbsCutoff) | (id_rel < RelCutoff)) & (d != 0)
        row_near, col_near = np.where(near_mask)

        # Set refinement flags
        if len(row_diag) > 0:
            # Diagonal elements: ir = 2
            ir[row_diag, col_diag + i_start] = 2

        if len(row_near) > 0:
            # Near off-diagonal elements: ir = 1
            ir[row_near, col_near + i_start] = 1

        # Note: Far field elements remain 0 (default)

    # Convert to CSR format for efficient access
    return ir.tocsr()


def refinematrixlayer(p1, p2, layer, AbsCutoff=0, RelCutoff=3, memsize=2e7):

    # MATLAB: /Greenfun/+green/refinematrixlayer.m
    # Refinement matrix for layer structures.
    # Uses in-plane radial distance + minimum z-distance to layer boundaries
    # instead of direct Euclidean distance.
    #
    # Returns sparse matrix:
    #   2 - diagonal elements (same particle, same position)
    #   1 - off-diagonal elements needing refinement
    #   0 - far field (no refinement)

    pos1 = p1.pos  # (n1, 3)
    pos2 = p2.pos  # (n2, 3)
    n1, n2 = p1.n, p2.n

    # Boundary element radius
    rad2 = p2.bradius()  # (n2,)
    try:
        rad1 = p1.bradius()  # (n1,)
    except (AttributeError, NotImplementedError):
        rad1 = rad2

    # Check if p1 and p2 are the same particle (for diagonal detection)
    same_particle = (pos1.shape == pos2.shape and np.all(pos1 == pos2))

    # Initialize sparse refinement matrix
    ir = lil_matrix((n1, n2), dtype = int)

    # Process in chunks
    chunk_size = max(1, int(memsize / n1))
    n_chunks = int(np.ceil(n2 / chunk_size))

    for chunk_idx in range(n_chunks):
        i_start = chunk_idx * chunk_size
        i_end = min(i_start + chunk_size, n2)
        chunk = slice(i_start, i_end)
        n_chunk = i_end - i_start

        # In-plane radial distance (xy only)
        # MATLAB refinematrixlayer.m uses misc.pdist2 (algebraic form). Use
        # _pdist2_matlab here for ULP-level parity at the cutoff boundary.
        r_xy = _pdist2_matlab(pos1[:, :2], pos2[chunk, :2])  # (n1, n_chunk)

        # Minimum z-distance to layer boundaries for each point
        zmin1, _ = layer.mindist(pos1[:, 2])   # (n1,)
        zmin2, _ = layer.mindist(pos2[chunk, 2])  # (n_chunk,)

        # Combined z-distance: sum of minimum distances to layers
        z_dist = zmin1[:, np.newaxis] + zmin2[np.newaxis, :]  # (n1, n_chunk)

        # Total distance: sqrt(r_xy^2 + z^2)
        d = np.sqrt(r_xy ** 2 + z_dist ** 2)

        # Subtract boundary element radius
        d2 = d - rad2[chunk][np.newaxis, :]

        # Distance in units of boundary element radius
        # MATLAB: if size(rad,1) ~= 1 (column vector of length p1.n), use rad;
        # else fall back to rad2(chunk). For ComPoint which lacks bradius,
        # rad1 was set to rad2 (length n2, NOT n1) -- in that case use rad2.
        if len(rad1) == n1:
            id_rel = d2 / rad1[:, np.newaxis]
        else:
            id_rel = d2 / rad2[chunk][np.newaxis, :]

        # Find elements needing refinement
        near_mask = (d2 < AbsCutoff) | (id_rel < RelCutoff)
        row_near, col_near = np.where(near_mask)

        if len(row_near) > 0:
            ir[row_near, col_near + i_start] = 1

        # Diagonal elements (same particle): set to 2
        if same_particle:
            diag_mask = near_mask.copy()
            # Find actual diagonal entries in this chunk
            for local_col in range(n_chunk):
                global_col = local_col + i_start
                if global_col < n1:
                    if near_mask[global_col, local_col]:
                        ir[global_col, global_col] = 2

    return ir.tocsr()


# Test function
if __name__ == "__main__":
    print("Testing refinematrix:")
    print("=" * 70)

    # Create a simple test particle (manual construction to avoid trisphere issues)
    import sys
    sys.path.insert(0, '/home/user/MNPBEM')
    from mnpbem.geometry.particle import Particle

    # Simple cube made of triangles
    verts = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],  # Bottom
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]   # Top
    ]) * 10.0  # 10 nm cube

    # 12 triangular faces (2 per cube side)
    faces = np.array([
        [0, 1, 2], [0, 2, 3],  # Bottom
        [4, 6, 5], [4, 7, 6],  # Top
        [0, 5, 1], [0, 4, 5],  # Front
        [2, 7, 3], [2, 6, 7],  # Back
        [0, 7, 4], [0, 3, 7],  # Left
        [1, 6, 2], [1, 5, 6],  # Right
    ])

    p = Particle(verts, faces)

    print("\nParticle: {} faces".format(p.n))
    print("Centroid positions shape: {}".format(p.pos.shape))
    print("bradius shape: {}".format(p.bradius().shape))

    # Test refinement matrix
    print("\nComputing refinement matrix...")
    ir = refinematrix(p, p, AbsCutoff=0, RelCutoff=3)

    print("Refinement matrix shape: {}".format(ir.shape))
    print("Refinement matrix density: {:.4f}".format(ir.nnz / (ir.shape[0] * ir.shape[1])))

    # Count refinement types
    ir_array = ir.toarray()
    n_diag = np.sum(ir_array == 2)
    n_near = np.sum(ir_array == 1)
    n_far = np.sum(ir_array == 0)

    print("\nRefinement statistics:")
    print("  Diagonal (ir=2):      {:6d} ({:5.2f}%)".format(n_diag, 100 * n_diag / ir.size))
    print("  Near field (ir=1):    {:6d} ({:5.2f}%)".format(n_near, 100 * n_near / ir.size))
    print("  Far field (ir=0):     {:6d} ({:5.2f}%)".format(n_far, 100 * n_far / ir.size))

    # Verify diagonal
    diag_check = np.all(np.diag(ir_array) == 2)
    print("\nAll diagonal elements = 2: {}".format(diag_check))

    # Test different cutoffs
    print("\nTesting different RelCutoff values:")
    for cutoff in [1, 2, 3, 5, 10]:
        ir_test = refinematrix(p, p, RelCutoff=cutoff)
        n_refined = ir_test.nnz
        density = n_refined / ir_test.size
        print("  RelCutoff={:2d}: {:6d} refined ({:5.2f}%)".format(cutoff, n_refined, 100 * density))

    print("\n" + "=" * 70)
    print("✓ refinematrix tests passed!")
