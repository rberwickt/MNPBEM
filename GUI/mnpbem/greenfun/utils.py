"""
Utility functions for Green function computations.

MATLAB: Misc/+misc/
"""

import numpy as np
from scipy.sparse import csr_matrix


def pdist2(p1, p2):
    """
    Distance array between positions p1 and p2.

    MATLAB: Misc/+misc/pdist2.m

    Parameters
    ----------
    p1 : ndarray, shape (n1, 3)
        First position array
    p2 : ndarray, shape (n2, 3)
        Second position array

    Returns
    -------
    d : ndarray, shape (n1, n2)
        Distance array between p1 and p2
    """
    # Square of distance: d^2 = ||p1||^2 + ||p2||^2 - 2*p1*p2^T
    p1_sq = np.sum(p1**2, axis=1, keepdims=True)  # (n1, 1)
    p2_sq = np.sum(p2**2, axis=1, keepdims=True)  # (n2, 1)
    d_sq = p1_sq + p2_sq.T - 2 * p1 @ p2.T

    # Avoid rounding errors
    d_sq[d_sq < 1e-10] = 0

    # Square root
    d = np.sqrt(d_sq)

    return d


def refinematrix(p1, p2, AbsCutoff=0, RelCutoff=3, memsize=2e7):
    """
    Refinement matrix for Green functions.

    MATLAB: Greenfun/+green/refinematrix.m

    Parameters
    ----------
    p1 : Particle
        Discretized particle boundary 1
    p2 : Particle
        Discretized particle boundary 2
    AbsCutoff : float, optional
        Absolute distance for integration refinement (default: 0)
    RelCutoff : float, optional
        Relative distance for integration refinement (default: 3, MATLAB standard)
    memsize : float, optional
        Deal at most with matrices of size memsize (default: 2e7)

    Returns
    -------
    mat : sparse matrix, shape (p1.n, p2.n)
        Refinement matrix
        - 2 for diagonal elements
        - 1 for off-diagonal elements requiring refinement
        - 0 for regular elements (no refinement needed)
    """
    # Positions
    pos1 = p1.pos
    pos2 = p2.pos
    n1 = p1.nfaces
    n2 = p2.nfaces

    # Boundary element radius
    rad2 = p2.bradius()

    # Radius for relative distances
    # Try to use boundary radius of first particle
    try:
        rad = p1.bradius()
    except:
        rad = rad2

    # Allocate sparse output array
    rows_all = []
    cols_all = []
    vals_all = []

    # Work through full matrix size N1 x N2 in portions of memsize
    chunk_size = max(1, int(memsize / n1))
    ind2_ranges = list(range(0, n2, chunk_size))
    if ind2_ranges[-1] != n2:
        ind2_ranges.append(n2)

    # Loop over portions
    for i in range(1, len(ind2_ranges)):
        start_idx = ind2_ranges[i - 1]
        end_idx = ind2_ranges[i]
        i2 = slice(start_idx, end_idx)

        # Distance between positions
        d = pdist2(pos1, pos2[i2])

        # Subtract radius from distances to get approximate distance
        # between pos1 and boundary elements
        d2 = d - rad2[np.newaxis, i2]

        # Distances in units of boundary element radius
        if rad.size == n1:
            # Each element of p1 has its own radius
            id_rel = d2 / rad[:, np.newaxis]
        else:
            # Use p2 radius for normalization
            id_rel = d2 / rad2[np.newaxis, i2]

        # Diagonal elements (d == 0)
        row1, col1 = np.where(d == 0)
        if len(row1) > 0:
            rows_all.append(row1)
            cols_all.append(col1 + start_idx)
            vals_all.append(np.full(len(row1), 2))

        # Off-diagonal elements for refinement
        # (d2 < AbsCutoff OR id < RelCutoff) AND d != 0
        mask = ((d2 < AbsCutoff) | (id_rel < RelCutoff)) & (d != 0)
        row2, col2 = np.where(mask)
        if len(row2) > 0:
            rows_all.append(row2)
            cols_all.append(col2 + start_idx)
            vals_all.append(np.full(len(row2), 1))

    # Combine all chunks
    if len(rows_all) > 0:
        rows = np.hstack(rows_all)
        cols = np.hstack(cols_all)
        vals = np.hstack(vals_all)
        mat = csr_matrix((vals, (rows, cols)), shape=(n1, n2))
    else:
        mat = csr_matrix((n1, n2))

    return mat
