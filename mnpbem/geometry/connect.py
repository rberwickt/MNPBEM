"""
Connectivity function for compound particles.

Matches MATLAB MNPBEM @compound/connect.m implementation.
"""

import numpy as np


def connect(p1, p2=None, ind=None):
    """
    Compute connectivity between compound particles.

    Determines whether boundary elements can "see" each other
    through the same medium.

    MATLAB: @compound/connect.m

    Parameters
    ----------
    p1 : ComParticle or similar
        First particle/point object with inout property
    p2 : ComParticle or similar, optional
        Second particle/point object. If None, uses p1.
    ind : array, optional
        Index array for replacing dielectric material indices

    Returns
    -------
    con : list of list of ndarray
        Connectivity matrices. con[i][j] is a (n1, n2) array
        where non-zero entries indicate connected faces and the
        value is the medium index.

    Notes
    -----
    For each face pair (i, j), con tells which medium connects them.
    If faces i and j are on the same medium boundary, con[inout][ip][ipt]
    will be non-zero (equal to the medium index).

    The connectivity is organized as:
    - con[0]: connections through inside surface (inout column 0)
    - con[1]: connections through outside surface (inout column 1)

    Examples
    --------
    >>> con = connect(particle, dipole_points)
    >>> # con[0] = connectivity through inside (inout column 0)
    >>> # con[1] = connectivity through outside (inout column 1)
    """
    # Get inout arrays (with mask if available)
    inout1 = _get_masked_inout(p1)

    if p2 is None:
        inout2 = inout1
    else:
        inout2 = _get_masked_inout(p2)

    # Apply index replacement if provided
    if ind is not None:
        inout1 = ind[inout1]
        inout2 = ind[inout2]

    # Size of inout arrays
    # n1, n2 are number of columns (inside/outside)
    ncol1 = inout1.shape[1]
    ncol2 = inout2.shape[1]

    # Allocate connectivity cell array (list of lists)
    con = [[None for _ in range(ncol2)] for _ in range(ncol1)]

    # Determine whether points can see each other
    for i in range(ncol1):
        for j in range(ncol2):
            # Get inout values for this column combination
            io1 = inout1[:, i]  # (nfaces1,)
            io2 = inout2[:, j]  # (nfaces2,)

            # Create comparison matrices
            # c1[k, l] = io1[k], c2[k, l] = io2[l]
            c1 = np.broadcast_to(io1[:, np.newaxis], (len(io1), len(io2)))
            c2 = np.broadcast_to(io2[np.newaxis, :], (len(io1), len(io2)))

            # Connectivity: non-zero where media match
            conn = np.zeros((len(io1), len(io2)), dtype=int)
            match = (c1 == c2)
            conn[match] = c1[match]

            con[i][j] = conn

    return con


def _get_masked_inout(p):
    """
    Get masked inout property from particle.

    MATLAB: get = @( p ) ( p.inout( p.mask, : ) );

    Parameters
    ----------
    p : ComParticle or point-like object
        Object with inout property

    Returns
    -------
    inout : ndarray
        Inout array, possibly masked
    """
    if hasattr(p, 'inout_faces'):
        # ComParticle: expand inout to all faces
        return p.inout_faces
    elif hasattr(p, 'inout'):
        # Already has inout array
        inout = np.atleast_2d(p.inout)
        if hasattr(p, 'mask') and p.mask is not None:
            mask = np.asarray(p.mask)
            if len(mask) == len(inout):
                return inout[mask]
        return inout
    else:
        # For point objects without explicit inout, assume medium 1
        # This is the case for dipole positions that are just in one medium
        if hasattr(p, 'n'):
            n = p.n
        elif hasattr(p, 'npt'):
            n = p.npt
        elif hasattr(p, 'shape'):
            n = p.shape[0]
        else:
            n = 1

        # Default: assume all points are in medium 1 (both inside and outside)
        # For dipole points, this means they're embedded in medium 1
        return np.ones((n, 2), dtype=int)
