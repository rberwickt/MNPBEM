import numpy as np
import scipy.linalg as sla
from scipy.sparse.linalg import eigs
from typing import Tuple, Any, Optional

from ..greenfun import CompGreenStat


def plasmonmode(
        p: Any,
        nev: int = 20,
        **options: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MATLAB: BEM/plasmonmode.m

    Compute plasmon eigenmodes for a discretized particle surface.

    The eigenvalue problem is solved for the surface derivative F of the
    quasistatic Green function.  Eigenvalues correspond to plasmon
    eigenenergies and eigenvectors to the associated surface charge
    distributions.

    Parameters
    ----------
    p : ComParticle
        Compound of discretized particles (see comparticle).
    nev : int
        Number of eigenmodes to compute.  Defaults to 20.
    **options
        Additional keyword arguments forwarded to CompGreenStat.

    Returns
    -------
    ene : np.ndarray, shape (nev,)
        Plasmon eigenenergies (sorted by ascending real part).
    ur : np.ndarray, shape (n, nev)
        Right eigenvectors (surface charge patterns), columns sorted to
        match *ene*.
    ul : np.ndarray, shape (nev, n)
        Left eigenvectors, rows sorted to match *ene*.
    """

    # Green function and its surface derivative F  (MATLAB: compgreenstat)
    g = CompGreenStat(p, p, **options)
    F = g.F  # (n, n)

    n = F.shape[0]

    # Clamp nev so it does not exceed the matrix size minus 1
    # (scipy.sparse.linalg.eigs requires k < n)
    nev_actual = min(nev, n - 1) if n > 1 else 1

    eigs_opts = dict(which = 'SR', maxiter = 1000)

    # Use dense solver for small-to-medium matrices (n < 2000)
    # for exact eigenvalues matching MATLAB's LAPACK-based eigs.
    # Sparse ARPACK solver has ~1% convergence error on higher modes.
    use_dense = (n < 2000)

    if not use_dense and nev_actual < n - 1:
        # sparse eigenvalue solver (same as MATLAB eigs(..., 'sr'))
        _, ul = eigs(F.T, k = nev_actual, **eigs_opts)
        ul = ul.T  # (nev, n)

        ene_diag, ur = eigs(F, k = nev_actual, **eigs_opts)
    else:
        # Dense eigensolver (LAPACK) -- exact eigenvalues
        # Use scipy.linalg.eig with left=True, right=True so left and right
        # eigenvectors are paired with the SAME eigenvalue ordering. Computing
        # eig(F) and eig(F.T) separately permutes near-degenerate eigenvalues
        # differently, breaking the biorthogonality (ul * ur)\ul step.
        ene_all, vl_all, vr_all = sla.eig(F, left = True, right = True)
        idx_sort = np.argsort(ene_all.real)[:nev_actual]
        ene_diag = ene_all[idx_sort]
        ur = vr_all[:, idx_sort]
        # Left eigenvectors satisfy vl^H F = ene * vl^H, so ul rows = vl^H
        ul = vl_all[:, idx_sort].conj().T  # (nev, n)

    # extract eigenvalues and sort by ascending real part
    # IMPORTANT: sort BEFORE bi-orthogonalization so ul and ur are properly
    # paired. Sorting after bi-orthogonalization breaks the identity
    # ul @ ur = I when sort_idx is not the identity permutation.
    # Keep complex eigenvalues (matches MATLAB eigs which preserves the
    # imaginary part of numerically near-degenerate conjugate pairs).
    sort_idx = np.argsort(ene_diag.real)
    ene = ene_diag[sort_idx]

    ur = ur[:, sort_idx]
    ul = ul[sort_idx, :]

    # make eigenvectors bi-orthogonal  (MATLAB: ul = (ul * ur) \ ul)
    overlap = ul @ ur  # (nev, nev)
    ul = np.linalg.solve(overlap, ul)

    return ene, ur, ul
