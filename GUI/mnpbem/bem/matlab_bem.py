"""MATLAB Engine bridge for BEM solver linear solves (Wave 66).

Opt-in helper: when BEMRetLayer is constructed with use_matlab_engine=True,
the dense complex linear solve of the 2n x 2n block matrix and downstream
solves are routed through a MATLAB Engine session. This mirrors MATLAB's
mldivide / lu / solve numerical behavior exactly, eliminating residual
differences from numpy's scipy lu_factor / lu_solve.
"""
import os

import numpy as np


_engine = None


def get_engine():
    global _engine
    if _engine is not None:
        return _engine

    import matlab.engine as _matlab_engine_mod

    eng = _matlab_engine_mod.start_matlab()
    eng.addpath(eng.genpath('/home/yoojk20/workspace/MNPBEM'), nargout=0)
    helper_dir = os.path.dirname(os.path.abspath(__file__))
    eng.addpath(helper_dir, nargout=0)

    _engine = eng
    return eng


def matlab_solve(M, b):
    """Solve M x = b using MATLAB's mldivide.

    Parameters
    ----------
    M : ndarray (n, n) complex
    b : ndarray (n,) or (n, k) complex

    Returns
    -------
    x : ndarray, same shape as b, complex
    """
    import matlab as _matlab_mod

    eng = get_engine()

    M = np.asarray(M, dtype=complex)
    b = np.asarray(b, dtype=complex)

    one_d = (b.ndim == 1)
    if one_d:
        b2 = b.reshape(-1, 1)
    else:
        b2 = b

    M_r = _matlab_mod.double(np.real(M).tolist())
    M_i = _matlab_mod.double(np.imag(M).tolist())
    b_r = _matlab_mod.double(np.real(b2).tolist())
    b_i = _matlab_mod.double(np.imag(b2).tolist())

    x_r, x_i = eng.mnpbem_bem_solve_helper(
        M_r, M_i, b_r, b_i, nargout=2)

    x_r_arr = np.asarray(x_r, dtype=float)
    x_i_arr = np.asarray(x_i, dtype=float)
    if x_r_arr.ndim == 0:
        x_r_arr = x_r_arr.reshape(1, 1)
    elif x_r_arr.ndim == 1:
        x_r_arr = x_r_arr.reshape(-1, 1)
    if x_i_arr.ndim == 0:
        x_i_arr = x_i_arr.reshape(1, 1)
    elif x_i_arr.ndim == 1:
        x_i_arr = x_i_arr.reshape(-1, 1)

    x = x_r_arr + 1j * x_i_arr

    if one_d:
        return x.ravel()
    return x.reshape(b2.shape)


def _to_matlab_complex(eng, A):
    """Convert numpy array to a MATLAB complex matrix variable in the engine."""
    import matlab as _matlab_mod
    A = np.asarray(A, dtype=complex)
    if A.ndim == 1:
        A = A.reshape(-1, 1)
    A_r = _matlab_mod.double(np.real(A).tolist())
    A_i = _matlab_mod.double(np.imag(A).tolist())
    return A_r, A_i


def _from_matlab(arr):
    """Convert a MATLAB return value to a numpy ndarray."""
    a = np.asarray(arr)
    return a


def matlab_bem_init(G11, G21, H11, H21,
                    G22, G12, H22, H12,
                    eps1_diag, eps2_diag, k, nvec):
    """Run MATLAB BEMRetLayer initmat.m on Python-supplied Green matrices.

    Parameters
    ----------
    G11, G21, H11, H21 : (n, n) complex
        Inner-surface Green-function matrices.
    G22, H22 : dict of (n, n) complex
        Outer-surface Green-function matrices, with keys
        'ss', 'hh', 'p', 'sh', 'hs'. Missing/zero entries should be
        passed as zero matrices.
    G12, H12 : (n, n) complex
        Cross outer scalar Green matrices.
    eps1_diag, eps2_diag : (n,) array
        Per-face dielectric function values inside/outside.
    k : float
        Wavenumber in vacuum.
    nvec : (n, 3) real
        Outer surface normals.

    Returns
    -------
    dict with keys G1, G2_ss, G2_hh, G2_p, G2_sh, G2_hs,
        G2e_*, H2_*, H2e_*, G1i, G2pi, Sigma1, Sigma1e, Sigma2p,
        L1, L2p, Gamma, Gammapar, m_full.
    """
    import matlab as _matlab_mod

    eng = get_engine()

    n = G11.shape[0]

    def _zero_if_scalar(x):
        if isinstance(x, np.ndarray):
            return x
        return np.zeros((n, n), dtype=complex)

    def _struct_or_zero(d, key):
        if isinstance(d, dict) and key in d and isinstance(d[key], np.ndarray):
            return d[key]
        return np.zeros((n, n), dtype=complex)

    # Build the input struct on the MATLAB side via workspace
    items = {
        'G11': G11, 'G21': _zero_if_scalar(G21),
        'H11': H11, 'H21': _zero_if_scalar(H21),
        'G22_ss': _struct_or_zero(G22, 'ss'),
        'G22_hh': _struct_or_zero(G22, 'hh'),
        'G22_p':  _struct_or_zero(G22, 'p'),
        'G22_sh': _struct_or_zero(G22, 'sh'),
        'G22_hs': _struct_or_zero(G22, 'hs'),
        'H22_ss': _struct_or_zero(H22, 'ss'),
        'H22_hh': _struct_or_zero(H22, 'hh'),
        'H22_p':  _struct_or_zero(H22, 'p'),
        'H22_sh': _struct_or_zero(H22, 'sh'),
        'H22_hs': _struct_or_zero(H22, 'hs'),
        'G12': _zero_if_scalar(G12),
        'H12': _zero_if_scalar(H12),
    }

    # Push all matrices into the workspace as real/imag pairs and
    # rebuild as complex inside MATLAB
    for name, A in items.items():
        A = np.asarray(A, dtype=complex)
        eng.workspace[name + '_r'] = _matlab_mod.double(np.real(A).tolist())
        eng.workspace[name + '_i'] = _matlab_mod.double(np.imag(A).tolist())
        eng.eval('{0} = complex({0}_r, {0}_i);'.format(name), nargout=0)

    # Eps diag (real or complex)
    e1 = np.asarray(eps1_diag, dtype=complex).ravel()
    e2 = np.asarray(eps2_diag, dtype=complex).ravel()
    eng.workspace['eps1_diag_r'] = _matlab_mod.double(np.real(e1).tolist())
    eng.workspace['eps1_diag_i'] = _matlab_mod.double(np.imag(e1).tolist())
    eng.workspace['eps2_diag_r'] = _matlab_mod.double(np.real(e2).tolist())
    eng.workspace['eps2_diag_i'] = _matlab_mod.double(np.imag(e2).tolist())
    eng.eval('eps1_diag = complex(eps1_diag_r(:), eps1_diag_i(:));', nargout=0)
    eng.eval('eps2_diag = complex(eps2_diag_r(:), eps2_diag_i(:));', nargout=0)

    eng.workspace['k_val'] = float(k)
    eng.workspace['nvec'] = _matlab_mod.double(np.asarray(nvec, dtype=float).tolist())

    # Build input struct
    eng.eval(
        "in_struct = struct(" +
        ", ".join("'{0}', {0}".format(name) for name in (
            list(items.keys()) + ['eps1_diag', 'eps2_diag', 'nvec'])) +
        ", 'k', k_val);", nargout=0)

    # Run helper
    eng.eval('out_struct = mnpbem_bem_init_helper(in_struct);', nargout=0)

    # Pull outputs
    keys = [
        'G1', 'G1i', 'G2pi',
        'G2_ss', 'G2_hh', 'G2_p', 'G2_sh', 'G2_hs',
        'G2e_ss', 'G2e_hh', 'G2e_p', 'G2e_sh', 'G2e_hs',
        'H2_ss', 'H2_hh', 'H2_p', 'H2_sh', 'H2_hs',
        'H2e_ss', 'H2e_hh', 'H2e_p', 'H2e_sh', 'H2e_hs',
        'Sigma1', 'Sigma1e', 'Sigma2p',
        'L1', 'L2p', 'Gamma', 'Gammapar', 'm_full',
    ]

    out = {}
    for key in keys:
        eng.eval('tmp_r = real(full(out_struct.{}));'.format(key), nargout=0)
        eng.eval('tmp_i = imag(full(out_struct.{}));'.format(key), nargout=0)
        r = np.asarray(eng.workspace['tmp_r'], dtype=float)
        i = np.asarray(eng.workspace['tmp_i'], dtype=float)
        out[key] = r + 1j * i

    return out
