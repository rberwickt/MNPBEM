import numpy as np
from typing import Tuple, Optional, Any

try:
    import fmm3dpy
    _HAS_FMM3D = True
except ImportError:
    _HAS_FMM3D = False


def fmm_available() -> bool:
    return _HAS_FMM3D


def _to_complex_charges(charges: np.ndarray) -> np.ndarray:
    arr = np.ascontiguousarray(charges, dtype=np.complex128)
    return arr


def _scalar_potential_grad(
        zk: complex,
        sources: np.ndarray,
        targets: np.ndarray,
        charges: Optional[np.ndarray] = None,
        dipvec: Optional[np.ndarray] = None,
        eps: float = 1e-12) -> Tuple[np.ndarray, np.ndarray]:

    src = np.ascontiguousarray(sources.T, dtype=np.float64)  # (3, n_src)
    tgt = np.ascontiguousarray(targets.T, dtype=np.float64)  # (3, n_tgt)

    kwargs = dict(eps = eps, zk = complex(zk), sources = src, targets = tgt, pgt = 2)
    if charges is not None:
        kwargs['charges'] = _to_complex_charges(charges)
    if dipvec is not None:
        kwargs['dipvec'] = np.ascontiguousarray(dipvec, dtype=np.complex128)

    out = fmm3dpy.hfmm3d(**kwargs)

    pot = np.asarray(out.pottarg)               # (n_tgt,)
    grad = np.asarray(out.gradtarg).T           # (n_tgt, 3)
    return pot, grad


def eval_freespace_green(
        zk: complex,
        src_pos: np.ndarray,
        src_area: np.ndarray,
        tgt_pos: np.ndarray,
        sigma: np.ndarray,
        eps: float = 1e-12) -> Tuple[np.ndarray, np.ndarray]:

    assert _HAS_FMM3D, '[error] <fmm3dpy> not installed'
    assert src_pos.shape[0] == src_area.shape[0], '[error] <src_pos>/<src_area> mismatch'
    assert src_pos.shape[0] == sigma.shape[0], '[error] <sigma> length mismatch'

    weighted = src_area.astype(np.complex128) * sigma.astype(np.complex128)
    pot, grad = _scalar_potential_grad(zk, src_pos, tgt_pos, charges = weighted, eps = eps)

    four_pi = 4.0 * np.pi
    pot = pot * four_pi
    grad = grad * four_pi

    return pot, grad


def eval_freespace_field(
        zk: complex,
        k_wave: complex,
        src_pos: np.ndarray,
        src_area: np.ndarray,
        tgt_pos: np.ndarray,
        sig_scalar: np.ndarray,
        h_current: np.ndarray,
        eps: float = 1e-12) -> Tuple[np.ndarray, np.ndarray]:

    assert _HAS_FMM3D, '[error] <fmm3dpy> not installed'

    n_src = src_pos.shape[0]
    n_tgt = tgt_pos.shape[0]
    four_pi = 4.0 * np.pi

    sig_scalar = np.atleast_1d(sig_scalar)
    if sig_scalar.ndim > 1:
        sig_scalar = sig_scalar[..., 0]

    h_current = np.asarray(h_current, dtype=np.complex128)
    if h_current.ndim == 2 and h_current.shape[1] == 3:
        h_xyz = h_current
    elif h_current.ndim == 3 and h_current.shape[1] == 3:
        h_xyz = h_current[..., 0]
    else:
        raise ValueError('[error] Invalid <h_current> shape: {}'.format(h_current.shape))

    src_w = np.ascontiguousarray(src_pos.T, dtype=np.float64)
    tgt_w = np.ascontiguousarray(tgt_pos.T, dtype=np.float64)

    area = src_area.astype(np.complex128)

    chg_sig = area * sig_scalar.astype(np.complex128)
    out_sig = fmm3dpy.hfmm3d(eps = eps, zk = complex(zk), sources = src_w, charges = chg_sig, targets = tgt_w, pgt = 2)
    grad_phi = np.asarray(out_sig.gradtarg).T * four_pi   # (n_tgt, 3)

    a_field = np.empty((n_tgt, 3), dtype=np.complex128)
    h_field = np.zeros((n_tgt, 3), dtype=np.complex128)

    for axis in range(3):
        chg_a = area * h_xyz[:, axis]
        out_a = fmm3dpy.hfmm3d(eps = eps, zk = complex(zk), sources = src_w, charges = chg_a, targets = tgt_w, pgt = 2)
        a_field[:, axis] = np.asarray(out_a.pottarg) * four_pi
        grad_a = np.asarray(out_a.gradtarg).T * four_pi   # (n_tgt, 3): d/dx_target a_axis
        if axis == 0:
            h_field[:, 1] += grad_a[:, 2]
            h_field[:, 2] -= grad_a[:, 1]
        elif axis == 1:
            h_field[:, 0] -= grad_a[:, 2]
            h_field[:, 2] += grad_a[:, 0]
        else:
            h_field[:, 0] += grad_a[:, 1]
            h_field[:, 1] -= grad_a[:, 0]

    e_field = 1j * k_wave * a_field - grad_phi

    return e_field, h_field


def eval_freespace_potential(
        zk: complex,
        src_pos: np.ndarray,
        src_area: np.ndarray,
        tgt_pos: np.ndarray,
        sig_scalar: np.ndarray,
        eps: float = 1e-12) -> np.ndarray:

    assert _HAS_FMM3D, '[error] <fmm3dpy> not installed'

    sig_scalar = np.atleast_1d(sig_scalar)
    if sig_scalar.ndim > 1:
        sig_scalar = sig_scalar[..., 0]

    weighted = src_area.astype(np.complex128) * sig_scalar.astype(np.complex128)
    pot, _ = _scalar_potential_grad(zk, src_pos, tgt_pos, charges = weighted, eps = eps)
    return pot * (4.0 * np.pi)
