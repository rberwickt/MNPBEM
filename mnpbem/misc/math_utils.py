"""
Mathematical utility functions for MNPBEM.

MATLAB: Misc/inner.m, outer.m, matcross.m, matmul.m, vecnorm.m, vecnormalize.m, spdiag.m
"""

import numpy as np
from scipy import sparse
from typing import Optional, Union


def matmul(a: np.ndarray, x: np.ndarray) -> Union[np.ndarray, int]:
    """
    MATLAB: Misc/matmul.m

    Generalized matrix multiplication for tensors. The matrix multiplication
    is performed along the last dimension of A and the first dimension of X.
    """
    if np.isscalar(a) or (isinstance(a, np.ndarray) and a.size == 1):
        a_val = float(a) if isinstance(a, np.ndarray) else a
        if a_val == 0:
            return 0
        else:
            return a_val * x
    elif np.isscalar(x) or (isinstance(x, np.ndarray) and x.size == 1):
        x_val = float(x) if isinstance(x, np.ndarray) else x
        if x_val == 0:
            return 0
        else:
            return a * x_val
    else:
        siza = a.shape
        sizx = x.shape

        if len(siza) == 2 and siza[-1] != sizx[0]:
            # diagonal multiplication case
            y = np.diag(a).reshape(-1, 1) * x.reshape(sizx[0], -1)
            return y.reshape(sizx)
        else:
            # combined size
            siz = siza[:-1] + sizx[1:]
            y = a.reshape(-1, siza[-1]) @ x.reshape(sizx[0], -1)
            if len(siz) == 0:
                return y.item()
            return y.reshape(siz)


def inner(nvec: np.ndarray,
        a: np.ndarray,
        mul: Optional[np.ndarray] = None) -> Union[np.ndarray, int]:
    """
    MATLAB: Misc/inner.m

    Inner product between a vector and a matrix or tensor.
    The generalized dot product is defined with respect to the second dimension.
    If mul is provided, a = matmul(mul, a) before the inner product.
    """
    if nvec.shape[0] != a.shape[0]:
        return 0

    if mul is not None:
        a = matmul(mul, a)

    if a.ndim == 2:
        val = np.sum(nvec * a, axis = 1)
    else:
        siz = a.shape
        # expand nvec to match a's dimensions
        expand_shape = (siz[0], siz[1]) + (1,) * (a.ndim - 2)
        nvec_exp = nvec.reshape(expand_shape)
        nvec_exp = np.broadcast_to(nvec_exp, a.shape)
        val = np.sum(nvec_exp * a, axis = 1)
        # squeeze to remove the vector dimension
        new_shape = (siz[0],) + siz[2:]
        val = val.reshape(new_shape)

    return val


def outer(nvec: np.ndarray,
        val: np.ndarray,
        mul: Optional[np.ndarray] = None) -> Union[np.ndarray, int]:
    """
    MATLAB: Misc/outer.m

    Outer product between vector and tensor.
    a(i, k, j, ...) = nvec(i, k) * val(i, j, ...)
    If mul is provided, val = matmul(mul, val) before the outer product.
    """
    if mul is not None:
        val = matmul(mul, val)

    if nvec.shape[0] != val.shape[0] if hasattr(val, 'shape') else True:
        return 0

    if np.isscalar(val) and val == 0:
        return 0

    n = nvec.shape[0]

    if val.ndim == 1:
        # val is (n,) -> result is (n, 3)
        result = nvec * val[:, np.newaxis]
    else:
        # val is (n, j, ...) -> result is (n, 3, j, ...)
        siz = val.shape
        result_shape = (n, 3) + siz[1:]
        result = np.empty(result_shape, dtype = np.result_type(nvec, val))
        extra_dims = val.ndim - 1
        for k in range(3):
            # element-wise: nvec[:, k] * val for each row
            expand_shape = (n,) + (1,) * extra_dims
            result[:, k] = nvec[:, k].reshape(expand_shape) * val

    return result


def matcross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    MATLAB: Misc/matcross.m

    Generalized cross product for tensors.
    a: (n, 3), b: (n, 3, ...)
    Returns cross product with same dimensions as b.
    """
    siz = list(b.shape)
    siz[1] = 1

    def fun(i: int, j: int) -> np.ndarray:
        bj = b[:, j]
        if bj.ndim > 1:
            bj = bj.squeeze()
        result = a[:, i].reshape(-1, *([1] * (bj.ndim - 1))) * bj
        return result.reshape(siz)

    c0 = fun(1, 2) - fun(2, 1)
    c1 = fun(2, 0) - fun(0, 2)
    c2 = fun(0, 1) - fun(1, 0)

    # stack along axis 1
    total = c0.shape[0] + c1.shape[0] + c2.shape[0]
    result_shape = list(c0.shape)
    result_shape[1] = 3
    result = np.empty(result_shape, dtype = np.result_type(a, b))
    result[:, 0:1] = c0
    result[:, 1:2] = c1
    result[:, 2:3] = c2

    return result


def vec_norm(v: np.ndarray, key: Optional[str] = None) -> np.ndarray:
    """
    MATLAB: Misc/vecnorm.m

    Norm of vector array.
    v: array of size (n, 3, ...)
    Returns norm array of size (n, ...) or max if key='max'.
    """
    n = np.squeeze(np.sqrt(np.sum(np.abs(v) ** 2, axis = 1)))

    if key is not None and key == 'max':
        n = np.max(n)

    return n


def vec_normalize(v: np.ndarray,
        v2: Optional[np.ndarray] = None,
        key: Optional[str] = None) -> np.ndarray:
    """
    MATLAB: Misc/vecnormalize.m

    Normalize vector array.
    v: array of size (n, 3, ...)
    v2: if provided, normalize v using v2's norm
    key: '' (default), 'max', or 'max2'
    """
    if v2 is None:
        v2 = v
    if key is None:
        key = ''

    n = np.sqrt(np.sum(np.abs(v2) ** 2, axis = 1, keepdims = True))

    if key == 'max':
        v = v / np.max(n)
    elif key == 'max2':
        # normalize v(:,:,i) for each i
        max_n = np.max(n, axis = 0, keepdims = True)
        v = v / max_n
    else:
        v = v / n

    return v


def spdiag(a: np.ndarray) -> sparse.spmatrix:
    """
    MATLAB: Misc/spdiag.m

    Put array values on the diagonal of a sparse matrix.
    """
    a = np.asarray(a).ravel()
    n = len(a)
    return sparse.diags(a, 0, shape = (n, n), format = 'csc')
