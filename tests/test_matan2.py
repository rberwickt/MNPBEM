"""Tests for matan2: MATLAB-bit-identical atan2 via libmwmathutil.

Verification strategy:
- A reference set of (y, x) -> atan2(y, x) triples was generated in MATLAB
  R2025b and stored as raw float64 binary in tests/data/matlab_atan2.bin.
- matan2 must reproduce every reference value bit-for-bit.
- Edge cases (signed zeros, inf, NaN, quadrant boundaries) are explicit.
"""
import os

import numpy as np
import pytest

from mnpbem.utils.matlab_compat import (
    matan2,
    _MATLAB_ATAN2_AVAILABLE,
    _matan2_scalar,
)


DATA_PATH = os.path.join(
    os.path.dirname(__file__), 'data', 'matlab_atan2.bin'
)


def _bits(a):
    return np.asarray(a, dtype=np.float64).view(np.uint64)


@pytest.mark.skipif(not _MATLAB_ATAN2_AVAILABLE,
                    reason='MATLAB libmwmathutil.so not installed')
def test_matan2_reference_bit_identical():
    """40k MATLAB-generated reference values must all match bit-for-bit."""
    assert os.path.exists(DATA_PATH), (
        'reference data missing; regenerate with tests/data/gen_matlab_atan2.m'
    )
    data = np.fromfile(DATA_PATH, dtype=np.float64)
    n = data.size // 3
    y = data[:n]
    x = data[n:2 * n]
    r_ref = data[2 * n:]
    r = matan2(y, x)
    diff = np.sum(_bits(r) != _bits(r_ref))
    assert diff == 0, '{} / {} samples differ (bit-level)'.format(diff, n)


@pytest.mark.skipif(not _MATLAB_ATAN2_AVAILABLE,
                    reason='MATLAB libmwmathutil.so not installed')
def test_matan2_quadrants():
    """Matches np.arctan2 signs in all four quadrants."""
    y = np.array([1.0, 1.0, -1.0, -1.0])
    x = np.array([1.0, -1.0, -1.0, 1.0])
    r = matan2(y, x)
    expected_signs = np.array([1, 1, -1, -1])
    assert np.all(np.sign(r) == expected_signs)


@pytest.mark.skipif(not _MATLAB_ATAN2_AVAILABLE,
                    reason='MATLAB libmwmathutil.so not installed')
def test_matan2_axes_and_origin():
    """Special values on the axes."""
    assert matan2(0.0, 1.0) == 0.0
    assert matan2(0.0, -1.0) == np.pi
    assert matan2(1.0, 0.0) == np.pi / 2
    assert matan2(-1.0, 0.0) == -np.pi / 2


@pytest.mark.skipif(not _MATLAB_ATAN2_AVAILABLE,
                    reason='MATLAB libmwmathutil.so not installed')
def test_matan2_matlab_signed_zero_policy():
    """MATLAB's atan2 ignores signed-zero input and always returns +0
    when both arguments are any zero. This differs from IEEE 754 /
    np.arctan2 but is what MATLAB ships — we must match it bit-for-bit.
    """
    neg_zero = -1.0 / np.inf
    for y, x in [(0.0, 0.0),
                 (0.0, neg_zero),
                 (neg_zero, 0.0),
                 (neg_zero, neg_zero)]:
        r = matan2(y, x)
        # MATLAB returns plain +0 regardless of input sign.
        assert r == 0.0
        assert not np.signbit(np.asarray(r, dtype=np.float64))


@pytest.mark.skipif(not _MATLAB_ATAN2_AVAILABLE,
                    reason='MATLAB libmwmathutil.so not installed')
def test_matan2_inf_nan():
    """Inf / NaN propagation."""
    assert np.isnan(matan2(np.nan, 1.0))
    assert np.isnan(matan2(1.0, np.nan))
    assert matan2(np.inf, 3.0) == np.pi / 2
    assert matan2(-np.inf, 3.0) == -np.pi / 2


@pytest.mark.skipif(not _MATLAB_ATAN2_AVAILABLE,
                    reason='MATLAB libmwmathutil.so not installed')
def test_matan2_scalar_and_array():
    """Scalar/array produce bit-identical values."""
    y = np.random.default_rng(0).standard_normal(500)
    x = np.random.default_rng(1).standard_normal(500)
    r_arr = matan2(y, x)
    r_scalar = np.array([_matan2_scalar(float(yi), float(xi))
                          for yi, xi in zip(y, x)])
    assert np.all(_bits(r_arr) == _bits(r_scalar))


@pytest.mark.skipif(not _MATLAB_ATAN2_AVAILABLE,
                    reason='MATLAB libmwmathutil.so not installed')
def test_matan2_broadcasting():
    """Accepts broadcasting shapes."""
    y = np.array([1.0, 2.0, 3.0])
    x = 2.0
    r = matan2(y, x)
    assert r.shape == (3,)
    for i in range(3):
        expected = matan2(float(y[i]), 2.0)
        a = np.float64(r[i]).view(np.uint64)
        b = np.float64(expected).view(np.uint64)
        assert a == b


@pytest.mark.skipif(not _MATLAB_ATAN2_AVAILABLE,
                    reason='MATLAB libmwmathutil.so not installed')
def test_matan2_differs_from_numpy_in_measurable_fraction():
    """Sanity: MATLAB and np.arctan2 disagree by exactly 1 ULP in a
    measurable fraction of random inputs.
    """
    rng = np.random.default_rng(42)
    y = rng.standard_normal(10000) * 50
    x = rng.standard_normal(10000) * 50
    r_m = matan2(y, x)
    r_np = np.arctan2(y, x)
    diff = np.sum(_bits(r_m) != _bits(r_np))
    assert diff > len(y) * 0.05
    ulp_delta = np.abs(_bits(r_m).astype(np.int64) -
                       _bits(r_np).astype(np.int64))
    assert ulp_delta.max() <= 1


def test_matan2_fallback_when_unavailable(monkeypatch):
    """When MATLAB lib isn't available, matan2 falls back to np.arctan2."""
    import mnpbem.utils.matlab_compat as mc
    monkeypatch.setattr(mc, '_MATLAB_ATAN2_AVAILABLE', False)
    y = np.array([0.3, -1.2, 4.5])
    x = np.array([0.7,  2.1, -0.8])
    assert np.all(mc.matan2(y, x) == np.arctan2(y, x))


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-v']))
