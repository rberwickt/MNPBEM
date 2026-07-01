"""
Tests for EpsNonlocal — hydrodynamic Drude nonlocal cover-layer
dielectric function (Yu Luo et al., PRL 111, 093901 (2013)).
"""

import numpy as np
import pytest

from mnpbem.materials import (
    EpsConst, EpsDrude, EpsTable, EpsFun, EpsNonlocal, make_nonlocal_pair,
)
from mnpbem.utils.constants import EV2NM


def test_gold_factory_basic():
    eps_b = EpsConst(1.0)
    eps_t = EpsNonlocal.gold(eps_b, delta_d = 0.05)

    assert eps_t.name == 'Au'
    assert eps_t.delta_d == pytest.approx(0.05)
    assert eps_t.eps_inf == pytest.approx(10.0)
    assert eps_t.omega_p > 8.5
    assert eps_t.omega_p < 9.5
    assert eps_t.gamma > 0.0
    assert eps_t.beta > 0.0


def test_silver_factory():
    eps_t = EpsNonlocal.silver(delta_d = 0.04)
    assert eps_t.name == 'Ag'
    assert eps_t.eps_inf == pytest.approx(3.3)


def test_aluminum_factory():
    eps_t = EpsNonlocal.aluminum(delta_d = 0.03)
    assert eps_t.name == 'Al'
    assert eps_t.eps_inf == pytest.approx(1.0)


def test_call_returns_tuple_with_complex_eps_and_k():
    eps_t = EpsNonlocal.gold()
    val, k = eps_t(600.0)
    assert np.iscomplexobj(val)
    assert np.iscomplexobj(k)
    # eps_t is small (cover layer is thin) but non-zero
    assert abs(val) > 0.0
    assert abs(val) < 1.0


def test_call_array_input():
    eps_t = EpsNonlocal.gold()
    ws = np.linspace(400.0, 800.0, 9)
    val, k = eps_t(ws)
    assert val.shape == ws.shape
    assert k.shape == ws.shape
    assert np.iscomplexobj(val)


def test_q_longitudinal_matches_formula():
    eps_t = EpsNonlocal.gold()

    # at enei = 600 nm:  omega = EV2NM / 600
    enei = 600.0
    omega = EV2NM / enei
    radicand = (eps_t.omega_p ** 2) / eps_t.eps_inf - omega * (omega + 1j * eps_t.gamma)
    expected = np.sqrt(radicand) / eps_t.beta
    actual = eps_t.q_longitudinal(enei)
    assert np.isclose(actual, expected, rtol = 1e-12, atol = 1e-12)


def test_eps_t_matches_yu_luo_formula_directly():
    """eps_t = (eps_m * eps_b) / (eps_m - eps_b) * q_L * delta_d"""
    eps_b = EpsConst(2.25)              # water-ish
    eps_m = EpsDrude.gold()
    eps_t = EpsNonlocal(eps_m, eps_b, delta_d = 0.07)

    enei = np.array([500.0, 600.0, 700.0])
    val, _ = eps_t(enei)

    em, _ = eps_m(enei)
    eb, _ = eps_b(enei)
    expected = (em * eb) / (em - eb) * eps_t.q_longitudinal(enei) * eps_t.delta_d
    assert np.allclose(val, expected, rtol = 1e-12, atol = 1e-12)


def test_delta_d_scales_linearly():
    eps_b = EpsConst(1.0)
    eps_m = EpsDrude.gold()

    eps_a = EpsNonlocal(eps_m, eps_b, delta_d = 0.05)
    eps_b_layer = EpsNonlocal(eps_m, eps_b, delta_d = 0.10)

    enei = np.array([500.0, 600.0, 700.0])
    val_a, _ = eps_a(enei)
    val_b, _ = eps_b_layer(enei)
    # eps_t is linear in delta_d
    assert np.allclose(val_b, 2.0 * val_a, rtol = 1e-12, atol = 1e-12)


def test_beta_overrides_default():
    eps_b = EpsConst(1.0)
    eps_m = EpsDrude.gold()

    custom_beta = 0.0036 * (1.0 / 8.0655477e-4)   # MATLAB demo value (eV*nm)
    eps_t = EpsNonlocal(eps_m, eps_b, delta_d = 0.05, beta = custom_beta)
    assert eps_t.beta == pytest.approx(custom_beta)


def test_make_nonlocal_pair_returns_consistent_objects():
    eps_b = EpsConst(1.5)
    core, shell = make_nonlocal_pair('gold', eps_embed = eps_b, delta_d = 0.05)
    assert isinstance(core, EpsDrude)
    assert isinstance(shell, EpsNonlocal)
    assert shell.delta_d == pytest.approx(0.05)
    assert shell.name == 'Au'


def test_make_nonlocal_pair_with_table_override():
    """Allow Johnson-Christy (or similar) tabulated metal as the inner core
    while the longitudinal correction parameters still come from the
    canonical Drude model."""
    eps_b = EpsConst(1.0)
    eps_table = EpsTable('gold.dat')
    core, shell = make_nonlocal_pair('gold', eps_embed = eps_b, delta_d = 0.05,
            eps_metal = eps_table)
    assert core is eps_table
    # ensure nonlocal still has Drude omega_p / gamma / eps_inf:
    drude_au = EpsDrude.gold()
    assert shell.eps_inf == pytest.approx(drude_au.eps0)
    assert shell.omega_p == pytest.approx(drude_au.wp)
    assert shell.gamma == pytest.approx(drude_au.gammad)


def test_invalid_inputs():
    eps_m = EpsDrude.gold()
    eps_b = EpsConst(1.0)

    with pytest.raises(ValueError):
        EpsNonlocal(None, eps_b)
    with pytest.raises(ValueError):
        EpsNonlocal(eps_m, None)
    with pytest.raises(ValueError):
        EpsNonlocal(eps_m, eps_b, delta_d = 0.0)
    with pytest.raises(ValueError):
        EpsNonlocal(eps_m, eps_b, delta_d = -0.1)


def test_unknown_metal_for_beta_default():
    """If the metal eps cannot supply beta (no Fermi velocity table) and the
    user did not pass beta explicitly, EpsNonlocal must raise."""
    eps_b = EpsConst(1.0)
    # Use Drude with custom name not in the table
    eps_m = EpsDrude(eps0 = 5.0, wp = 7.0, gammad = 0.05, name = 'Cu')

    with pytest.raises(ValueError):
        EpsNonlocal(eps_m, eps_b, delta_d = 0.05)


def test_wavenumber_helper():
    eps_t = EpsNonlocal.gold()
    enei = 550.0
    val, k_full = eps_t(enei)
    k_via_helper = eps_t.wavenumber(enei)
    assert np.isclose(k_full, k_via_helper, rtol = 1e-14, atol = 1e-14)


def test_repr_and_str():
    eps_t = EpsNonlocal.gold(delta_d = 0.05)
    s_repr = repr(eps_t)
    s_str = str(eps_t)
    assert 'EpsNonlocal' in s_repr
    assert 'Au' in s_str
    assert 'delta_d' in s_str


def test_top_level_export_available():
    """EpsNonlocal must be importable from the top-level `mnpbem` package."""
    import mnpbem
    assert hasattr(mnpbem, 'EpsNonlocal')
    assert hasattr(mnpbem, 'make_nonlocal_pair')


def test_matlab_demo_formula_consistency():
    """Reproduce the formula used in MATLAB demospecstat19.m / bem_ug_coverlayer.m

    eps3 = epsfun(@(enei) eps2(enei) .* eps1(enei) ./ ...
                  (eps2(enei) - eps1(enei)) .* ql(eV2nm./enei) * d);

    with ql(w) = 2*pi * sqrt(3.3^2 - w*(w + 1i*0.165)) / (0.0036 * eV2nm).

    EpsNonlocal must match that exactly when configured with the same
    parameters: eps_inf=3.3, omega_p=sqrt(3.3) (so omega_p^2/eps_inf = 1
    -- actually demo uses 1 - 3.3^2/(w(w+0.165i)) so the prefactor in
    epsfun-form is 3.3^2 / 1, i.e. omega_p^2 = 3.3^2 and eps_inf = 1; and
    the bare Drude eps_m = 1 - 3.3^2/(w(w+0.165i)) so eps_inf_in_drude = 1).
    """
    # demo parameters
    eV2nm = 1.0 / 8.0655477e-4
    delta_d = 0.05
    beta = 0.0036 * eV2nm   # eV * nm
    gamma = 0.165           # eV
    wp = 3.3                # eV
    eps_inf = 1.0           # the "1" in eps_m = 1 - wp^2/(w(w+ig))

    eps_b = EpsConst(1.0)
    eps_m = EpsDrude(eps0 = eps_inf, wp = wp, gammad = gamma, name = 'demo')
    # Provide beta explicitly because 'demo' is not in the v_F table.
    eps_t = EpsNonlocal(eps_m, eps_b,
            delta_d = delta_d,
            eps_inf = eps_inf,
            omega_p = wp,
            gamma = gamma,
            beta = beta)

    # MATLAB demo formula evaluated at e.g. enei = 500 nm
    enei = np.array([400.0, 500.0, 600.0, 700.0, 800.0])
    omega = eV2nm / enei
    ql_demo = np.sqrt(wp ** 2 - omega * (omega + 1j * gamma)) / beta
    em_demo = 1.0 - wp ** 2 / (omega * (omega + 1j * gamma))
    eb_demo = 1.0 + 0.0 * omega   # eps_b = 1
    eps3_demo = (em_demo * eb_demo) / (em_demo - eb_demo) * ql_demo * delta_d

    eps3_ours, _ = eps_t(enei)

    assert np.allclose(eps3_ours, eps3_demo, rtol = 1e-12, atol = 1e-12)


def test_shell_eps_combined_with_coverlayer_in_comparticle():
    """Smoke-test geometry+materials combination: thin gold sphere with
    shifted cover layer using EpsNonlocal — instantiate ComParticle and
    verify epstab indexing is sane.
    """
    from mnpbem.geometry import ComParticle, trisphere
    from mnpbem.greenfun import coverlayer

    eps_b = EpsConst(1.0)
    core_eps, shell_eps = make_nonlocal_pair('gold',
            eps_embed = eps_b,
            delta_d = 0.05)

    diameter = 10.0
    delta_d = shell_eps.delta_d
    # inner Drude core
    p_core = trisphere(144, diameter - 2 * delta_d)
    # artificial outer shell via coverlayer.shift
    p_shell = coverlayer.shift(p_core, delta_d)

    # epstab : [embedding, core_drude, nonlocal_shell]
    epstab = [eps_b, core_eps, shell_eps]
    # MATLAB index: [shell <-> embed], [core <-> shell] = [3, 1; 2, 3]
    inds = [[3, 1], [2, 3]]
    p = ComParticle(epstab, [p_shell, p_core], inds, 1, 2)

    # basic invariants
    assert len(p.eps) == 3
    assert p.eps[2] is shell_eps
    assert p.eps[1] is core_eps

    # `coverlayer.refine` must still construct a callable refun for [1,2]
    refun = coverlayer.refine(p, [[1, 2]])
    assert callable(refun)
