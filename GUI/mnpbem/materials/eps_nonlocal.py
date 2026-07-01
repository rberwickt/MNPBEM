"""
Hydrodynamic Drude nonlocal dielectric function (Luo, Pendry et al.,
PRL 111, 093901 (2013); see also Ciraci et al., Science 337, 1072 (2012)
and Mortensen et al., Nat. Commun. 5, 3809 (2014)).

The hydrodynamic / nonlocal correction is mapped onto an *effective local*
problem by introducing a thin artificial cover layer of thickness Delta_d
on top of the metal boundary. The thin-shell permittivity is

    eps_t(omega) = (eps_m * eps_b) / (eps_m - eps_b) * q_L(omega) * Delta_d

with the longitudinal plasmon wavenumber

    q_L(omega) = sqrt(omega_p^2 / eps_inf - omega * (omega + i * gamma)) / beta

Here eps_m is the *local* Drude permittivity of the core metal, eps_b is the
embedding dielectric, eps_inf the ionic background of the metal, and beta the
hydrodynamic velocity (units of eV * nm so that beta * q_L is in eV).

Usage pattern (matches MATLAB demospecstat19.m / bem_ug_coverlayer.m):

    eps_b = EpsConst(1.0)
    eps_m = EpsDrude.gold()
    eps_t = EpsNonlocal(eps_m, eps_b, delta_d = 0.05)   # nm

    p2 = trisphere(144, diameter - 2 * 0.05)            # inner Drude core
    p1 = coverlayer.shift(p2, 0.05)                     # artificial shell
    p  = ComParticle([eps_b, eps_m, eps_t], [p1, p2],
                     [[3, 1], [2, 3]], inout = [1, 2], op = op)
    bem = BEMStat(p, op, refun = coverlayer.refine(p, [[1, 2]]))

MATLAB MNPBEM does not provide a dedicated `@epsnonlocal` class -- the same
effect is implemented by hand-rolling an `epsfun(@(enei) ...)` evaluating the
formula above. EpsNonlocal packages that pattern.
"""

import numpy as np

from ..utils.constants import EV2NM
from .eps_const import EpsConst
from .eps_drude import EpsDrude
from .eps_table import EpsTable


# hbar in eV * s, used for converting Fermi velocity (m/s) to beta (eV * nm)
_HBAR_EV_S = 6.582119569e-16

# Tabulated Fermi velocities (m/s) for common metals (Ashcroft / Mermin).
# beta = sqrt(3/5) * v_F  (Thomas-Fermi limit of the hydrodynamic model).
_FERMI_VELOCITY_M_S = {
    'au':       1.40e6,
    'gold':     1.40e6,
    'ag':       1.39e6,
    'silver':   1.39e6,
    'al':       2.03e6,
    'aluminum': 2.03e6,
    'aluminium':2.03e6,
}


def _beta_eV_nm_from_vF(v_F_m_s):
    """Hydrodynamic velocity beta = sqrt(3/5) * v_F, in (eV * nm).

    beta_eV_nm = hbar[eV*s] * (sqrt(3/5) * v_F[m/s]) * 1e9 [nm/m]
    """
    return _HBAR_EV_S * np.sqrt(3.0 / 5.0) * float(v_F_m_s) * 1.0e9


class EpsNonlocal(object):
    """
    Hydrodynamic Drude nonlocal effective-layer dielectric function.

    Encapsulates the Yu Luo / Pendry mapping of a nonlocal Drude metal onto a
    *local* problem with a thin artificial cover layer. Calling an
    EpsNonlocal instance with a wavelength returns the cover-layer
    permittivity (and the associated wavenumber), so it slots into the same
    epstab list as EpsConst / EpsTable / EpsDrude / EpsFun and is consumed by
    BEMStat / BEMRet without changes to the solver.

    Parameters
    ----------
    eps_metal : EpsDrude or EpsTable or EpsFun or EpsConst
        Local dielectric function of the *core* metal. Drude parameters
        (eps_inf, omega_p, gamma) are taken from this object when available
        (EpsDrude). When a table or arbitrary function is passed, the user
        must supply omega_p / gamma / eps_inf explicitly via the keyword
        arguments below (the table itself is then used only as the local
        eps_m in the Yu Luo formula).
    eps_embed : EpsConst or EpsFun or EpsTable
        Dielectric function of the embedding medium (eps_b in the formula).
    delta_d : float, default 0.05
        Thickness of the artificial cover layer in nm. Yu Luo et al. show
        that the mapping is exact in the limit delta_d -> 0; in practice
        0.02 - 0.1 nm is sufficient.
    eps_inf : float, optional
        Override for ionic background. Default: pulled from eps_metal.eps0
        when eps_metal is EpsDrude.
    omega_p : float, optional
        Override for plasma frequency in eV. Default: from eps_metal.wp.
    gamma : float, optional
        Override for damping in eV. Default: from eps_metal.gammad.
    beta : float, optional
        Hydrodynamic velocity in eV * nm. Default: derived from the metal
        name via beta = sqrt(3/5) * v_F * hbar; v_F lookup table covers
        gold / silver / aluminum. Pass explicitly for other metals.
    name : str, optional
        Display name (e.g., 'Au-nonlocal'). When set, it also drives the
        Fermi velocity lookup if `beta` is not given.

    Examples
    --------
    >>> eps_b = EpsConst(1.0)
    >>> eps_m = EpsDrude.gold()
    >>> eps_t = EpsNonlocal(eps_m, eps_b, delta_d = 0.05)
    >>> eps_val, k = eps_t(600.0)        # at 600 nm
    >>> # gold factory shorthand:
    >>> eps_t = EpsNonlocal.gold(eps_b, delta_d = 0.05)

    Notes
    -----
    The eps_t returned by this class is the *artificial cover-layer*
    permittivity (the ε_3 of MATLAB demospecstat19). The core metal still
    uses its local Drude / Johnson-Christy permittivity. To set up a
    nonlocal BEM simulation the geometry must contain both:
      (i) the core Drude particle  (eps_metal),
      (ii) a coverlayer.shift'ed shell on top of it  (this EpsNonlocal).
    See `make_nonlocal_pair` for the convenience builder.
    """

    def __init__(self,
            eps_metal,
            eps_embed,
            delta_d = 0.05,
            eps_inf = None,
            omega_p = None,
            gamma = None,
            beta = None,
            name = None):

        if eps_metal is None:
            raise ValueError("[error] EpsNonlocal: <eps_metal> must not be None.")
        if eps_embed is None:
            raise ValueError("[error] EpsNonlocal: <eps_embed> must not be None.")
        if not callable(eps_metal):
            raise TypeError("[error] EpsNonlocal: <eps_metal> must be callable (an EpsConst/EpsTable/EpsDrude/EpsFun).")
        if not callable(eps_embed):
            raise TypeError("[error] EpsNonlocal: <eps_embed> must be callable (an EpsConst/EpsTable/EpsDrude/EpsFun).")

        delta_d = float(delta_d)
        if delta_d <= 0.0:
            raise ValueError("[error] EpsNonlocal: <delta_d> must be > 0 (got {}).".format(delta_d))

        # extract Drude parameters from the metal eps if available
        if eps_inf is None:
            eps_inf = getattr(eps_metal, 'eps0', None)
        if omega_p is None:
            omega_p = getattr(eps_metal, 'wp', None)
        if gamma is None:
            gamma = getattr(eps_metal, 'gammad', None)

        if eps_inf is None or omega_p is None or gamma is None:
            raise ValueError(
                "[error] EpsNonlocal: cannot infer Drude parameters from <eps_metal>={}; "
                "supply <eps_inf>, <omega_p>, <gamma> explicitly.".format(type(eps_metal).__name__))

        if beta is None:
            metal_name = name
            if metal_name is None:
                metal_name = getattr(eps_metal, 'name', None)
            if metal_name is None:
                raise ValueError(
                    "[error] EpsNonlocal: <beta> not given and metal <name> is unknown; "
                    "pass beta in eV*nm or supply <name> ('Au'/'Ag'/'Al').")
            v_F = _FERMI_VELOCITY_M_S.get(str(metal_name).lower(), None)
            if v_F is None:
                raise ValueError(
                    "[error] EpsNonlocal: no default Fermi velocity for <{}>; pass beta explicitly.".format(metal_name))
            beta = _beta_eV_nm_from_vF(v_F)

        self.eps_metal = eps_metal
        self.eps_embed = eps_embed
        self.delta_d = float(delta_d)
        self.eps_inf = float(eps_inf)
        self.omega_p = float(omega_p)
        self.gamma = float(gamma)
        self.beta = float(beta)
        self.name = name

    def q_longitudinal(self, enei):
        """
        Longitudinal plasmon wavenumber q_L(omega) in 1/nm.

        q_L = sqrt(omega_p^2 / eps_inf - omega * (omega + i * gamma)) / beta

        with omega = EV2NM / enei (eV) and beta in (eV * nm), giving q_L in
        1/nm.
        """
        enei = np.asarray(enei, dtype = float)
        omega = EV2NM / enei
        radicand = (self.omega_p ** 2) / self.eps_inf - omega * (omega + 1j * self.gamma)
        return np.sqrt(radicand) / self.beta

    def __call__(self, enei):
        """
        Evaluate the artificial cover-layer permittivity.

        eps_t(enei) = eps_m(enei) * eps_b(enei) / (eps_m(enei) - eps_b(enei))
                       * q_L(enei) * delta_d

        Returns (eps_t, k_t) with k_t = 2 pi / enei * sqrt(eps_t).
        """
        enei = np.asarray(enei, dtype = float)

        eps_m, _ = self.eps_metal(enei)
        eps_b, _ = self.eps_embed(enei)
        eps_m = np.asarray(eps_m, dtype = complex)
        eps_b = np.asarray(eps_b, dtype = complex)

        ql = self.q_longitudinal(enei)

        eps_t = (eps_m * eps_b) / (eps_m - eps_b) * ql * self.delta_d

        k = 2.0 * np.pi / enei * np.sqrt(eps_t)

        return eps_t, k

    def wavenumber(self, enei):
        _, k = self(enei)
        return k

    @classmethod
    def gold(cls,
            eps_embed = None,
            delta_d = 0.05,
            beta = None):
        if eps_embed is None:
            eps_embed = EpsConst(1.0)
        eps_metal = EpsDrude.gold()
        return cls(eps_metal, eps_embed, delta_d = delta_d, beta = beta, name = 'Au')

    @classmethod
    def silver(cls,
            eps_embed = None,
            delta_d = 0.05,
            beta = None):
        if eps_embed is None:
            eps_embed = EpsConst(1.0)
        eps_metal = EpsDrude.silver()
        return cls(eps_metal, eps_embed, delta_d = delta_d, beta = beta, name = 'Ag')

    @classmethod
    def aluminum(cls,
            eps_embed = None,
            delta_d = 0.05,
            beta = None):
        if eps_embed is None:
            eps_embed = EpsConst(1.0)
        eps_metal = EpsDrude.aluminum()
        return cls(eps_metal, eps_embed, delta_d = delta_d, beta = beta, name = 'Al')

    @classmethod
    def from_table(cls,
            eps_table,
            eps_embed,
            eps_inf,
            omega_p,
            gamma,
            beta,
            delta_d = 0.05,
            name = None):
        """
        Build an EpsNonlocal whose local metal contribution is taken from a
        tabulated table (e.g., Johnson-Christy gold). Drude parameters
        (eps_inf / omega_p / gamma) describe the *Drude part* of the metal
        and must be provided explicitly because they cannot be read off a
        table.
        """
        return cls(eps_table, eps_embed,
                delta_d = delta_d,
                eps_inf = eps_inf,
                omega_p = omega_p,
                gamma = gamma,
                beta = beta,
                name = name)

    def __repr__(self):
        return ("EpsNonlocal(name = {!r}, eps_inf = {}, omega_p = {}, gamma = {}, "
                "beta = {}, delta_d = {})".format(
                self.name, self.eps_inf, self.omega_p, self.gamma, self.beta, self.delta_d))

    def __str__(self):
        nm = "" if self.name is None else " ({})".format(self.name)
        return ("Hydrodynamic-Drude nonlocal cover layer{}: "
                "eps_inf = {}, wp = {} eV, gamma = {} eV, beta = {} eV*nm, "
                "delta_d = {} nm").format(nm, self.eps_inf, self.omega_p, self.gamma,
                self.beta, self.delta_d)


def make_nonlocal_pair(metal_name = 'gold',
        eps_embed = None,
        delta_d = 0.05,
        beta = None,
        eps_metal = None):
    """
    Convenience builder for a (eps_metal, eps_nonlocal_cover) pair.

    Parameters
    ----------
    metal_name : str
        'gold' / 'silver' / 'aluminum'. Used both for the core EpsDrude
        instance and for the default Fermi velocity in beta.
    eps_embed : EpsConst-like, optional
        Embedding dielectric. Defaults to vacuum (EpsConst(1.0)).
    delta_d : float
        Cover-layer thickness in nm.
    beta : float, optional
        Hydrodynamic velocity in eV * nm. Default: sqrt(3/5)*v_F*hbar from
        the table.
    eps_metal : callable, optional
        Override the core dielectric (e.g., pass an EpsTable for
        Johnson-Christy gold). Drude parameters are still pulled from the
        canonical EpsDrude.<metal>() so the longitudinal correction is
        well-defined.

    Returns
    -------
    eps_metal : callable
        Local-Drude (or override) permittivity for the inner Drude core.
    eps_nonlocal : EpsNonlocal
        Thin-shell permittivity for the artificial cover layer.

    Notes
    -----
    Build the geometry as:
        p_core  = trisphere(N, D - 2*delta_d)
        p_shell = coverlayer.shift(p_core, delta_d)
    and add `coverlayer.refine(p, ind)` to the BEM solver via the `refun`
    keyword.
    """
    if eps_embed is None:
        eps_embed = EpsConst(1.0)

    name = str(metal_name).lower()
    drude_factory = {
        'au': EpsDrude.gold,
        'gold': EpsDrude.gold,
        'ag': EpsDrude.silver,
        'silver': EpsDrude.silver,
        'al': EpsDrude.aluminum,
        'aluminum': EpsDrude.aluminum,
        'aluminium': EpsDrude.aluminum,
    }
    if name not in drude_factory:
        raise ValueError("[error] make_nonlocal_pair: unknown <metal_name>='{}'".format(metal_name))

    eps_drude = drude_factory[name]()
    eps_for_layer = eps_metal if eps_metal is not None else eps_drude
    eps_nl = EpsNonlocal(eps_for_layer, eps_embed,
            delta_d = delta_d,
            eps_inf = eps_drude.eps0,
            omega_p = eps_drude.wp,
            gamma = eps_drude.gammad,
            beta = beta,
            name = eps_drude.name)

    return (eps_for_layer, eps_nl)
