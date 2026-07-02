from mnpbem.materials.eps_drude import EpsDrude
from typing import Callable # typing isn't strictly necessary (handled by the internal state class)

# drude gold function from example file
def generate_eps_func() -> Callable[[float], tuple[complex, float]]:
    """
    Create Drude model for gold.

    Parameters from Johnson & Christy (approximate):
    eps_inf = 9.5, omega_p = 8.95 eV, gamma = 0.069 eV
    """
    # Drude parameters for gold (in nm units)
    eps_inf = 9.5       # High-frequency dielectric constant
    lambda_p = 138.0    # Plasma wavelength (nm) ~ 2*pi*c/omega_p
    gamma = 0.069       # Damping rate (eV) converted to nm^-1

    # gamma in eV -> nm^-1: gamma_nm = gamma_eV * 2*pi*c / (hc)
    # hc = 1240 eV*nm, so gamma_nm = gamma_eV / 1240 * 2*pi*c
    # Simpler: use wavelength-based damping
    lambda_gamma = 1240.0 / gamma  # ~ 17971 nm

    return EpsDrude(eps_inf, lambda_p, lambda_gamma)