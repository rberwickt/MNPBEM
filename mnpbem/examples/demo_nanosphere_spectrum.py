"""
Demo: Gold Nanosphere Optical Spectrum Calculation

This example calculates the scattering, absorption, and extinction
cross sections for a gold nanosphere in vacuum.

MATLAB equivalent demo: demospecstat1.m, demospecret1.m

Note: For sphere on glass substrate, layer structure support is needed
(layerstructure, compgreenstatlayer, bemstatlayer classes).
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/home/user/MNPBEM')

from mnpbem.materials import EpsDrude, EpsConst
from mnpbem.geometry import trisphere, ComParticle
from mnpbem.bem import BEMStat, BEMRet
from mnpbem.excitation import PlaneWaveStat, PlaneWaveRet


def drude_gold():
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


def calculate_spectrum_quasistatic(radius=10.0, n_faces=144, wavelengths=None):
    """
    Calculate optical spectrum using quasistatic approximation.

    Good for particles much smaller than wavelength (r << lambda).

    Parameters
    ----------
    radius : float
        Sphere radius in nm
    n_faces : int
        Number of mesh faces
    wavelengths : array_like, optional
        Wavelengths in nm (default: 400-800 nm)

    Returns
    -------
    results : dict
        Contains wavelengths, sca, abs, ext
    """
    print("=" * 60)
    print("Quasistatic Calculation (BEMStat)")
    print("Sphere radius: {} nm, Mesh faces: {}".format(radius, n_faces))
    print("=" * 60)

    if wavelengths is None:
        wavelengths = np.linspace(400, 800, 41)

    # Materials: vacuum outside, gold inside
    eps_vac = EpsConst(1.0)
    eps_gold = drude_gold()
    eps_tab = [eps_vac, eps_gold]

    # Create gold sphere
    sphere = trisphere(n_faces, radius)
    p = ComParticle(eps_tab, [sphere], [[2, 1]])  # inside=gold(2), outside=vac(1)

    print("Created particle with {} faces".format(p.nfaces))

    # BEM solver
    bem = BEMStat(p)

    # Plane wave excitation (x-polarized)
    exc = PlaneWaveStat([1, 0, 0])

    # Calculate spectrum
    sca = np.zeros(len(wavelengths))
    abs_cs = np.zeros(len(wavelengths))
    ext = np.zeros(len(wavelengths))

    print("\nCalculating spectrum...")
    for i, enei in enumerate(wavelengths):
        # Get excitation and solve
        exc_dict = exc.potential(p, enei)
        sig = bem.solve(exc_dict)

        # Calculate cross sections
        sca[i] = exc.scattering(sig)[0]
        abs_cs[i] = exc.absorption(sig)[0]
        ext[i] = exc.extinction(sig)[0]

        if i % 10 == 0:
            print("  lambda = {:.0f} nm: ext = {:.2f} nm^2".format(enei, ext[i]))

    print("\nDone!")

    return {
        'wavelengths': wavelengths,
        'sca': sca,
        'abs': abs_cs,
        'ext': ext,
        'method': 'quasistatic'
    }


def calculate_spectrum_retarded(radius=50.0, n_faces=144, wavelengths=None):
    """
    Calculate optical spectrum using full Maxwell equations (retarded).

    Required for larger particles where retardation effects matter.

    Parameters
    ----------
    radius : float
        Sphere radius in nm
    n_faces : int
        Number of mesh faces
    wavelengths : array_like, optional
        Wavelengths in nm (default: 400-800 nm)

    Returns
    -------
    results : dict
        Contains wavelengths, sca, abs, ext
    """
    print("=" * 60)
    print("Retarded Calculation (BEMRet)")
    print("Sphere radius: {} nm, Mesh faces: {}".format(radius, n_faces))
    print("=" * 60)

    if wavelengths is None:
        wavelengths = np.linspace(400, 800, 41)

    # Materials
    eps_vac = EpsConst(1.0)
    eps_gold = drude_gold()
    eps_tab = [eps_vac, eps_gold]

    # Create gold sphere
    sphere = trisphere(n_faces, radius)
    p = ComParticle(eps_tab, [sphere], [[2, 1]])

    print("Created particle with {} faces".format(p.nfaces))

    # BEM solver
    bem = BEMRet(p)

    # Plane wave excitation (x-polarized, z-propagating)
    exc = PlaneWaveRet([1, 0, 0], [0, 0, 1])

    # Calculate spectrum
    sca = np.zeros(len(wavelengths))
    abs_cs = np.zeros(len(wavelengths))
    ext = np.zeros(len(wavelengths))

    print("\nCalculating spectrum...")
    for i, enei in enumerate(wavelengths):
        # Get excitation and solve
        exc_dict = exc.potential(p, enei)
        sig = bem.solve(exc_dict)

        # Calculate cross sections
        sca[i], _ = exc.scattering(sig)
        ext[i] = exc.extinction(sig)[0]
        abs_cs[i] = ext[i] - sca[i]

        if i % 10 == 0:
            print("  lambda = {:.0f} nm: ext = {:.2f} nm^2".format(enei, ext[i]))

    print("\nDone!")

    return {
        'wavelengths': wavelengths,
        'sca': sca,
        'abs': abs_cs,
        'ext': ext,
        'method': 'retarded'
    }


def mie_sphere_analytical(radius, eps_metal_func, wavelengths, n_max=10):
    """
    Analytical Mie solution for comparison.

    Parameters
    ----------
    radius : float
        Sphere radius in nm
    eps_metal_func : callable
        Function returning (eps, k) for given wavelength
    wavelengths : array
        Wavelengths in nm
    n_max : int
        Maximum multipole order

    Returns
    -------
    results : dict
        Contains wavelengths, ext, sca
    """
    ext = np.zeros(len(wavelengths))
    sca = np.zeros(len(wavelengths))

    for i, lam in enumerate(wavelengths):
        eps_m, _ = eps_metal_func(lam)
        n_m = np.sqrt(eps_m)

        k = 2 * np.pi / lam  # Wavenumber in vacuum
        x = k * radius       # Size parameter
        m = n_m              # Relative refractive index

        # Mie coefficients (simplified for small x)
        qext = 0
        qsca = 0

        for n in range(1, n_max + 1):
            # Spherical Bessel functions (simplified)
            # For small x, use asymptotic forms
            # This is a simplified version
            an_num = m**2 * x**2 - n*(n+1)*(1 - 1/m**2)
            an_den = m**2 * x**2 - n*(n+1)*(1 - 1/m**2) + 1j * (2*n+1) * x / (n*(n+1))

            # Very simplified - full Mie requires Bessel function calculations
            # For accurate results, use scipy.special
            pass

        # For now, return zeros - full Mie implementation needed
        ext[i] = 0
        sca[i] = 0

    return {
        'wavelengths': wavelengths,
        'ext': ext,
        'sca': sca,
        'method': 'mie'
    }


def plot_spectrum(results, title=None, save_path=None):
    """
    Plot optical spectrum.

    Parameters
    ----------
    results : dict
        Results from calculate_spectrum_*
    title : str, optional
        Plot title
    save_path : str, optional
        Path to save figure
    """
    wavelengths = results['wavelengths']

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(wavelengths, results['ext'], 'b-', linewidth=2, label='Extinction')
    ax.plot(wavelengths, results['sca'], 'g--', linewidth=2, label='Scattering')
    ax.plot(wavelengths, results['abs'], 'r:', linewidth=2, label='Absorption')

    ax.set_xlabel('Wavelength (nm)', fontsize=12)
    ax.set_ylabel('Cross section (nm²)', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    if title:
        ax.set_title(title, fontsize=14)
    else:
        method = results.get('method', 'BEM')
        ax.set_title('Optical Spectrum ({})'.format(method), fontsize=14)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print("Saved figure to {}".format(save_path))

    plt.show()


def demo_small_sphere():
    """
    Demo: Small gold nanosphere (quasistatic valid).

    For r = 10 nm << λ, quasistatic approximation is accurate.
    """
    print("\n" + "=" * 70)
    print("DEMO: Small Gold Nanosphere (r = 10 nm)")
    print("Quasistatic approximation is valid for r << λ")
    print("=" * 70)

    wavelengths = np.linspace(400, 800, 41)
    results = calculate_spectrum_quasistatic(
        radius=10.0,
        n_faces=144,
        wavelengths=wavelengths
    )

    # Find plasmon resonance
    idx_max = np.argmax(results['ext'])
    print("\nPlasmon resonance wavelength: {:.0f} nm".format(wavelengths[idx_max]))
    print("Peak extinction: {:.2f} nm^2".format(results['ext'][idx_max]))

    # Energy conservation check
    error = np.abs(results['ext'] - results['sca'] - results['abs']).max()
    print("Energy conservation error: {:.2e}".format(error))

    return results


def demo_large_sphere():
    """
    Demo: Large gold nanosphere (retardation effects).

    For r = 50 nm, retardation effects become important.
    """
    print("\n" + "=" * 70)
    print("DEMO: Large Gold Nanosphere (r = 50 nm)")
    print("Retardation effects are important for larger particles")
    print("=" * 70)

    wavelengths = np.linspace(400, 800, 21)  # Fewer points (slower)
    results = calculate_spectrum_retarded(
        radius=50.0,
        n_faces=256,  # More faces for larger particle
        wavelengths=wavelengths
    )

    # Find plasmon resonance
    idx_max = np.argmax(results['ext'])
    print("\nPlasmon resonance wavelength: {:.0f} nm".format(wavelengths[idx_max]))
    print("Peak extinction: {:.2f} nm^2".format(results['ext'][idx_max]))

    return results


def demo_size_dependence():
    """
    Demo: Size-dependent plasmon shift.

    Shows how plasmon resonance shifts with particle size.
    """
    print("\n" + "=" * 70)
    print("DEMO: Size-Dependent Plasmon Resonance")
    print("=" * 70)

    radii = [5, 10, 20, 30]
    wavelengths = np.linspace(400, 700, 31)

    results_list = []
    resonance_wavelengths = []

    for r in radii:
        print("\nRadius = {} nm".format(r))
        res = calculate_spectrum_quasistatic(
            radius=r,
            n_faces=144,
            wavelengths=wavelengths
        )
        results_list.append(res)

        idx_max = np.argmax(res['ext'])
        resonance_wavelengths.append(wavelengths[idx_max])

    print("\n" + "-" * 40)
    print("Size-dependent plasmon resonance:")
    for r, lam_res in zip(radii, resonance_wavelengths):
        print("  r = {:2d} nm: lambda_res = {:.0f} nm".format(r, lam_res))

    return results_list, radii


def main():
    """Main demo function."""
    print("=" * 70)
    print("MNPBEM Python - Gold Nanosphere Optical Spectrum Demo")
    print("=" * 70)

    # Run small sphere demo
    results_small = demo_small_sphere()

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
This demo calculated the optical spectrum of gold nanospheres using
the Boundary Element Method (BEM).

Key features demonstrated:
1. Quasistatic approximation (BEMStat) for small particles
2. Extinction, scattering, and absorption cross sections
3. Plasmon resonance identification

For larger particles (r > 30 nm), use the retarded solver (BEMRet)
which includes radiation damping and retardation effects.

For particles on a substrate (e.g., glass), layer structure support
is needed (not yet implemented in Python version).

MATLAB equivalent functions:
- trisphere(n, r) -> trisphere(n, r)
- comparticle(eps, particles, inout) -> ComParticle(eps, particles, inout)
- bemstat(p) -> BEMStat(p)
- planewavestat(pol) -> PlaneWaveStat(pol)
- exc = potential(planewave, p, enei) -> exc = planewave.potential(p, enei)
- sig = bem \\ exc -> sig = bem.solve(exc)
- sca = scattering(planewave, sig) -> sca = planewave.scattering(sig)
""")

    return results_small


if __name__ == '__main__':
    results = main()
