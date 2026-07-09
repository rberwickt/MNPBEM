"""
13_shapes: BEMStat extinction spectrum for 7 particle shapes (Python)

All shapes: Au nanoparticle in vacuum, BEMStat + PlaneWaveStat([1,0,0])
ext vs 400-800nm (41pt), timing for each shape.

Shapes:
  1. trisphere(144, 20.0)
  2. trirod(10.0, 40.0, n=[15,15,15])  -- x-pol + z-pol
  3. tricube(10, 20.0)
  4. tritorus(15.0, 5.0, n=[20,20])
  5. trispheresegment(phi, theta, 20.0)  -- hemisphere
  6. trispherescale(trisphere(144,20), [1,1,2])  -- ellipsoid
  7. tripolygon(hexagon, edge)  -- hexagonal prism
"""

import sys
import os
import time

import numpy as np

sys.path.insert(0, '/home/yoojk20/workspace/MNPBEM')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from mnpbem import (
    EpsConst, EpsTable,
    trisphere, trirod, tricube, tritorus,
    trispheresegment, trispherescale, tripolygon,
    Polygon, EdgeProfile,
    ComParticle, BEMStat, PlaneWaveStat,
)


DATA_DIR = '/home/yoojk20/workspace/MNPBEM/validation/13_shapes/data'
FIG_DIR = '/home/yoojk20/workspace/MNPBEM/validation/13_shapes/figures'
ENEI = np.linspace(400, 800, 41)


def run_bemstat(p_shape, label, pols=None):
    """Run BEMStat extinction calculation for a shape.

    Parameters
    ----------
    p_shape : Particle
        The nanoparticle mesh.
    label : str
        Name for logging.
    pols : list of ndarray, optional
        Polarization vectors.  Default is [[1,0,0]].

    Returns
    -------
    ext_dict : dict
        Keys are polarization labels, values are extinction arrays.
    elapsed : float
        Total wall-clock seconds.
    """
    eps_tab = [EpsConst(1.0), EpsTable('gold.dat')]
    p = ComParticle(eps_tab, [p_shape], [[2, 1]], 1, interp='curv')

    if pols is None:
        pols = [np.array([1.0, 0.0, 0.0])]

    ext_dict = {}
    t_total = 0.0

    for ip, pol in enumerate(pols):
        exc = PlaneWaveStat(pol)
        bem = BEMStat(p)
        ext = np.zeros(len(ENEI))

        t0 = time.perf_counter()
        for i, wl in enumerate(ENEI):
            pot = exc.potential(p, wl)
            sig, _ = bem.solve(pot)
            ext[i] = exc.extinction(sig)
        dt = time.perf_counter() - t0

        pol_label = 'xpol' if np.allclose(pol, [1, 0, 0]) else 'zpol'
        ext_dict[pol_label] = ext
        t_total += dt
        print('  [{}] pol={}: {:.4f} s'.format(label, pol_label, dt))

    return ext_dict, t_total


def save_csv(filename, enei, ext_dict):
    """Save extinction data to CSV."""
    cols = list(ext_dict.keys())
    header = 'wavelength_nm,' + ','.join('extinction_{}'.format(c) if len(cols) > 1 else 'extinction' for c in cols)
    if len(cols) == 1:
        header = 'wavelength_nm,extinction'
    data = np.column_stack([enei] + [ext_dict[c] for c in cols])
    np.savetxt(filename, data, delimiter=',', header=header, comments='', fmt='%.15e')


# =========================================================================
# Shape 1: trisphere
# =========================================================================
def shape_trisphere():
    print('=== 1. trisphere(144, 20.0) ===')
    p_shape = trisphere(144, 20.0)
    print('  nfaces: {}'.format(p_shape.nfaces))
    ext_dict, dt = run_bemstat(p_shape, 'trisphere')
    save_csv(os.path.join(DATA_DIR, 'trisphere_python.csv'), ENEI, ext_dict)
    return ext_dict, dt


# =========================================================================
# Shape 2: trirod (x-pol + z-pol)
# =========================================================================
def shape_trirod():
    print('=== 2. trirod(10.0, 40.0, n=[15,15,15]) ===')
    p_shape = trirod(10.0, 40.0, n=[15, 15, 15])
    print('  nfaces: {}'.format(p_shape.nfaces))
    pols = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    ]
    ext_dict, dt = run_bemstat(p_shape, 'trirod', pols=pols)
    # Save with two columns
    header = 'wavelength_nm,extinction_xpol,extinction_zpol'
    data = np.column_stack([ENEI, ext_dict['xpol'], ext_dict['zpol']])
    np.savetxt(os.path.join(DATA_DIR, 'trirod_python.csv'), data,
               delimiter=',', header=header, comments='', fmt='%.15e')
    return ext_dict, dt


# =========================================================================
# Shape 3: tricube
# =========================================================================
def shape_tricube():
    print('=== 3. tricube(10, 20.0) ===')
    p_shape = tricube(10, 20.0)
    print('  nfaces: {}'.format(p_shape.nfaces))
    ext_dict, dt = run_bemstat(p_shape, 'tricube')
    save_csv(os.path.join(DATA_DIR, 'tricube_python.csv'), ENEI, ext_dict)
    return ext_dict, dt


# =========================================================================
# Shape 4: tritorus
# =========================================================================
def shape_tritorus():
    print('=== 4. tritorus(15.0, 5.0, n=[20,20]) ===')
    p_shape = tritorus(15.0, 5.0, n=[20, 20])
    print('  nfaces: {}'.format(p_shape.nfaces))
    ext_dict, dt = run_bemstat(p_shape, 'tritorus')
    save_csv(os.path.join(DATA_DIR, 'tritorus_python.csv'), ENEI, ext_dict)
    return ext_dict, dt


# =========================================================================
# Shape 5: trispheresegment (hemisphere)
# =========================================================================
def shape_trispheresegment():
    print('=== 5. trispheresegment (hemisphere, d=20) ===')
    phi = np.linspace(0, 2 * np.pi, 15)
    theta = np.linspace(0, np.pi / 2, 10)
    p_shape = trispheresegment(phi, theta, 20.0)
    print('  nfaces: {}'.format(p_shape.nfaces))
    ext_dict, dt = run_bemstat(p_shape, 'trispheresegment')
    save_csv(os.path.join(DATA_DIR, 'trispheresegment_python.csv'), ENEI, ext_dict)
    return ext_dict, dt


# =========================================================================
# Shape 6: trispherescale (ellipsoid)
# =========================================================================
def shape_trispherescale():
    print('=== 6. scale(trisphere(144,20), [1,1,2]) ===')
    p_shape = trisphere(144, 20.0)
    # Use Particle.scale() which handles per-axis scaling (like MATLAB scale())
    p_shape.scale([1.0, 1.0, 2.0])
    print('  nfaces: {}'.format(p_shape.nfaces))
    ext_dict, dt = run_bemstat(p_shape, 'trispherescale')
    save_csv(os.path.join(DATA_DIR, 'trispherescale_python.csv'), ENEI, ext_dict)
    return ext_dict, dt


# =========================================================================
# Shape 7: tripolygon (hexagonal prism)
# =========================================================================
def shape_tripolygon():
    print('=== 7. tripolygon (hexagon + EdgeProfile) ===')
    poly = Polygon(6, size=[20, 20])
    edge = EdgeProfile(5.0, 11)
    p_shape = tripolygon(poly, edge)
    print('  nfaces: {}'.format(p_shape.nfaces))
    ext_dict, dt = run_bemstat(p_shape, 'tripolygon')
    save_csv(os.path.join(DATA_DIR, 'tripolygon_python.csv'), ENEI, ext_dict)
    return ext_dict, dt


# =========================================================================
# Main
# =========================================================================
def main():
    results = {}
    timing = {}

    for name, func in [
        ('trisphere',        shape_trisphere),
        ('trirod',           shape_trirod),
        ('tricube',          shape_tricube),
        ('tritorus',         shape_tritorus),
        ('trispheresegment', shape_trispheresegment),
        ('trispherescale',   shape_trispherescale),
        ('tripolygon',       shape_tripolygon),
    ]:
        ext_dict, dt = func()
        results[name] = ext_dict
        timing[name] = dt

    # Save timing
    with open(os.path.join(DATA_DIR, 'python_timing.csv'), 'w') as f:
        f.write('test,time_seconds\n')
        for name in timing:
            f.write('{},{:.6f}\n'.format(name, timing[name]))

    # Plot individual Python figures
    for name, ext_dict in results.items():
        fig, ax = plt.subplots(figsize=(8, 5))
        if 'zpol' in ext_dict:
            ax.plot(ENEI, ext_dict['xpol'], 'b-', lw=1.5, label='x-pol')
            ax.plot(ENEI, ext_dict['zpol'], 'r--', lw=1.5, label='z-pol')
            ax.legend(loc='best')
        else:
            key = list(ext_dict.keys())[0]
            ax.plot(ENEI, ext_dict[key], 'b-', lw=1.5)
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('Extinction (nm$^2$)')
        ax.set_title('Python BEMStat -- {}'.format(name))
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, '{}_python.png'.format(name)), dpi=150)
        plt.close(fig)
        print('[saved] {}_python.png'.format(name))

    # Print timing summary
    print('\n=== Timing Summary ===')
    total = 0.0
    for name, dt in timing.items():
        print('  {:<20s} : {:.4f} s'.format(name, dt))
        total += dt
    print('  {:<20s} : {:.4f} s'.format('TOTAL', total))

    print('\n[info] Python 13_shapes validation complete.')


if __name__ == '__main__':
    main()
