"""5000-face benchmark suite for BEMStat / BEMRet.

Measures assembly_seconds, lu_factor_seconds, solve_seconds, total_seconds
for representative large-mesh problems.

Note: trisphere() in this port is capped at 1444 vertices (2884 faces) due to
the available pre-computed sphere data files (see mnpbem/data/sphere*.bin).
For benchmark sizes larger than 2884 faces, we fall back to the 1444-vertex
sphere — see CSV `n_faces_actual` column.

Usage
-----
python validation/perf_5000face.py [--out PATH]

Output CSV columns
------------------
case, n_faces_requested, n_faces_actual, assembly_seconds,
lu_factor_seconds, solve_seconds, total_seconds, status
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, '/home/yoojk20/workspace/MNPBEM')

from mnpbem.materials import EpsConst, EpsDrude
from mnpbem.geometry import ComParticle
from mnpbem.geometry.mesh_generators import trisphere
from mnpbem.bem import BEMStat, BEMRet
from mnpbem.excitation import PlaneWaveStat, PlaneWaveRet


def _drude_gold() -> EpsDrude:
    """Same Drude gold parameters as demo_nanosphere_spectrum."""
    eps_inf = 9.5
    lambda_p = 138.0
    gamma = 0.069
    lambda_gamma = 1240.0 / gamma
    return EpsDrude(eps_inf, lambda_p, lambda_gamma)


def _build_particle(n_faces_requested: int, radius_nm: float = 50.0):
    sphere = trisphere(n_faces_requested, radius_nm * 2.0)  # diameter
    eps_tab = [EpsConst(1.0), _drude_gold()]
    p = ComParticle(eps_tab, [sphere], [[2, 1]], [1])
    return p, sphere


def benchmark_bemstat(n_faces_requested: int, enei: float = 600.0) -> dict:
    """Build BEMStat, run one solve at one wavelength."""
    p, sphere = _build_particle(n_faces_requested)
    n_faces_actual = sphere.faces.shape[0]

    # ---- Assembly: building Green function + F matrix ----
    t0 = time.perf_counter()
    bem = BEMStat(p)
    t_assembly = time.perf_counter() - t0

    # ---- LU factor: _init_matrices on first wavelength ----
    t0 = time.perf_counter()
    bem._init_matrices(enei)
    t_lu = time.perf_counter() - t0

    # ---- Solve: one PlaneWave excitation -> sigma ----
    pol = np.array([[1.0, 0.0, 0.0]])
    direction = np.array([[0.0, 0.0, 1.0]])
    exc = PlaneWaveStat(pol, direction)
    excs = exc(p, enei)
    t0 = time.perf_counter()
    sig, _ = bem.solve(excs)
    t_solve = time.perf_counter() - t0

    return {
        'n_faces_actual': n_faces_actual,
        'assembly_seconds': t_assembly,
        'lu_factor_seconds': t_lu,
        'solve_seconds': t_solve,
        'total_seconds': t_assembly + t_lu + t_solve,
    }


def benchmark_bemret(n_faces_requested: int, enei: float = 600.0) -> dict:
    """Build BEMRet, run one solve at one wavelength."""
    p, sphere = _build_particle(n_faces_requested)
    n_faces_actual = sphere.faces.shape[0]

    t0 = time.perf_counter()
    bem = BEMRet(p)
    t_assembly = time.perf_counter() - t0

    t0 = time.perf_counter()
    bem.init(enei)
    t_lu = time.perf_counter() - t0

    pol = np.array([[1.0, 0.0, 0.0]])
    direction = np.array([[0.0, 0.0, 1.0]])
    exc = PlaneWaveRet(pol, direction)
    excs = exc(p, enei)
    t0 = time.perf_counter()
    sig, _ = bem.solve(excs)
    t_solve = time.perf_counter() - t0

    return {
        'n_faces_actual': n_faces_actual,
        'assembly_seconds': t_assembly,
        'lu_factor_seconds': t_lu,
        'solve_seconds': t_solve,
        'total_seconds': t_assembly + t_lu + t_solve,
    }


CASES = [
    # (case_name, kind, n_faces_requested)
    # NOTE: trisphere() in this Python port maxes out at 1444 vertices →
    # 2884 faces (after midpoint subdivision). Larger requests round down.
    # See mnpbem/data/sphere*.bin and mesh_generators.trisphere().
    ('trisphere2562_BEMStat',  'stat', 2562),   # → 2884 faces
    ('trisphere2562_BEMRet',   'ret',  2562),   # → 2884 faces (~25 min)
    ('trisphere10242_BEMStat', 'stat', 10242),  # → 2884 faces (capped)
    # Smaller BEMRet variant for quicker turnaround
    ('trisphere1024_BEMRet',   'ret',  1024),   # → 2044 faces
]


def run_all(out_path: str) -> None:
    rows = []
    for name, kind, n_req in CASES:
        print(f'==> {name} (n_faces_requested={n_req}, kind={kind})', flush=True)
        try:
            if kind == 'stat':
                res = benchmark_bemstat(n_req)
            else:
                res = benchmark_bemret(n_req)
            status = 'OK'
        except Exception:
            traceback.print_exc()
            res = {
                'n_faces_actual': -1,
                'assembly_seconds': float('nan'),
                'lu_factor_seconds': float('nan'),
                'solve_seconds': float('nan'),
                'total_seconds': float('nan'),
            }
            status = 'FAIL'
        row = {
            'case': name,
            'n_faces_requested': n_req,
            'n_faces_actual': res['n_faces_actual'],
            'assembly_seconds': res['assembly_seconds'],
            'lu_factor_seconds': res['lu_factor_seconds'],
            'solve_seconds': res['solve_seconds'],
            'total_seconds': res['total_seconds'],
            'status': status,
        }
        rows.append(row)
        print(
            f'    n_faces_actual={row["n_faces_actual"]} '
            f'assembly={row["assembly_seconds"]:.3f}s '
            f'lu={row["lu_factor_seconds"]:.3f}s '
            f'solve={row["solve_seconds"]:.3f}s '
            f'total={row["total_seconds"]:.3f}s [{status}]',
            flush=True,
        )

    fieldnames = [
        'case', 'n_faces_requested', 'n_faces_actual',
        'assembly_seconds', 'lu_factor_seconds', 'solve_seconds',
        'total_seconds', 'status',
    ]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f'wrote {out_path}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--out',
        default='/home/yoojk20/workspace/MNPBEM/validation/perf_5000face_2026-04-27.csv',
    )
    args = parser.parse_args()
    run_all(args.out)
