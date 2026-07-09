"""Quick GPU vs CPU timing for trisphere BEMStat at multiple resolutions.

Run as a script:
    python tests/perf_gpu_lu.py
"""

import os
import time

import numpy as np


def _bench_lu(N, n_solves=2):
    import mnpbem.utils.gpu as gmod

    rng = np.random.default_rng(42)
    A = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
    b = rng.standard_normal((N, 3)) + 1j * rng.standard_normal((N, 3))

    results = {}
    for mode, use_gpu in (("cpu", False), ("gpu", True)):
        gmod.USE_GPU = use_gpu
        # warm-up
        piv = gmod.lu_factor_dispatch(A.copy())
        _ = gmod.lu_solve_dispatch(piv, b.copy())
        # timing
        t0 = time.perf_counter()
        piv = gmod.lu_factor_dispatch(A.copy())
        t_factor = time.perf_counter() - t0
        t0 = time.perf_counter()
        for _ in range(n_solves):
            _ = gmod.lu_solve_dispatch(piv, b.copy())
        t_solve = (time.perf_counter() - t0) / n_solves
        results[mode] = (t_factor, t_solve, piv[0])
    return results


if __name__ == "__main__":
    for N in (1024, 2562, 5120, 10242):
        try:
            r = _bench_lu(N)
        except Exception as e:
            print(f"N={N}: error {e}")
            continue
        cpu_f, cpu_s, _ = r["cpu"]
        gpu_f, gpu_s, gpu_tag = r["gpu"]
        print(f"N={N:6d}  CPU LU={cpu_f:7.3f}s solve={cpu_s*1000:7.1f}ms  "
              f"GPU({gpu_tag}) LU={gpu_f:7.3f}s solve={gpu_s*1000:7.1f}ms  "
              f"speedup_LU={cpu_f / max(gpu_f, 1e-9):5.2f}x")
