# MNPBEM Python Port — Architecture

This document is aimed at contributors and maintainers (including future-self).
It describes how the Python port is laid out, *why* the major design choices
were made, and where to look when something behaves differently from the
MATLAB MNPBEM toolbox.

External users should start with `README.md` and `docs/API_REFERENCE.md`
instead.

## 1. High-level overview

The Python port is a faithful, MATLAB-compatible re-implementation of
Hohenester & Trügler's MNPBEM toolbox. The goals, in priority order, are:

1. **Numerical parity with MATLAB** on the official demo set (50 + 22 demos)
   — bit-identical where the underlying math libraries permit, and within
   floating-point ULP tolerance otherwise.
2. **Pure Python distribution** — no MATLAB runtime required at import or
   call time. MATLAB Engine is only used as an *optional* validation backend
   (opt-in via env var) for cross-checking specific demos.
3. **Performance parity or better** — CPU-only path keeps up with MATLAB on
   small/medium meshes; CuPy GPU path scales beyond what MATLAB can do.
4. **A drop-in API** — class names, method names, and option keywords
   mirror the MATLAB OOP layout so MATLAB users can port scripts mechanically.

The core stack is:

- Python 3.11 / 3.12
- `numpy` + `scipy` (LU, GMRES, Bessel functions, ODE integrators)
- `numba` (JIT for the dense Green-function and meshfield hot loops)
- `cupy` (optional, single- and multi-GPU dense LU / matmul / refinement)
- `mpi4py` (optional, multi-node wavelength dispatch)
- `fmm3dpy` (optional, free-space retarded meshfield acceleration)

## 2. Repository layout

```
MNPBEM/
├── mnpbem/                     # the importable package
│   ├── __init__.py             # re-exports the public API surface
│   ├── geometry/               # particles, polygons, mesh2d, layer structure
│   ├── materials/              # dielectric functions
│   ├── greenfun/               # Green's functions, ACA, H-matrix
│   ├── bem/                    # BEM solvers (direct + iterative)
│   ├── simulation/             # excitations + meshfield evaluation
│   ├── spectrum/               # spectrum / cross-section helpers
│   ├── mie/                    # Mie reference solver
│   ├── misc/                   # math, plotting, options, units
│   └── utils/                  # GPU dispatch, multi-GPU, MPI, parallel
├── docs/                       # ARCHITECTURE / API_REFERENCE / PERF / ...
├── tests/                      # unit + regression tests
├── validation/                 # Mie / sphere / rod / dimer / shapes
└── 72demos_validation/         # MATLAB-vs-Python demo harness
```

### 2.1 `mnpbem.geometry`

| File                     | Role                                                          |
|--------------------------|---------------------------------------------------------------|
| `particle.py`            | `Particle` base class (`verts`, `faces`, `nvec`, `area`, ...) |
| `comparticle.py`         | Multi-particle composite; tracks `inout`, `closed`, `eps`     |
| `comparticle_mirror.py`  | Mirror-symmetric variant; whitelisted `sym ∈ {x,y,z,xy,...}`  |
| `compoint.py`            | `Point` / `ComPoint` — observation-point bag with `inout`     |
| `polygon.py`             | 2D polygon — boolean ops, normalization, plotting             |
| `polygon3.py`            | Lift a 2D polygon into 3D; `plate`, `vribbon`, etc.           |
| `edgeprofile.py`         | Edge profile generator for rounded prism edges                |
| `mesh_generators.py`     | `trisphere`, `trirod`, `tricube`, `tritorus`, `trispheresegment`, `tripolygon`, `fvgrid` |
| `mesh2d.py`              | 2D Delaunay mesher — line-by-line port of MATLAB `mesh2d`     |
| `mesh2d_core.py` / `mesh2d_utils.py` | helpers split out for reuse                       |
| `shape_functions.py`     | Linear/curv face shape functions and quadrature mappings      |
| `connect.py`             | Particle-particle connectivity & edge stitching               |
| `compound.py`            | `@compound` MATLAB OOP class (10 public methods)              |
| `layer_structure.py`     | Stratified layer system + Sommerfeld integrators              |

### 2.2 `mnpbem.materials`

`EpsConst`, `EpsTable` (loads MATLAB `.mat` data files in `materials/data/`),
`EpsDrude`, `EpsFun`, plus the `epsfun(...)` factory.

### 2.3 `mnpbem.greenfun`

Quasistatic and retarded Green functions for free space, mirror symmetry,
and stratified layer systems.

| File                          | Role                                                |
|-------------------------------|-----------------------------------------------------|
| `compgreen_stat.py`           | Quasistatic Green function (G, F, Gp)               |
| `compgreen_ret.py`            | Retarded Green function (G, F, H1/H2 + Cartesian)   |
| `compgreen_stat_mirror.py` / `compgreen_ret_mirror.py` | Mirror-symmetric extensions |
| `compgreen_stat_layer.py` / `compgreen_ret_layer.py`   | Layered-medium extensions  |
| `greenstat.py` / `greenret_layer.py` / `greenret_refined.py` | Lower-level kernel + diagonal/off-diagonal refinement |
| `greentab_layer.py` / `compgreentab_layer.py` | Tabulated layered Green functions; bilinear/trilinear interp |
| `coverlayer.py`               | `+coverlayer` — refinement on layer-interface particles |
| `clustertree.py` / `hmatrix.py` | Cluster tree + hierarchical low-rank matrix      |
| `aca_compgreen_stat.py` / `aca_compgreen_ret.py` / `aca_compgreen_ret_layer.py` | ACA-accelerated Green functions |
| `aca_gpu.py` / `h_matrix_gpu.py` | GPU prototypes of ACA and H-matrix              |
| `_numba_kernels.py` / `_numba_ret_kernels.py` / `_numba_layer.py` | Numba JIT kernels for dense fill / interpolation |

### 2.4 `mnpbem.bem`

| File                   | Role                                               |
|------------------------|----------------------------------------------------|
| `bembase.py`           | Abstract `BemBase` + 5 factory functions           |
| `bem_stat.py`          | Direct quasistatic BEM solver                      |
| `bem_ret.py`           | Direct retarded BEM solver (2x2 block form)        |
| `bem_stat_layer.py`    | Quasistatic BEM with layer Green functions         |
| `bem_ret_layer.py`     | Retarded BEM with layer Green functions            |
| `bem_stat_mirror.py` / `bem_ret_mirror.py` | Mirror-symmetric direct solvers |
| `bem_iter.py`          | Common GMRES/iterative solver scaffolding          |
| `bem_stat_iter.py` / `bem_ret_iter.py` / `bem_ret_layer_iter.py` | Iterative + ACA H-matrix variants |
| `bem_stat_eig.py` / `bem_stat_eig_mirror.py` | Eigenmode-based quasistatic solver |
| `bem_layer_mirror.py`  | Mirror x layer combination                         |
| `plasmonmode.py`       | Plasmon eigenmode extraction (left/right pairing)  |
| `solver_factory.py`    | `bem.solver(...)` dispatcher                       |
| `matlab_bem.py`        | MATLAB Engine adapter (opt-in validation backend)  |

### 2.5 `mnpbem.simulation`

| File                           | Role                                              |
|--------------------------------|---------------------------------------------------|
| `planewave_*.py`               | Stat / Ret / Mirror / Layer plane-wave excitation |
| `dipole_*.py`                  | Stat / Ret / Mirror / Layer dipole excitation     |
| `eels_base.py` / `eels_stat.py` / `eels_ret.py` | EELS excitation + loss      |
| `meshfield.py` / `_meshfield_numba.py` / `meshfield_fmm.py` | Near-field evaluator on grids; Numba + optional FMM3D |
| `electronbeam_factory.py` / `dipole_factory.py` / `planewave_factory.py` | Construction helpers |
| `retarded_utils.py`            | shared helpers for the ret excitation classes     |

### 2.6 `mnpbem.spectrum`

`SpectrumStat`, `SpectrumRet`, `SpectrumStatLayer`, `SpectrumRetLayer`, plus
the `spectrum(...)` factory. The retarded variants embed the MATLAB
`pinfty256.bin` reference far-field for bit-identical extinction/scattering.

### 2.7 `mnpbem.mie`

Reference Mie solver (`MieStat`, `MieRet`, `MieGans`) used by the validation
suite to cross-check spherical-particle results without going through BEM.

### 2.8 `mnpbem.misc`

Math primitives (`matmul`, `vec_norm`, `pdist2`, `bdist2`, ...), Gauss-Legendre
nodes (`lglnodes`, `lgwt`), `QuadFace` polar quadrature, options
(`bemoptions`, `getbemoptions`), units (`EV2NM`, `BOHR`, `HARTREE`, `FINE`),
`BemPlot`/`coneplot` plotting, `IGrid2`/`IGrid3` interpolators,
`ValArray`/`VecArray` containers.

### 2.9 `mnpbem.utils`

| File              | Role                                                        |
|-------------------|-------------------------------------------------------------|
| `matlab_compat.py`| Bit-identical MATLAB primitives (`matan2`, `mexp`, `msqrt`, `mlinspace`, `m_unique`, ...) |
| `gpu.py`          | `lu_factor_dispatch` / `lu_solve_dispatch` / `solve` / `eigh` / `matmul` over CPU and CuPy |
| `multi_gpu.py`    | Wavelength-batched dispatch across local GPUs (subprocess-per-GPU) |
| `mpi_dispatch.py` | Multi-node wavelength dispatch on top of `multi_gpu` (mpi4py) |
| `parallel.py`     | `compute_spectrum`, `compute_spectrum_parallel` (CPU)       |
| `quadface.py` / `quadrature.py` | shared quadrature helpers                     |
| `matlab_ode45.py` | 1:1 re-implementation of MATLAB `ode45` step controller     |
| `constants.py`    | `EV2NM` and friends                                         |

## 3. Key design decisions

### 3.1 `matlab_compat` — why it exists

`np.linspace`, `np.arctan2`, `np.exp`, `np.log`, `np.sqrt` and friends each
differ from MATLAB by up to 1 ULP because of different accumulation order or
fdlibm vs. MKL implementations. A 1 ULP drift in `theta` or `linspace`
propagates through `p @ rot` and `np.unique`, eventually changing the mesh
**topology** (different vertex ordering produces different face adjacency,
which changes BEM matrix sparsity patterns). To eliminate this drift the
port loads MATLAB's own `libmwmathutil.so` via `ctypes` when available and
calls its fdlibm-derived primitives directly. When MATLAB is not installed
the wrappers fall back to numpy.

This is the single most important design choice for parity with MATLAB:
without it most demos regress to "warn" or "BAD" because of mesh-induced
divergence rather than algorithm bugs.

See: `mnpbem/utils/matlab_compat.py`, `docs/MESH2D_FP_LIMIT.md`.

### 3.2 Numba JIT (default ON)

- Activation: env var `MNPBEM_NUMBA` defaults to `1`; set `0` to disable.
- Used for: dense Green-function fill (`compgreen_stat`, `compgreen_ret`,
  per-particle distance kernels), bilinear/trilinear interpolation in
  `greentab_layer`, ACA inner loops, and the `meshfield` per-wavelength
  dense-Green evaluator.
- Kernels are decorated `@njit(cache=True)`. `fastmath` is **off** because
  we observed it breaks IEEE-754 sign-of-zero handling, which in turn
  changes results in off-diagonal panels of `compgreen_ret`.
- This is the M4 N1-N6 work; without it the CPU path lags MATLAB on dense
  meshes.

### 3.3 GPU dispatch (opt-in)

- Activation: env var `MNPBEM_GPU=1` (default OFF — explicit opt-in to avoid
  surprising users without CUDA).
- Threshold: `MNPBEM_GPU_THRESHOLD` (default 1500). Below the threshold,
  scipy CPU LU is faster than the host↔device round trip.
- Layer-Green specialization: `MNPBEM_GPU_LAYER` / `MNPBEM_GPU_LAYER_THRESHOLD`
  control the GEMM fast path inside the layer Sommerfeld batcher.
- Native cupy outputs: `MNPBEM_GPU_NATIVE=1` keeps refined Green tensors on
  the device end-to-end through the BEMRet pipeline (`Sigma1 = H @ G^-1`
  is solved with cuSolver `lu_solve` directly).
- All GPU calls return numpy arrays unless the caller is already cupy-aware,
  so the public API does not change when the env var is set.
- See `mnpbem/utils/gpu.py` for the dispatcher and `bem_*_iter.py`,
  `bem_*_mirror.py`, `bem_stat_eig*.py` for the consumers.

### 3.4 Multi-GPU and multi-node

Wavelength sweeps are embarrassingly parallel for the BEMRet solve: each
λ builds and solves an independent system. This is the only axis the port
parallelises across processes:

- `mnpbem.utils.multi_gpu.solve_spectrum_multi_gpu` — splits λ across local
  GPUs, one subprocess per CUDA device, `CUDA_VISIBLE_DEVICES` pinning,
  results merged through a `multiprocessing.Queue`.
- `mnpbem.utils.mpi_dispatch.solve_spectrum_mpi` — adds an MPI rank axis on
  top: each rank gets a wavelength slice, then internally calls
  `solve_spectrum_multi_gpu`. Falls back to a serial CPU loop if the rank
  has no GPU and to `solve_spectrum_multi_gpu` if `mpi4py` is missing or
  the world has size 1.

### 3.5 ACA (Adaptive Cross Approximation) and H-matrix

Far-field BEM blocks are compressed with rank-revealing ACA on a binary
cluster tree. Defaults match MATLAB:

- `htol=1e-6` (Frobenius approximation tolerance)
- `kmax=[4, 100]` (rank bounds)
- `cleaf=200` (leaf cluster size)
- `ACATOL=1e-10` (cross-pivot abort)
- `fadmiss` is k-aware for retarded problems; for layered Green functions
  it accepts a per-λ override (`make_kaware_fadmiss`).

Consumers: `BEMStatIter`, `BEMRetIter`, `BEMRetLayerIter`. Direct solvers
keep the dense path for now; the iterative + ACA path is the recommended
route for meshes above ~5000 faces.

The complex128 ACA inner loops are Numba-JITted (`hmatrix.py`) and
`fadmiss` admissibility takes the wavelength k into account
(`make_kaware_fadmiss`). An experimental GPU port lives in
`aca_gpu.py` / `h_matrix_gpu.py` (see `docs/H_MATRIX_GPU.md`).

### 3.6 Mesh quadrature

Surface integrals use the polar-quadrature scheme of MATLAB MNPBEM
(`QuadFace` + diagonal refinement via `refinematrix` / `refinematrixlayer`).
Triangles and quads are refined separately; refinement points are picked
by MATLAB-compatible `pdist2` (we re-implemented the algorithm because
`scipy.spatial.distance` reorders pairs of equal distance differently and
that reorder propagates through the mesh).

### 3.7 Sommerfeld integrator

The layer-Green tabulation needs the radial Sommerfeld integral over the
complex k-plane. Two backends:

- Default: composite Gauss-Legendre on adaptive panels with batched RHS
  (vectorised across query points), faster than MATLAB `ode113` and
  trivially differentiable.
- ODE backend (opt-in via option): `scipy.solve_ivp` with a Numba-compiled
  RHS, plus a custom `matlab_ode45.py` step controller for cases where
  exact MATLAB step pattern is required.

### 3.8 Direct retarded solver — block matrix form

`BEMRet` rebuilds the MATLAB `initmat.m` 2x2 structured block system rather
than the older single-monolith form. This was required to match MATLAB on
multi-component particles and on layer-coupled cases, and it lets the
matrix assembly stage be eagerly pushed to GPU (Lane A2 in M4).

### 3.9 Why we deviate from MATLAB (intentionally)

| Item | Difference | Reason |
|---|---|---|
| FP roundoff | up to ~ULP in some demos | MATLAB libmwmathutil vs. Python math libs; minimised by `matlab_compat` but not always eliminated (see `MESH2D_FP_LIMIT.md`) |
| ACA tie-break | Slightly different pivot order in degenerate panels | No measurable impact on Green-function accuracy; left as numpy default for clarity |
| Sommerfeld integration | Default is GL panels, not `ode113` | 5-10× faster, same accuracy at default tolerance |
| Direct BEMRet | 2x2 block reformulation | Required for multi-particle layer cases; equivalent to MATLAB |
| `MNPBEM_GPU` default OFF | Opt-in instead of opt-out | Avoids initialising CUDA when the user did not ask for it |
| `pinfty` default | MATLAB `pinfty256.bin` for ret spectra | MATLAB ships a fixed reference far-field; using anything else introduces a 0.5-1% bias on extinction |
| `EV2NM` constant | `1/8.0655477e-4` | Match MATLAB `Misc/units.m` exactly (Wave 28 D) |

### 3.10 Why we *do not* deviate

The single biggest pull on the port has been resisting the temptation to
"fix" MATLAB quirks. Several BAD demos turned out to be Python being more
correct than MATLAB; we matched MATLAB anyway, because parity is the
acceptance criterion. Examples:

- `_minrectangle` tie-break uses MATLAB's strict `<` comparison instead of
  Python's `<=`, even though either is mathematically valid — so that
  regular-N-gon meshes orient identically.
- `dipoleretlayer` keeps MATLAB's `pinfty` bug for compatibility on
  `demodipret*` (Wave 22 B).
- `intbessel` / `inthankel` use MATLAB's specific multiplication order so
  the layer Sommerfeld values agree to ULP (Wave 49).

If you find a MATLAB behaviour that looks wrong, please flag it before
"fixing" it — the test suite will likely regress.

### 3.11 Schur complement for cover-layer BEM (v1.2.0)

`EpsNonlocal` cover-layer formulation 은 mesh face count 를 약 2× 증가시켜
BEM 행렬 메모리를 약 4× 폭증하게 만든다 (n × n 행렬에서 n 가 2배 → 메모리
4배). Schur complement 으로 shell 변수를 LU 풀이 전 소거하면, 코어 변수만
풀어 reduced matrix 의 메모리는 ~2× 만 사용 + LU 풀이 시간은 약 30% 감소.

블록 행렬 표기로

```
[ A_cc  A_cs ] [ x_c ]   [ b_c ]
[ A_sc  A_ss ] [ x_s ] = [ b_s ]
```

이고, shell 블록 (s) 을 소거하면

```
S = A_cc - A_cs * A_ss^-1 * A_sc       # Schur complement
S * x_c = b_c - A_cs * A_ss^-1 * b_s
x_s = A_ss^-1 * (b_s - A_sc * x_c)
```

만 풀면 된다. 수학적으로 standard formulation 과 동등 — 회귀 테스트에서
machine precision 일치 (rel < 1e-12).

- 활성: `BEMStat(p, schur=True)`, `BEMRet(p, schur=True)` 또는
  `schur='auto'` 로 cover layer 자동 감지.
- 구현: `mnpbem/bem/schur_helpers.py` (블록 indexing,
  `lu_factor_dispatch` 의 reduced-matrix 호출).
- BEMRetIter / BEMRetLayer 미적용 (M5+ 후속 — H-matrix preconditioner 와
  결합).

### 3.12 VRAM share — multi-GPU LU dispatch (v1.2.0)

단일 GPU VRAM (예: RTX A6000 48 GB) 을 초과하는 큰 dense LU
(25k+ face, 50+ GB) 를 multi-GPU 메모리 풀로 처리하는 경로. Lane D
(현행 multi-GPU = wavelength 분배) 와는 직교하는 새 축 — 1 worker 가
N개 GPU의 메모리를 합쳐 *하나의* 큰 계산을 처리한다.

| 모드 | worker 수 | 1 worker 의 VRAM | 사용 사례 |
|---|---|---|---|
| Lane D (v1.0+) | N | 1 GPU 분 | wavelength 분배 |
| **VRAM share (v1.2.0)** | 1 | M GPU 합산 | 큰 단일 계산 |
| 두 모드 결합 | N | M GPU 합산 | 큰 계산 × wavelength 분배 |

기본 backend 는 NVIDIA cuSOLVER MG (`cusolverMgGetrf` /
`cusolverMgGetrs`) — block-cyclic distributed matrix 로 NVLink/PCIe
전송이 NVIDIA 측에서 자동 최적화. cupy MemoryPool 위에서 동작.

- 활성:
  - 환경변수 `MNPBEM_VRAM_SHARE_GPUS=N` (default 1 = single GPU 동작).
  - `MNPBEM_VRAM_SHARE_BACKEND=cusolvermg` (default; magma / nccl 예정).
  - `mnpbem.utils.gpu.lu_factor_dispatch(A, n_gpus=N)` 직접 호출.
- 구현: `mnpbem/utils/multi_gpu_lu.py` (cuSolverMg ctypes wrapper +
  block-cyclic distributor).
- `pymnpbem_simulation` wrapper 의 `compute.n_gpus_per_worker > 1` 가
  자동으로 환경변수를 설정.
- 한계: NVIDIA GPU 전용 (cusolvermg backend), AMD GPU 미지원.
  56k+ face 같이 LU 자체 메모리가 ~250 GB 인 경우 4 GPU pool (192 GB)
  로도 fit 안되며 H-matrix 와 결합 필요 (v1.3.0 §3.13 참고).

### 3.13 H-matrix BEMRetIter integration (v1.3.0, Lane E2 후속)

큰 mesh (25k+ face) 의 dense LU 메모리 한계 (50+ GB peak) 를
ACA H-matrix + GMRES 로 해소. v1.2.0 의 VRAM share 가 dense LU 의
*메모리 풀* 을 만들어 GPU 한계를 우회한 반면, v1.3.0 의 H-matrix iter
는 *알고리즘 차원* 에서 메모리·matvec 모두 `O(N log N)` 으로 줄인다.

| 모드 | 메모리 | matvec 비용 | 적합 mesh |
|---|---|---|---|
| dense BEMRet | `O(N^2)` | `O(N^2)` | < 5k face |
| dense + VRAM share (v1.2.0) | `O(N^2)`, multi-GPU pool | `O(N^2)` | 25k face (2 GPU pool) |
| **H-matrix BEMRetIter (v1.3.0)** | `O(N log N)` | `O(N log N)` per iter | 25k+ face |
| H-matrix + VRAM share | `O(N log N)`, multi-GPU pool | `O(N log N)` | 56k+ face (실험적) |

구현 요점:

- M4 H1 의 H-matrix Green function module 을 `BEMRetIter` /
  `BEMStatIter` 와 통합. ACA 가 block-level 압축에서 *전체 H-tree*
  level 로 확장됨.
- GMRES 의 matvec op 가 H-matrix matvec 을 호출 — `O(N log N)` per
  iter.
- `hmatrix=True` opt-in (default 동작은 v1.2.0 과 동일).
- VRAM share 와 결합 가능: H-matrix 자체는 단일 GPU 메모리 fit,
  dense 부분 (preconditioner 등) 만 multi-GPU pool 이 처리.

활성:

- `BEMRetIter(p, hmatrix=True)`, `BEMStatIter(p, hmatrix=True)`.
- `htol`, `kmax`, `cleaf` 파라미터 노출 (default 는 v1.0.0 ACA 와
  동일).
- `pymnpbem_simulation` 의 `iter.hmatrix: 'auto'` 가 5000+ face mesh
  에서 자동 활성.

한계:

- `BEMRetLayerIter + hmatrix` 미지원 (`NotImplementedError`) —
  cover layer + planar substrate 결합 시나리오 없음.
- `BEM*Iter + Schur (v1.2.0)` 동시 활성 미지원 — H-matrix + Schur
  통합은 후속 작업.
- preconditioner 는 현재 Jacobi (`precond='diag'`) 만 H-matrix 와
  호환. H-matrix block-LU preconditioner 는 후속.

### 3.14 CPU/GPU 분리 build (v1.4.0)

핵심 결정: **single wheel + extras** (별도 wheel 분리 X).

이유:

- PyPI 표준 패턴 — 대부분의 ML/numerical 패키지가 `[gpu]` 같은
  extras 형태로 GPU 의존성을 분리한다 (예: `tensorflow`,
  `jax[cuda12]`, `pytorch-lightning[extra]`).
- 코드 자체는 cupy lazy import — `mnpbem/utils/gpu.py` 가
  cupy `ImportError` 를 catch 하여 CPU path 로 fallback. 따라서
  별도 wheel 을 만들지 않아도 단일 코드 베이스로 CPU/GPU 양쪽
  동작이 가능.
- 사용자 가치 대비 별도 wheel (예: `mnpbem-cpu`, `mnpbem-gpu`)
  빌드 + 유지 비용이 큼 — 두 build matrix 를 CI 에서 별도로
  돌려야 하고, 사용자도 이름이 다른 패키지를 골라야 한다.

대신 **정교한 extras + runtime auto-detect** 로 사용자 경험 개선:

| Extra | 추가 의존성 | 용도 |
|---|---|---|
| `mnpbem` (default) | (none) | CPU only — numpy/scipy/numba 만 |
| `mnpbem[gpu]` | cupy-cuda12x | NVIDIA GPU 가속 (Tier 4 G1/G2) |
| `mnpbem[mpi]` | mpi4py | multi-node wavelength 분배 (Lane D) |
| `mnpbem[fmm]` | fmm3dpy | free-space ret meshfield 가속 (F1) |
| `mnpbem[all]` | gpu + mpi + fmm | 전부 |
| `mnpbem[dev]` | pytest / ruff / build / twine | 개발 환경 |
| `mnpbem[test]` | pytest | 회귀 테스트만 |
| `mnpbem[docs]` | (sphinx 등) | docs build (예약) |

Runtime auto-detect (`mnpbem/utils/gpu.py`):

- `has_gpu_capability(verbose=True)` — cupy import + CUDA driver +
  GPU device 검사. 누락된 항목별 친절한 메시지.
- `get_install_hint()` — 현재 환경에서 GPU 활성을 위해 필요한
  `pip install mnpbem[gpu]` 명령 안내 문자열.
- `MNPBEM_GPU=1` env var 가 명시되었지만 cupy 가 미설치되면
  BEM solver 호출 시점에 `RuntimeError` + install 명령 안내
  (기존 v1.3.0 까지의 silent fallback 보다 명확).

사용자 가이드: `docs/INSTALL.md` (v1.4.0 신규) — 시나리오별
(CPU only / single GPU / multi-GPU / multi-node / 개발) install
절차를 한 곳에 모아 둠.

### 3.15 H-matrix LU preconditioner + Schur × Iter (v1.5.0)

큰 nonlocal mesh 의 GMRES stall 해결 + cover layer 변수 implicit
소거. v1.3.0 의 H-matrix BEMRetIter (`O(N log N)` 메모리/matvec) 가
"풀리는가" 문제는 해결했지만, "충분히 빠르게 수렴하는가" 문제는
preconditioner 가 필요했고, "cover layer 가 추가된 8N×8N 시스템"
문제는 Schur × Iter 통합이 필요했다. v1.5.0 이 두 격차를 메운다.

#### Preconditioner

- 구현: `mnpbem/bem/preconditioner.py` (`HMatrixLUPreconditioner`).
- modes: `dense` (alpha-1, full N×N LU on `H.full()`), `tree`
  (alpha-2, hierarchical block-Schur LU on H-tree root partition),
  `auto` (size 기반 dispatch).
- BEM solver 측 노출: `BEMRetIter(p, hmatrix=True,
  preconditioner='auto', htol_precond=1e-4)`. `BEMStatIter` 동일.
- 256-face sphere benchmark: GMRES iter 55 → 1 (55× 감소),
  wall 1.03 s → 0.82 s.
- 한계 — `BEMRetIter` 의 8N×8N 결합 시스템에서는 G-only H-tree LU
  단독 효과가 제한적이라 `hlu_tree` 가 dense fallback 으로 동작하는
  경우가 있음. 25k face 의 진정한 memory-friendly preconditioner 는
  Sigma/Delta 자체를 H-matrix 로 재구성해야 하는데, 이는 v1.6+
  과제. `BEMStatIter` tree mode 는 diagonal term 깨져서 dense
  fallback (one-time log).

#### Schur × Iter

- 구현: `mnpbem/bem/schur_iter_helpers.py` (`SchurIterOperator`,
  `LinearOperator` subclass).
- 연산: `A_eff(x_c) = A_cc x_c − A_cs · A_ss⁻¹ · A_sc x_c` —
  GMRES 가 reduced (core) 차원만 보면 됨. `A_ss⁻¹` 은
  `lu_dense` (한 번 dense LU + reuse) / `gmres` (inner Krylov) /
  `callable` (user-supplied) / `auto` (shell DOF 기반 dispatch)
  중 선택.
- BEM solver 측 노출: `BEMRetIter(p, hmatrix=True, schur=True)` —
  v1.4 까지 `NotImplementedError` 였던 조합이 v1.5.0 부터 가능.
  cover layer 자동 감지 (v1.2.0 `detect_shell_core_partition` 재활용).
- 568-face nano-gap nonlocal benchmark: solve 21.17 s → 16.65 s
  (21.3% 절감). reduced GMRES Krylov 차원 + `A_ss⁻¹` reuse 효과.
- v1.2.0 dense Schur 와 비교 — dense 는 `G_ss` 를 직접 inverse
  하므로 H-matrix 와 호환 X. `SchurIterOperator` 는 *full matvec
  A·v* 만 사용해 H-matrix 와 자연스럽게 맞물린다.

### 3.16 v1.6.0 — B-Schur full coverage + BEMRetLayerIter operator-form + CLI

#### B-Schur (mnpbem/bem/schur_iter_helpers.py)

`SchurIterOperator` 가 `eps_form='operator'` 시 inner GMRES 부담 회피 위해
`g_ss_solver='auto'` 의 lu_dense threshold 를 500 → 4096 으로 상향. dense LU
probe 가 N=4096 (≈128 MB at complex128) 까지 효율적.

수학: Schur reduction 자체는 `A_full` 에 무관하게 정확. 통찰 — operator-form
Schur 재구현 불필요. inner GMRES 비용이 진짜 bottleneck.

#### BEMRetLayerIter (mnpbem/bem/bem_ret_layer_iter.py)

`_afun / _init_precond / _mfun` 모두 operator-form 적용 (β v1.5.1 패턴 동일).
substrate + iter + multi-material 케이스 drift 해결.

#### pymnpbem CLI (cli.py)

`--str-conf <X.py> --sim-conf <Y.py> --verbose` 패턴 (mnpbem_simulation 호환).
sim_conf 의 nested compute 블록에 모든 worker/GPU 파라미터.

## 4. Performance summary

See `docs/PERFORMANCE.md` for the numbers. The strategy is documented in
`docs/PERFORMANCE_STRATEGY.md` (M4 four-tier roadmap). The short version is:

- Tier 1 (`scipy` `check_finite=False`/`overwrite_a=True`): 10-20% LU gain.
- Tier 2 (multi-RHS wavelength batching, GMRES iterative path).
- Tier 3 (Numba kernels): 5-50× on the dense Green fill.
- Tier 4 (GPU + H-matrix + FMM3D): order-of-magnitude on large meshes.

Headline numbers: CPU geometry-build speedup 2.21×, GPU geometry-build
speedup 3.60×, vs. the pre-M4 baseline.

## 5. Testing

See `tests/regression/README.md` for the regression infrastructure.
The validation hierarchy is:

- `tests/` — unit tests (pytest). 600+ tests covering Mie, EELS, layer,
  mirror, iterative, edge cases.
- `validation/` — sphere-and-rod numerical cross-checks against MATLAB,
  driven from `validation/_common`. Each subdirectory is one validation
  axis (Mie, BEMStat, BEMRet, BEMStatLayer, BEMRetLayer, mirror,
  eigenmode, iterative, dipole, dipole-layer, EELS, near-field, shapes).
- `72demos_validation/` — full MATLAB-vs-Python demo harness driving the
  72 demo scripts and producing the `compare_smart_v3.py` accuracy table.

CI runs unit tests on every commit and regression suites on tagged
releases (see `.github/workflows/`).

## 6. Pointers for new contributors

- Adding a particle shape: extend `mnpbem.geometry.mesh_generators`, mirror
  the MATLAB `+particles/` files line-by-line, and add a Mie cross-check
  if it has a closed-form reference.
- Adding a Green function: subclass the appropriate `CompGreen*` and
  register a factory entry in `mnpbem.greenfun.greenfunction.greenfunction`.
- Adding a BEM solver: extend `BemBase`, mirror the MATLAB `@bem*` class,
  register in `mnpbem.bem.solver_factory.solver`, and add the regression
  hook in `tests/`.
- Adding an excitation: place it under `mnpbem.simulation`, expose it in
  `simulation/__init__.py`, and register the factory.

When in doubt, the rule is: **read the MATLAB source, port it line-by-line,
and only deviate when explicitly justified in this document**.
