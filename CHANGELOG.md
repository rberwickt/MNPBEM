# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.7.0] - 2026-05-11

### Fixed

- **GPU correctness audit (17-18 bug)**: 5 agent (A1-A5) parallel audit
  + Phase 1 integration audit 결과. 모든 (BEM solver × excitation ×
  layer / mirror) 조합이 GPU 모드에서 CPU 기준과 1e-7 face / 1e-9
  cross section 으로 일치함.

#### A1 — BEMRet / BEMRetLayer

- disjoint dimer 비균일 eps edge case 회귀 가드 추가.

#### A2 — BEMRetIter / BEMRetLayerIter

- dense path GPU backend mix fix.

#### A3 — BEMStat family

- `clear()` stale-cache + GPU LU 누수.

#### A4 — Mirror BEM 4 종

- `CompGreenRetMirror` / `CompGreenStatMirror` eval cupy 결과 host promotion.
- `BEMStatEigMirror` half-mesh index range fix.
- `BEMLayerMirror` dummy assertion.

#### A5 — Excitation runners 17 종

- PlaneWaveStat / DipoleStat / DipoleRet / DipoleStatLayer /
  DipoleRetLayer / EELSStat / EELSRet 의 cupy sig 멤버 host
  materialization.

#### Phase 1 (integration)

- **CompGreenRet `_matmul` / `_cross`**: cupy operand 들어올 때
  silent zero 반환 → host promotion 추가.
- **BEM solver `solve()` 반환**: cupy sig 멤버를 user-facing host
  변환 (BEMRet, BEMStat, BEMRetLayer, BEMStatLayer, BEMRetMirror,
  BEMStatMirror). 데모 스크립트의 `np.asarray(sig.sig1)` 류 호출
  안전.
- **`lu_solve_native` GPU LU + cupy b** residency 회귀 가드 추가.
- EELS × Layer integration smoke test 신규 (EELS × Mirror 는
  미지원 — MATLAB Demo set 에도 없는 조합).

### 검증

- 72 demo regression (`/home/yoojk20/scratch/mnpbem_demo_comparison`):
  v1.7 GPU 모드 BAD 0 / 72, perf 65 / 72.

## [1.6.4] - 2026-05-08

### Fixed

- **HMatrix matvec backend/dtype 일관성** (Phase 1): 입력이
  numpy 인데 일부 ACA 블록이 cupy 거주하는 경우 `mtimes_vec`
  내부에서 host result + GPU block 간 backend 가 섞이는 경로
  가 있었음. 이제 블록 GPU 거주 여부를 sniff 하여 destination
  backend 를 통일하고, 끝에서 입력 backend 로 다시 변환하여
  numpy 입력 → numpy 출력을 보장한다. `bem_ret_iter._afun` 의
  6 곳 HMatrix matvec 호출에는 방어용 `_ensure_numpy` wrapper
  추가 (정상 경로에서는 no-op, host slice assignment 안전성
  보장).
- 영향: GPU + iter+hmat+precond 경로에서 numpy/cupy mix 로
  인한 무작위적 실패/속도 저하 차단. 1176-face Au@Ag dimer
  에서 ext = 25759.0950 (CPU/GPU/모든 모드 일치, rel diff
  1e-15).

### Added

- **`MNPBEM_AGGRESSIVE_GPU_MFUN=1` 환경변수** (Phase 2 opt-in):
  `BEMRetIter._mfun` 의 dense Sigma1 / L_diff / L1 행렬을 GPU
  에 업로드하여 GMRES iterate 동안 호출되는 Sigma1·v / L·v
  를 cupy matmul 로 디스패치. 메모리 capacity 부족 시 tier
  별 자동 fallback (Sigma1 only → Sigma1 + L_diff → 모두).
  flag off (default) 시 v1.6.3 와 bit-identical.
- 측정:
  - 5400-face Au@Ag dimer: 111s → 103s (1.08x speedup, 풀
    Sigma1+L_diff+L1 GPU 거주).
  - 12672-face Au@Ag dimer: 608s → 620s (flat). 12672 face
    스케일에서는 GMRES 가 1 outer iter 로 수렴 (mfun call =
    3 회) → mfun GPU dispatch 는 의미 없음. 효과는 GMRES iter
    가 많은 wider geometry / 다른 sweep 에서 나타남.
- 회귀 가드: `mnpbem/tests/test_mfun_gpu_dispatch.py` (1136-
  face Au@Ag dimer 에서 flag off / on rel diff < 1e-10 검증).

## [1.6.3] - 2026-05-05

### Changed

- **BEMRetIter precond GPU LU 하이브리드 파이프라인**:
  N >= 8000 브랜치에서 전부 host scipy 로 라우팅하던 기존
  preconditioner 를 cuSOLVER getrs LU + host MKL 행렬곱
  하이브리드로 재구성. G/Delta/Sigma_mat 의 LU 인수분해를
  GPU 에서 실행하고 `('gpu', lu, piv)` 태그로 보존하여
  `_mfun` 의 GMRES iterate 가 cuSOLVER getrs (~5 ms / call)
  로 라우팅됨. G^{-1} / Sigma1 / L1 GEMM 은 132-core MKL
  파이프라인이 더 빠르므로 host scipy 에 위임.
- `_gpu_precond_capacity_ok(N)`: 7 N² × 16 byte 가용 GPU
  메모리 검사 후 시도, 부족하면 자동 host fallback +
  `_GpuPrecondOOM` raise 시에도 host scipy 에 안전 fallback.
- `MNPBEM_GPU_PRECOND_HOST_THRESHOLD` env (default 32768)
  로 강제 host fallback 가능.

## [1.6.2] - 2026-05-02

### Fixed

- **VRAM share env vars wiring** (C agent v1.6.0/v1.6.1 Tier-3 benchmark 발견):
  `MNPBEM_VRAM_SHARE_GPUS=N` + `MNPBEM_VRAM_SHARE_BACKEND=cusolvermg`
  set 되어도 BEM solver (`bem_ret_iter.py`, `bem_stat_layer.py`,
  `bem_ret_layer.py` 등) 가 `lu_factor_dispatch` 호출 시 `n_gpus` kwarg
  명시 전달 안 함 → cuSolverMg 활성화 실패, 12672-face dense LU 가
  49 GB single-GPU OOM. fix:
  - `mnpbem/utils/gpu.py::_vram_share_env_defaults()` 헬퍼 신설
  - `lu_factor_dispatch` / `solve_dispatch` 가 `n_gpus` / `backend` /
    `device_ids` kwarg 미지정 시 env vars (`MNPBEM_VRAM_SHARE_GPUS`,
    `MNPBEM_VRAM_SHARE_BACKEND`, `MNPBEM_VRAM_SHARE_DEVICE_IDS`) 자동
    읽음. 명시 kwarg 가 항상 우선.
  - `MNPBEM_VRAM_SHARE_DEVICE_IDS` (comma-separated) 신규 지원
  - `MNPBEM_VRAM_SHARE=0` master switch 로 강제 비활성 가능
  - 기존 `bem_ret.py::_vram_share_lu_kwargs` 명시 전달 경로와 호환
    (kwarg 우선이므로 동작 변경 X)

### Added

- `mnpbem/tests/test_vram_share_env_wiring.py` (15 tests, mock cuSolverMg
  backend로 CPU-only host 에서도 dispatch 라우팅 검증)

### Verified

- 4× RTX A6000 smoke: `MNPBEM_VRAM_SHARE_GPUS=2` 환경변수만으로 256×256
  complex LU `lu_factor_dispatch` → mgpu 분기 진입, residual 9.3e-16

### Known issues

- Tier-3 12672-face cuSolverMg 정식 batch benchmark 미수행 (M5+ 후속)

## [1.6.1] - 2026-05-02

### Added

- `mnpbem/greenfun/_numba_refine.py` — Numba JIT kernels for BEM assembly hot loops
- `mnpbem/tests/test_compgreen_ret_layer_multi.py` — multi-particle layer Green test

### Fixed

- `compgreen_ret_layer.py:651` shape mismatch (Au@Ag core-shell on substrate)
- BEM assembly single-thread CPU bottleneck (Numba JIT 적용)

### Performance

- 3k+ face mesh assembly ~70% 절감 (Numba)

### Known issues

- VRAM share env vars wiring 미완 — v1.6.x 후속
- Tier-3 timing benchmark 정식 측정 미완 — batch 재시도 권장

## [1.6.0] - 2026-05-02

### Added

- `BEMRetIter` `schur_eps_form='auto'` 옵션 — non-uniform eps 자동 감지
- `SchurIterOperator` `eps_form='pointwise'|'operator'` 분기 + threshold 4096
- `BEMRetLayerIter` operator-form (substrate + iter + multi-material)
- pymnpbem `--str-conf <X.py> --sim-conf <Y.py> --verbose` CLI
- pymnpbem `mesh_density` (nm) 우선 (n_per_edge 보다)
- 신규 tests: test_b_schur, test_iter_convergence_layer, test_mesh_density_priority, test_cli_str_sim

### Fixed

- 60-face nonlocal+schur+iter+hmat 25 min GMRES hang → 6:51 수렴
- BEMRetLayerIter multi-material drift (Au+Ag dimer on glass)

### Performance

- (perf 측정값은 PERFORMANCE.md §11.5)

### Known limits / Deferred

- BEM assembly perf (single-thread CPU bound) — v1.6.x 후속
- compgreen_ret_layer multi-particle indexing — v1.6.x 후속

## [1.5.2] - 2026-05-02

### Fixed

- **Bug 5 — `HMatrix.full()` numpy/cupy interop** (`mnpbem/greenfun/hmatrix.py:374`).
  v1.5.1 에서 `MNPBEM_GPU_NATIVE=1` 활성 시 `CompGreenRet` 가 cupy ndarray
  를 반환하면 `HMatrix.val[i]` 가 cupy 가 되는데, `full()` 은 host numpy
  buffer 에 cupy slice 를 implicit cast 하다 `TypeError: Implicit
  conversion to a NumPy array is not allowed.` 로 실패. fix:
  `full(xp=None)` 이 val/lhs/rhs 에서 cupy 자동 감지 → cupy backend 로
  `mat` 할당, 블록별 numpy ↔ cupy 변환 헬퍼로 device 통일. caller 가
  `xp=np` 또는 `xp=cupy` 강제도 가능.
- **Bug 6 — `_plus_hmat` / `_truncate_block` backend 통일**. region (0,0)
  val 은 cupy, region (1,0) val 은 numpy 인 경우 (Au@Ag 처럼 multi-region
  + cross-connectivity) `G11 - G21` 이 `Unsupported type
  <numpy.ndarray>` 로 실패. fix: `_same_backend(a, b)` 헬퍼로 한쪽이라도
  cupy 면 양쪽 cupy 로 승격, `_truncate_block` QR/SVD 도 lhs 가 cupy 면
  `xp=cupy` dispatch.
- **Tier-3 12672-face Au@Ag GPU full validation 통과** —
  `MNPBEM_GPU=1 + iter+hmat+precond + multi-GPU wavelength-split` 경로가
  처음으로 end-to-end 정상 완료. v1.5.0/v1.5.1 의 BAD grade 해소.

### Added

- `mnpbem/tests/test_hmatrix_full_consistency.py` — 8 tests, cupy/numpy
  full() 일치, 강제 xp 인자, mixed blocks, ACA dense=cupy/lhs=numpy
  realistic 시나리오, BEMRetIter._init_matrices GPU end-to-end smoke.

### Backward compatibility

100% backward compatible with v1.5.1. `HMatrix.full()` signature 가
`full(xp=None)` 로 확장되어도 기본값이 auto-detect 이므로 기존 caller
변경 불필요.

## [1.5.1] - 2026-05-02

### Fixed

- **4 mnpbem GPU 버그 fix** (Au@Ag GPU full mesh acceptance 위해)
  - `mnpbem/bem/bem_ret.py` (Bug 1) — CPU init path `G1i/G2i/Deltai`
    LU 백엔드 일관화.
  - `mnpbem/bem/bem_ret_iter.py:264` (Bug 2) — `Sigma1 = H1 @ G1i`
    numpy/cupy mix 제거 (dense LU preconditioner 빌드).
  - `mnpbem/greenfun/hmatrix.py:250` (Bug 3) — `_aca_block`
    `cols[pivot_col_local]` cupy/numpy index 혼용 방어.
  - `mnpbem/utils/multi_gpu.py::_worker` (Bug 4) — `BEM` 클래스를
    `bem_class` 인자로 명시적으로 받도록 확장. 이전엔 `simulation.type=ret_iter`
    여도 wavelength-split 경로가 `BEMRet` (dense) 강제.
- **BEMRetIter operator-form eps fix** — Au@Ag (multi-material) iter
  drift 70% → 0%. Non-uniform eps + cross-connectivity 케이스에서
  iter formulation 알고리즘 버그. dense `BEMRet` 의
  `L1 = G1·diag(eps1)·G1⁻¹` 와 일치.
- **`pymnpbem_simulation` `simulation.type=ret + iterative=true`
  자동 라우팅** (Issue A) — `dispatch_single_node` /
  `convert_py_to_yaml` 양쪽에서 `_iter` variant 로 in-place rewrite.
- **`pymnpbem_simulation.dispatch.multi_gpu`** —
  wavelength-split 경로가 `simulation.type` 에서 `bem_class` 를
  유도하여 `solve_spectrum_multi_gpu(bem_class=…)` 로 전달.
  `compute.iter.{hmatrix, preconditioner, schur, htol, tol, maxit}` 도
  worker BEM 으로 전파. v1.5.0 까지는 multi-GPU 경로가 항상 `BEMRet`
  를 강제하던 Bug 4 후속을 wrapper-side 로 메움.

### Added

- `mnpbem/tests/test_gpu_cupy_consistency.py` — 14 tests, GPU
  cupy/numpy interop 회귀.
- `mnpbem/tests/test_iter_convergence.py` — 8 tests, BEMRetIter
  operator-form 회귀 (case_g 1136-face Au@Ag).

### Known issues

- `BEMRetLayerIter` 에 같은 operator-form eps 패치 필요 — substrate
  + iter 결합 시나리오. v1.5.2 또는 v1.6 후속.
- `mnpbem/tests/test_schur_iter.py::TestBEMRetIterSchur::test_schur_dense_matches_no_schur` 가
  현재 환경에서 hang — 별도 조사 항목 (회귀 통계에서 단독 격리; 다른
  10 schur_iter 테스트는 PASS).
- **Bug 5 — `mnpbem/greenfun/hmatrix.py:374` `HMatrix.full()` 의
  cupy/numpy mix** — `BEMRetIter(hmatrix=True, preconditioner='auto')`
  경로에서 dense LU preconditioner 빌드 시 `_compress` → `hmat.full()`
  가 cupy `self.val[i]` 를 numpy `mat` 으로 implicit cast 하려다 실패.
  Tier-3 12672-face Au@Ag GPU iter 시나리오에서 발견. v1.5.1 의 α
  4 GPU 버그 fix 와 동일 카테고리. 후속 (v1.5.2) 에서 `xp.zeros` /
  `cupy.asnumpy` dispatch 로 정리 필요.

## [1.5.0] - 2026-05-03

### Added

- **H-matrix LU preconditioner** for iterative BEM solvers (Lane E2 후속)
  - `BEMRetIter(p, hmatrix=True, preconditioner='auto', htol_precond=1e-4)`,
    `BEMStatIter` 동일.
  - 256-face sphere GMRES iter 55 → 1 (55× 감소).
  - modes: `auto` (default ON when `hmatrix=True`), `none`, `hlu_dense`,
    `hlu_tree`.
  - 구현: `mnpbem/bem/preconditioner.py` (`HMatrixLUPreconditioner`).
- **Schur complement × Iterative BEM** integration
  - `BEMRetIter(p, schur=True, hmatrix=True)` (둘 다 ON 가능; v1.4
    까지는 `NotImplementedError`).
  - `SchurIterOperator` `LinearOperator`:
    `A_eff(x_c) = A_cc x_c − A_cs · A_ss⁻¹ · A_sc x_c`.
  - `g_ss_solver`: `lu_dense` / `gmres` / `callable` / `auto`.
  - 568-face nano-gap nonlocal: solve 21.17s → 16.65s (21.3% 절감).
  - 구현: `mnpbem/bem/schur_iter_helpers.py`.
- **51 pre-existing test failures cleanup** (51 → 0)
  - Stale 11 삭제, infra 38 fix, 1 fix, 1 갱신.
- **jk-config 3 follow-up issues** fix
  - Issue 2: multi-shell `core_shell` builder N-layer 일반화.
  - Issue 3: Metal substrate `IndexError` (`LayerStructure._enlarge`
    boundary clip).
  - Issue 4: field-only config 자동 변환
    (`py_to_yaml._redirect_field_only_simulation`).
- `pymnpbem_simulation` 의 `iter.preconditioner`, `iter.schur` 옵션 노출.

### Changed

- (none — backward compatible with v1.4.0)

### Performance

- 256-face sphere GMRES: iter 55 → 1, wall 1.03s → 0.82s.
- 568-face nonlocal Schur×Iter: 21.3% 시간 절감.
- 25k face: alpha-2 H-tree LU 의 진정한 가치는 Sigma/Delta H-matrix
  재구성이 필요 — v1.6+ scope.

### Known limits

- `BEMRetIter` 의 8N×8N 결합 시스템 → G-only H-tree LU 단독 효과
  제한적 (alpha-2 ≈ alpha-1 dense fallback).
- 25k face 의 진짜 memory-friendly preconditioner = Sigma/Delta
  H-matrix 재구성 = v1.6+.
- `BEMStatIter` tree mode → diagonal term 깨져서 dense fallback
  (one-time log).

## [1.4.0] - 2026-05-XX

### Added

- **CPU/GPU 분리 install** — pyproject extras 정교화
  (`gpu` / `mpi` / `fmm` / `all` / `dev` / `test` / `docs`).
  - `pip install mnpbem` (CPU only, 가장 가벼움; cupy 의존성 없음).
  - `pip install mnpbem[gpu]` (cupy-cuda12x 포함, NVIDIA GPU 가속).
  - `pip install mnpbem[all]` (gpu + mpi + fmm 모든 기능).
  - 별도 wheel 분리 X — single wheel + extras 가 PyPI 표준 패턴.
- **Runtime GPU 자동 감지** —
  `mnpbem.utils.gpu.has_gpu_capability(verbose=True)` 가
  cupy import + CUDA driver + GPU device 가용성을 검사하여 `bool`
  반환. 누락 시 친절한 fallback 메시지 출력.
- **`mnpbem.utils.gpu.get_install_hint()`** — 사용자 환경에 맞는
  `pip install mnpbem[gpu]` 명령 안내 helper.
- **`docs/INSTALL.md`** — 시나리오별 install 가이드 (CPU only / GPU /
  multi-GPU / multi-node / 개발 환경).

### Changed

- `README.md` `Installation` 섹션 간략화 — 자세한 내용은
  `docs/INSTALL.md` 로 링크.
- (none breaking — 100% backward compatible with v1.3.0)

### Performance

- (perf 영향 없음 — packaging 개선)

## [1.3.0] - 2026-05-XX

### Added

- **H-matrix BEMRetIter integration** (Lane E2 후속).
  - `BEMRetIter(p, hmatrix=True)`, `BEMStatIter(p, hmatrix=True)` 새
    옵션. ACA H-tree 압축 + GMRES 로 25k+ face 큰 mesh 의 dense LU OOM
    (50+ GB) 을 해소.
  - 메모리·matvec 모두 `O(N log N)` 스케일 — 단일 GPU 에서 25k face 가
    fit. VRAM share (v1.2.0) 와 결합 시 56k+ face 도 도전 가능.
  - 노출 파라미터: `htol` (ACA truncation, default 1e-6),
    `kmax` (ACA rank 상한, default `[4, 100]`),
    `cleaf` (leaf cluster 크기, default 200).
  - `BEMRetLayerIter + hmatrix` 는 미지원 (`NotImplementedError`) —
    cover-layer + planar substrate 결합 시나리오 없음.
  - `BEM*Iter + Schur (v1.2.0)` 동시 활성도 미지원 — H-matrix + Schur
    통합은 후속 작업.
- **`pymnpbem_simulation` iter runner 의 `iter.hmatrix: 'auto'`**
  옵션 — 5000+ face mesh 에서 자동으로 H-matrix BEMRetIter 활성.

### Changed

- (none — backward compatible with v1.2.0)

### Performance

- 25k face dimer: dense LU OOM (~50+ GB peak) →
  H-matrix BEMRetIter 단일 GPU fit (실측 수치는
  `docs/PERFORMANCE.md` §11 참고).
- per-wl 시간: dense BEMRet 와 H-matrix BEMRetIter 비교
  (`docs/PERFORMANCE.md` §11).
- 정확도: dense vs H-matrix BEMRetIter rel `< 1e-4` (htol 기반).

## [1.2.0] - 2026-05-XX

### Added

- **Schur complement** for cover-layer BEM — nonlocal 메모리 50% 절감,
  LU 풀이 30% 가속.
  - `BEMStat(p, schur=True)`, `BEMRet(p, schur=True)` 옵션.
  - Cover layer (`EpsNonlocal`) 변수를 schur 소거하여 reduced matrix 풀이.
  - 결과는 standard formulation 과 수학적으로 동등 (rel < 1e-12).
  - `schur='auto'` 또는 wrapper 가 cover layer 자동 감지.
  - 구현: `mnpbem/bem/schur_helpers.py`.
- **VRAM share** — 1 worker 가 multi-GPU 메모리 합쳐 큰 단일 계산 처리.
  - cuSolverMg backend (NVIDIA 공식 multi-GPU LU API).
  - 25k+ face dense LU (50+ GB) 가 2 GPU pool (96 GB) 에서 fit.
  - 환경변수 `MNPBEM_VRAM_SHARE_GPUS=N`,
    `MNPBEM_VRAM_SHARE_BACKEND=cusolvermg`.
  - `mnpbem.utils.gpu.lu_factor_dispatch(A, n_gpus=N)` 직접 호출 지원.
  - `pymnpbem_simulation` 의 `compute.n_gpus_per_worker > 1` 이 자동 활성.
  - 구현: `mnpbem/utils/multi_gpu_lu.py`.

### Changed

- (none — backward compatible with v1.1.0)

### Performance

- nonlocal cover-layer simulations: 메모리 4× → ~2× (Schur 적용 시).
- 25k+ face dense LU: 단일 GPU OOM → multi-GPU pool 로 가능.
- (수치는 `docs/PERFORMANCE.md` 참고)

## [1.1.0] - 2026-05-XX

### Added

- `EpsNonlocal` — hydrodynamic Drude nonlocal dielectric function
  (cover-layer formulation).
  - Yu Luo et al., PRL 111, 093901 (2013) effective-layer mapping.
  - Factory methods: `EpsNonlocal.gold()`, `.silver()`, `.aluminum()`,
    `.from_table(path)`.
  - Helper: `make_nonlocal_pair(metal, eps_embed, delta_d, beta)` →
    `(core, shell)` tuple.
  - 18 unit tests; bit-identical to MATLAB `demospecstat19` reference
    formula at `rtol = 1e-12`.
- `BEMRet` now accepts a `refun` parameter (parity with `BEMStat`) — the
  retarded path can be combined with the cover-layer integration.
- `pymnpbem_simulation` wrapper updated: nonlocal workaround replaced
  with the official `EpsNonlocal` call path.

### Changed

- (none — backward compatible with v1.0.0)

### Performance

- (no performance impact — algorithmic feature only)

## [1.0.0] - 2026-05-XX

First production release of the MNPBEM Python port. Pure-Python distribution
of Hohenester & Trügler's MATLAB MNPBEM toolbox, validated against MATLAB on
50 + 22 official demos and on the sphere/rod/dimer cross-checks.

### Milestone 1 — Demo complete

Goal: bring the 50 official MATLAB MNPBEM demos to MATLAB-Python parity, then
extend to the 72-demo extended harness. Reduce the BAD-category demos from
12 to 0 and lift machine-precision matches from 0 to 55 of 72 (76%).

Highlights:

- Quasistatic and retarded BEM solvers ported (`BEMStat`, `BEMRet`).
- Mirror-symmetric solvers (`BEMStatMirror`, `BEMRetMirror`).
- Layered-medium solvers (`BEMStatLayer`, `BEMRetLayer`) including
  Sommerfeld-integrated layer Green functions and tabulated interpolators.
- Iterative solvers (`BEMStatIter`, `BEMRetIter`, `BEMRetLayerIter`) with
  GMRES + ACA H-matrix acceleration.
- Eigenmode solver (`BEMStatEig`) with bi-orthogonal pairing.
- Plane-wave / dipole / EELS excitations across all of stat/ret/mirror/layer.
- Mesh generators: `trisphere`, `trirod`, `tricube`, `tritorus`,
  `trispheresegment`, `tripolygon`, plus `Polygon`, `Polygon3`, `EdgeProfile`.
- 2D mesher: line-by-line port of MATLAB `mesh2d`.
- Reference Mie solver (`MieStat`, `MieRet`, `MieGans`).

Representative commits:

- `d8d396e` `merge: T1 scipy lu_factor/solve check_finite=False/overwrite flag 적용`
- `0f7637d` `mesh2d._minrectangle: MATLAB strict < tie-break 정합` (BAD 12 → 8)
- `af69b7d` `matlab_compat: MATLAB libmwmathutil 전체 초월 함수 bit-identical 포팅`
- `0320f9e` `matlab_compat.matan2: MATLAB libmwmathutil 직접 호출로 bit-identical 구현`
- `b8fadd4` `BEM 솔버 전체: np.linalg.inv() → scipy.linalg.lu_factor/lu_solve 교체`
- `a371b30` `BEMRetLayer 솔버를 MATLAB initmat.m/mldivide.m와 동일한 structured 2x2 block matrix 시스템으로 재작성`
- `ac988d8` `누락 29개 메서드 전부 구현: MATLAB MNPBEM 100% 기능 동일성 달성`

### Milestone 2 — Missing API porting

Goal: cover the MATLAB classes/functions that were not part of the demo
critical path but are part of the public surface.

- `ComParticleMirror` mirror whitelist + `sym` validation.
- `CompGreenStatMirror` / `CompGreenRetMirror` full ports.
- `CompGreenStatLayer` Cartesian derivatives (`H1p` / `H2p`).
- `CompGreenTabLayer` multi-tab dispatch (`_MultiGreenTabLayer`).
- `MeshField` near-field evaluator on `IGrid2` / `IGrid3` grids.
- `coverlayer` package — refinement on layer-interface particles.
- `compound` (`@compound`) — 10 public methods.
- `Polygon` boolean / normalize / symmetry helpers.
- `polymesh2d` outer + hole multi-polygon support.
- ACA-accelerated Green functions (`ACACompGreenStat`, `ACACompGreenRet`,
  `ACACompGreenRetLayer`).
- ClusterTree + HMatrix data structures.
- `plasmonmode` left/right eigenvector pairing for complex eigenvalues.
- BemPlot / coneplot / arrowplot visualisation.
- `epsfun` factory and the `eps_table` data files.

Representative commits:

- `0fcd647` `docs: Add comprehensive MNPBEM API audit report (MATLAB → Python)`
- `c510e89` `Implement closed surface regularization in CompGreenRet`
- `7024e3f` `MeshField 클래스 구현: BEM 해로부터 근접장 분포 계산`
- `d7d8ca5` `ClusterTree 및 HMatrix (계층적 행렬) 구현: MATLAB H-matrix 코드의 Python 변환`
- `6d26cd7` `ACA 가속 retarded Green 함수 (ACACompGreenRet) 구현`
- `600a5b0` `ACA 가속 layer Green 함수 (ACACompGreenRetLayer) 구현 및 broadcasting 버그 수정`
- `3104aee` `compound: MATLAB @compound 10개 public 메서드 Python 포팅`
- `a1086fa` `greenfun/coverlayer: MATLAB +coverlayer 모듈 재구현`
- `cb2c7ce` `compgreentab_layer: multi-tab per-query dispatch 구현 (Wave 8 β)`

### Milestone 3 — Edge cases & robustness

Goal: handle the harder demos and the corners of the parameter space —
plate-with-hole geometry, EELS over layered structures, dipoles near layer
interfaces, mesh FP drift, and degenerate input validation.

- Plate-with-hole geometry (`polygon3.plate` with `verts2`, `tripolygon` with
  `sym`, `polymesh2d` with holes).
- EELS over layered structures (`demoeelsret7/8`).
- Dipole near-surface layer demos (`demodipret10`, `demospecret13`).
- Mesh FP drift mitigation through `matlab_compat` (Wave 7-49).
- ODE-based Sommerfeld integrator backend with custom `matlab_ode45` step
  controller (Waves 33, 48).
- `intbessel`/`inthankel` MATLAB FP multiplication-order alignment (Wave 49).
- `pinfty` MATLAB-bin reference far-field for ret spectra.
- Input validation across `Particle`, `ComParticle`, `EpsConst`,
  `PlaneWaveStat`, `PlaneWaveRet`, `BEMStat`, `BEMRet` (M3 Wave 2 B3).
- ComPoint nudge on layer interface (`Wave 46`, +1e-8 in z).
- `vertcat` quad-rule inheritance for combined particles (Wave 29 C).
- `EpsFun` `_EV2NM` aligned to MATLAB `Misc/units.m` value (Wave 28 D).
- `BEMRetLayer` MATLAB Engine LU/solve route (opt-in, Waves 51, 66, 67) for
  validation backends.
- Sphere-and-rod numerical cross-checks: Mie, BEMStat, BEMRet, BEMStatLayer,
  BEMRetLayer, mirror, eigenmode, iterative, dipole, dipole-layer, EELS,
  near-field, 7-shape catalog (`validation/01_mie` through
  `validation/13_shapes`, plus `validation/summary`).

Representative commits:

- `e7beedf` `layer_structure: ODE 기반 Sommerfeld 적분 백엔드 추가 (Wave 33, opt-in)`
- `9c45543` `matlab_ode45: MATLAB ode45.m step controller 1:1 재구현 (Wave 48)`
- `08f754a` `intbessel/inthankel: MATLAB FP 곱셈 순서로 정렬 (Wave 49)`
- `8e0329e` `trisphere: 모든 sphere 사이즈에 MATLAB 사전 triangulation 추가 (Wave 62)`
- `4d72e1c` `BEMRetLayer: Wave 67 — MATLAB initmat.m 전체 BEM matrix 재구성 인프라`
- `c55e2a3` `M3 Wave2 B3: spectrum/MeshField/output edge case 테스트 추가`
- `2de8ae0` `validation/summary: 전체 MATLAB vs Python 집계 리포트`

### Milestone 4 — Performance optimisation

Goal: bring the CPU path within MATLAB's runtime envelope and add a GPU
path that scales beyond it.

Tier 1 — scipy LAPACK flags

- `T1` — `lu_factor`/`lu_solve` `check_finite=False`/`overwrite_a=True` across
  all BEM solvers and the H-matrix path. 10-20% LU win.

Tier 2 — Multi-RHS wavelength batching and GMRES

- `R1` — `BEMRet` multi-pol multi-RHS vectorisation.
- `R2` — Hot-loop unnecessary `.copy()` removed on `H1`/`H2`/`H1p`/`H2p`.

Tier 3 — Numba JIT (`MNPBEM_NUMBA=1`, default ON)

- `N1` — Numba JIT `compgreen_stat` G/F/Gp assembly kernel.
- `N2` — Numba JIT `compgreen_ret` distance kernel.
- `N3` — Numba JIT `compgreen_layer` bilinear/trilinear interpolation.
- `N4` — Numba JIT `meshfield` per-wavelength dense Green evaluator.
- `N5` — (subsumed into N1-N4 dispatch helpers).
- `N6` — `closedparticle` `loc` matching vectorised, O(n^2) → O(1) (450×).

Tier 4 — GPU and external solvers

- `G1` — CuPy GPU LU / solve dual-path dispatch (`MNPBEM_GPU=1`,
  `MNPBEM_GPU_THRESHOLD=1500`). 5-14× on RTX A6000.
- `G2` — All BEM solvers + eigenmode path moved to GPU dispatch
  (`BEMStatIter`, `BEMRetIter`, `BEMRetLayerIter`, `BEMStatMirror`,
  `BEMRetMirror`, `BEMStatEig`, `BEMStatEigMirror`).
- `H1` — ACA complex128 Numba + k-aware admissibility +
  `hmatrix=False` opt-out option.
- `F1` — fmm3dpy free-space ret meshfield potential/field acceleration
  (5K × 10K, 5×).
- `C1` — cython_lapack small-matrix bypass (subsumed into Tier 1 dispatch).

Phase 2 — multi-GPU and multi-node (Lanes A-D)

- Lane A — Refined Green function refinement element on CuPy.
- Lane A2 — `BEMRet` matrix assembly on CuPy (eager).
- Lane B — `PlaneWaveRet` / `SpectrumRet` / `EpsTable` field/potential
  GPU dispatch.
- Lane C — Layer Sommerfeld batch + `BEMRetLayer` GEMM GPU dispatch +
  `_intbessel_batch` / `_inthankel_batch` on-device weighted sum.
- Lane D — Multi-GPU wavelength batch dispatch
  (`solve_spectrum_multi_gpu`, subprocess-per-GPU). Extended to multi-node
  via `mpi4py` (`solve_spectrum_mpi`).
- Lane E — H-matrix GPU prototype.

Phase 3 — Native CuPy round-trip

- T1 — `GreenRetRefined` CuPy native return (`MNPBEM_GPU_NATIVE=1`).
- T2 — `BEMRet` end-to-end CuPy native + `Sigma1 = H @ G^-1` direct
  `lu_solve`.
- T3 — `SpectrumRet` GPU path with auto-detected CuPy inputs.

Headline results:

- CPU geometry-build speedup 2.21×.
- GPU geometry-build speedup 3.60×.
- 02 BEMStat sphere: 3.68 s → 1.5 s.
- 03 BEMRet sphere: 42.5 s → 15-20 s.
- 05 BEMRet layer: 71.6 s → 25-30 s.
- 12 ret meshfield: 18.1 s → 3-5 s.

Representative commits:

- `d8d396e` `merge: T1 scipy lu_factor/solve check_finite=False/overwrite flag 적용`
- `7969e02` `merge: N1 Numba JIT compgreen_stat G/F/Gp kernel`
- `b4ca3dd` `merge: N2 Numba JIT compgreen_ret distance kernel`
- `eb0439a` `merge: N3 Numba JIT compgreen_layer bilinear/trilinear interp`
- `e957143` `merge: N4 Numba meshfield + R2 H1p/H2p Gp.copy() NameError 수정`
- `7d0befd` `merge: N6 closedparticle loc 매칭 vectorize (450×)`
- `12415db` `merge: G1 GPU cupy LU/solve dual-path (RTX A6000 5-14×)`
- `30ede18` `merge: G2 모든 BEM solver + eig GPU 확장 (iter/mirror/eig dispatch)`
- `73d98d3` `merge: H1 ACA complex128 numba + k-aware admissibility + hmatrix=False 옵션`
- `a270bdf` `merge: F1 fmm3dpy potential/field 보조 가속 (5K×10K 5×)`
- `942d487` `merge: Lane D multi-GPU wavelength batch`
- `5aa34dc` `merge: Lane A2 BEM matrix assembly cupy-eager`
- `f30d3a7` `multi-node MPI wavelength dispatch (Lane D 확장)`

### Milestone 5 — Final validation

Goal: production-readiness — acceptance criteria, comprehensive regression
suite, documentation, CI/CD, PyPI release. Resolve or document the BEM 1.6%
drift through Lanes A-E investigation.

- `M5-1` — Acceptance criteria fixed in `docs/ACCEPTANCE_CRITERIA.md`
  (accuracy ≥ 55/72 machine-precision, BAD = 0; CPU ≥ 1.5×, GPU ≥ 3×;
  ACA / iter / dense MATLAB-aligned).
- `M5-2` — Comprehensive regression suite under `tests/regression/`
  (72 demos + sphere/rod 51 + dimer + large-mesh edge cases) with
  CI hash comparison.
- `M5-3` — Documentation: `README.md`, `docs/API_REFERENCE.md`,
  `docs/MIGRATION_GUIDE.md`, `docs/ARCHITECTURE.md`, `CHANGELOG.md`,
  `docs/PERFORMANCE.md`.
- `M5-4` — CI/CD: GitHub Actions matrix (Python 3.11/12 × CUDA), PyPI
  publish, dependabot weekly.
- `M5-5` — Release prep: `pyproject.toml`, `__version__ = 1.0.0`,
  `LICENSE`, PyPI dry-run.
- `M5-6` — BEM 1.6% drift decision (Lanes A-E). 9.1e-8 acceptance
  reached after mesh-fix; residual drift documented.

### Fixed (cumulative across M1-M4)

- `_minrectangle` tie-break — MATLAB strict `<` instead of Python `<=`,
  fixing `demodipstat4` (83.07 → 0.297, 280×).
- `mesh2d.fixmesh` + `meshpoly` MATLAB L64/L174 alignment.
- `mesh2d._mydelaunayn` MATLAB qhull options (`Qt Qbb Qc`).
- `mesh2d._mytsearch` `inpolygon`-loop fallback.
- `mesh2d.quadtree` ceil(n/2) start vertex + triangle order +
  `nnode=5` jj off-by-one.
- `_minrectangle` `hull.vertices` (CCW) for MATLAB `convhulln` order.
- `boundarynodes` smoothing alignment.
- `quadtree` triangulation (n2n-based) MATLAB alignment.
- `trisphere` `.mat` → `.bin` conversion (no MATLAB at runtime).
- `trispheresegment` `_surf2patch` winding alignment.
- `clean()` degenerate quad winding.
- `tripolygon` `sym` quarter-mesh option.
- `BEMRetLayer` `greentab` single-z2 interpolation logic
  (27% → 0.01%, 9.5× speedup).
- Mirror BEM solver end-to-end bugs (5 in one commit).
- Closed-surface regularisation in `CompGreenStat` and `CompGreenRet`.
- `EELS` log-branch fix and dedup removal.
- `BEMStatLayer` Fresnel coefficient cross-section formula.
- `BEMRet` extinction area / quadrature rule alignment.
- `PlaneWaveRetLayer` extinction conjugate + scattering `nb.real` removed.
- `dipoleretlayer.farfield` MATLAB algorithm rewrite.
- `DipoleStatLayer` 3D `phip` reshape support.
- `Particle` / `trisphere` shape and positivity validation.
- `BEMStat` / `BEMRet` `particle` None / type validation.
- BEMRetIter / BEMRetLayerIter multi-pol broadcast (M4 fix).
- `compgreen_ret`: `closed_args` defaults to per-particle self-closed.
- `_norm_flat`: `sqrt(dot(.,.,2))` MATLAB alignment.
- `surf2patch` quad/triangle output order MATLAB bit-identical.

### Performance summary

| Demo | MATLAB (s) | Python pre-M4 (s) | Python post-M4 (s) | Speedup |
|---|---|---|---|---|
| 02 BEMStat sphere | ~1.2 | 3.68 | 1.5 | 2.5× |
| 03 BEMRet sphere | ~12 | 42.5 | 15-20 | 2.1× |
| 05 BEMRet layer | ~22 | 71.6 | 25-30 | 2.4× |
| 12 ret meshfield | ~5 | 18.1 | 3-5 | 4× |

CPU geometry-build speedup: **2.21× faster than MATLAB**.
GPU geometry-build speedup: **3.60× faster than MATLAB** on RTX A6000.

See `docs/PERFORMANCE.md` for the full table.

[1.0.0]: https://github.com/Yoo-JK/MNPBEM/releases/tag/v1.0.0
[1.1.0]: https://github.com/Yoo-JK/MNPBEM/releases/tag/v1.1.0
[1.2.0]: https://github.com/Yoo-JK/MNPBEM/releases/tag/v1.2.0
[1.3.0]: https://github.com/Yoo-JK/MNPBEM/releases/tag/v1.3.0
[1.4.0]: https://github.com/Yoo-JK/MNPBEM/releases/tag/v1.4.0
[1.5.0]: https://github.com/Yoo-JK/MNPBEM/releases/tag/v1.5.0
[1.5.1]: https://github.com/Yoo-JK/MNPBEM/releases/tag/v1.5.1
[Unreleased]: https://github.com/Yoo-JK/MNPBEM/compare/v1.5.1...HEAD
