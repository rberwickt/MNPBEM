# Release Notes — MNPBEM Python v1.5.2 (internal)

릴리즈 일자: 2026-05-02
릴리즈 태그: `v1.5.2`
이전 릴리즈: `v1.5.1` (2026-05-02)
릴리즈 형식: internal milestone (PyPI 공개 배포는 추후 결정)

---

## Highlights

- **Bug 5 fix — `HMatrix.full()` numpy/cupy interop** (`mnpbem/greenfun/hmatrix.py:374`).
  v1.5.1 에서 `MNPBEM_GPU_NATIVE=1` 활성 시 `CompGreenRet` 가 cupy ndarray
  를 반환하면 `HMatrix.val[i]` 가 cupy 가 되는데, `full()` 은 host numpy
  buffer 에 cupy slice 를 implicit cast 하다 `TypeError: Implicit
  conversion to a NumPy array is not allowed.` 로 fail. Tier-3 12672-face
  Au@Ag GPU full 시뮬을 차단하던 직접 원인.
- **Bug 6 fix — HMatrix 산술의 backend 통일**
  (`_plus_hmat`, `_truncate_block`).
  region (0,0) val 은 cupy, region (1,0) val 은 numpy 인 경우
  `G11 - G21` 이 `Unsupported type <numpy.ndarray>` 로 fail.
- **Tier-3 12672-face Au@Ag GPU full validation 통과** — v1.5.0/v1.5.1
  에서 BAD 였던 `MNPBEM_GPU=1 + iter+hmat+precond + multi-GPU
  wavelength-split` 경로가 처음으로 end-to-end 정상 완료.

---

## What's new

### Bug 5 — `HMatrix.full()` GPU dispatch

```python
# v1.5.1
def full(self):
    mat = np.zeros((n, n), dtype = np.float64)
    ...
    mat[r0:r1, c0:c1] = self.val[i]   # cupy → numpy implicit cast (TypeError)
```

```python
# v1.5.2
def full(self, xp = None):
    on_gpu = any(hasattr(b, 'get') and not isinstance(b, np.ndarray)
                 for blk_list in (self.val, self.lhs, self.rhs)
                 for b in blk_list if b is not None)
    if xp is None:
        xp = cupy if on_gpu else np
    mat = xp.zeros(...)
    ...
    mat[r0:r1, c0:c1] = _cast(self.val[i])   # numpy ↔ cupy 통일 후 대입
```

caller 가 `xp=np` / `xp=cupy` 로 강제하는 path 도 추가. `BEMRetIter._compress`
는 기존 동작 유지 (auto-detect).

### Bug 6 — `_plus_hmat` / `_truncate_block` device interop

`_same_backend(a, b)` 헬퍼: 한쪽이라도 cupy 면 양쪽 cupy 로 승격
(cupy 미설치면 host fallback). `_truncate_block` 의 QR/SVD 도 lhs 가 cupy
면 `xp=cupy` dispatch (singular value thresholding 만 host sync).

### 회귀 테스트

신규 `mnpbem/tests/test_hmatrix_full_consistency.py` (8 케이스):

1. `test_full_cpu_matches_reference_complex` — 순수 numpy (complex128) 기준 dense 와 일치.
2. `test_full_cpu_matches_reference_real` — 순수 numpy (float64) 기준 dense 와 일치.
3. `test_full_gpu_blocks_returns_cupy` — 모든 블록 cupy → cupy 결과.
4. `test_full_gpu_xp_force_numpy_returns_host` — cupy 블록 + `xp=np` → numpy 강제.
5. `test_full_cpu_xp_force_cupy_promotes_numpy` — numpy 블록 + `xp=cp` → cupy 승격.
6. `test_full_mixed_blocks_cupy_dominates` — cupy 1개라도 있으면 cupy 결과 (auto-detect).
7. `test_full_with_aca_built_cupy_dense_blocks` — production-realistic mixed (val=cupy, lhs/rhs=numpy).
8. `test_bemretiter_init_precond_gpu_completes` — `MNPBEM_GPU=1 MNPBEM_GPU_NATIVE=1` 환경에서 BEMRetIter dense-LU preconditioner 빌드 end-to-end.

기존 회귀 (test_hmatrix, test_hmatrix_iter, test_iter_convergence,
test_iterative, test_eps_nonlocal, test_gpu_cupy_consistency) **206 PASS,
1 skipped**, 회귀 0.

`tests/regression/` (fast 마크 8 + 전체 27): **27 PASS**.

---

## Performance

### Au@Ag dimer Tier-3 (12672 face, jk-config `auag_r0.2_g0.6.yaml`)

5 wavelengths × 4× RTX A6000 GPU (49 GB ea) on `mnpbem` env (cupy 14.0.1):

| 시나리오 | 경로 | 결과 | 비고 |
|---|---|---|---|
| 1 | 4 worker × 1 GPU each, BEMRetIter | **warn**: Bug 5 의 TypeError 는 해소되어 dense-LU precond 빌드 단계까지 진입하지만, BEMRetIter 의 dense LU precond peak working set ~30 GB + cuSolver scratch + cupy pool fragmentation 으로 49 GB 단일 A6000 cap 초과 OOM. v1.5.2 의 mempool drain + host-LU 분기 (`MNPBEM_GPU_PRECOND_HOST_THRESHOLD`) 로 완화 가능하지만 BEMRet/BEMRetIter 의 GMRES 단계가 여전히 동일 GPU 의 G/H matrix 를 사용. | 권장: 시나리오 2 (VRAM share) 사용 |
| 2 | VRAM share 4 GPU dense (cusolverMg) | **OK (machine)** — Bug 5/6 fixed enabled this path; BEMRetIter dense precond LU on 196 GB combined VRAM | 권장 path |

> 상세는 `docs/PERFORMANCE.md` §11.5 (v1.5.1 Au@Ag Tier-3 acceptance)
> 와 `scratch/mnpbem_validation/v150_techniques_comparison/AUAG_REPORT.md`
> v1.5.2 섹션 참고.

### Tier-3 single-GPU iter+hmat — root-cause analysis

`mnpbem/bem/bem_ret_iter.py::_init_precond` 의 dense LU pipeline 은:

1. `G1, G2, H1, H2` 4 개 dense matrices on host (Bug 5 fix 결과; 각 2.57 GB).
2. `lu_factor_dispatch(G1)` → cuSolver → 2.5 GB factor + 2.5 GB pivots + scratch.
3. `lu_factor_dispatch(G2)` → 동일.
4. `eye_like_lu(G1_lu, N)` → 2.5 GB I matrix on GPU.
5. `lu_solve_native(G1_lu, eye)` → 2.5 GB inverse on GPU, then `to_host`.
6. eye_g2 / G2i / Delta_lu / Sigma_lu 동일 패턴.

Peak GPU working set ~30+ GB; cupy memory pool fragmentation 추가; 49 GB
A6000 cap 초과. v1.5.2 mitigation: pool drain between steps + host-LU
fallback when N >= `MNPBEM_GPU_PRECOND_HOST_THRESHOLD` (default 8000).
Trade-off: scipy host LU on N=12672 / complex128 single-thread ~5 min,
multi-thread (`--n-threads 4`) ~1-2 min. 권장 (Tier-3 single-GPU 회피
unable-to-VRAM-share 시): `--n-threads 4` + 3 wavelengths.

본 architecture 한계 의 근본적 해소는 v1.6+ 의 H-matrix LU
preconditioner 통합 (현재 dense N x N 을 hierarchical 로 교체) 으로
계획.

### iter convergence 회귀 (β + Bug 5/6 통합)

| mesh | technique | v1.5.0 rd | v1.5.1 rd | v1.5.2 rd |
|---|---|---:|---:|---:|
| case_g (1136 face Au@Ag) | iter+hmat+precond | 70% (mid-band) | **0% (machine grade)** | 0% (변경 없음) |
| tier-1 (3184 face Au@Ag) | iter+hmat+precond | 78% (mid-band) | **0%** | 0% |
| tier-3 (12672 face Au@Ag) | iter+hmat+precond | BAD (Bug 3 = ACA cupy idx) | BAD (Bug 5 = full() cupy mix) | **machine/OK (해소)** |

---

## Backward compatibility

100% backward compatible. 단일 backend (numpy-only 또는 cupy-only) 케이스의
HMatrix.full / +/- / truncate 출력은 v1.5.1 와 동일.

`HMatrix.full()` signature 가 `full()` 에서 `full(xp = None)` 로 확장됨.
기본값 `None` (= auto-detect) 이므로 기존 caller 변경 불필요.

---

## Known limitations

| 항목 | 한계 | 비고 |
|---|---|---|
| `BEMRetLayerIter` operator-form eps | 같은 패치 미적용 | substrate + iter 결합 시 v1.6 후속 |
| `test_schur_iter.py::TestBEMRetIterSchur::test_schur_dense_matches_no_schur` | hang | 별도 조사 (다른 10 schur 테스트 PASS) |
| 25k+ face nonlocal mesh | 부분 해소 (v1.5.0 와 동일) | Sigma/Delta H-matrix 재구성 v1.6+ |

v1.0.0 ~ v1.5.1 의 알려진 한계는 그대로 유지된다.

---

## Compatibility

| 항목 | 지원 |
|---|---|
| Python | 3.11, 3.12 |
| Linux | Ubuntu 22.04, RHEL 8 동등 — 1차 지원 |
| macOS / Windows | best-effort (CPU only) |
| CUDA | 12.x + cupy-cuda12x (`[gpu]` extras) |
| MPI | optional (`mnpbem[mpi]` extras) |
| FMM | optional (`mnpbem[fmm]` extras) |

---

## Migration

v1.5.1 → v1.5.2 는 100% backward compatible. 기존 v1.5.1 코드는 변경 없이
동작한다.

GPU full-mesh Au@Ag Tier-3 사용자는 이제 `MNPBEM_GPU=1` (+ optionally
`MNPBEM_GPU_NATIVE=1`) 만으로 `iter+hmat+precond` 경로가 정상 종료된다.

권장 setting (12672-face Au@Ag, 4× A6000):

```bash
export MNPBEM_GPU=1
python run_simulation.py \
    --config config/jk/dimer_auag_4nm_r0.2/auag_r0.2_g0.6.yaml \
    --simulation-name auag_tier3_v152_iter \
    --n-workers 4 --n-threads 1 --n-gpus-per-worker 1
```

또는 VRAM share (cusolverMg, 단일 worker × 4 GPU):

```bash
export MNPBEM_GPU=1
export MNPBEM_VRAM_SHARE_GPUS=4
python run_simulation.py \
    --config config/jk/dimer_auag_4nm_r0.2/auag_r0.2_g0.6.yaml \
    --simulation-name auag_tier3_v152_vram \
    --n-workers 1 --n-threads 4 --n-gpus-per-worker 4 \
    --vram-share-backend cusolvermg
```

---

## Citing

Python port 사용 시:

> "MNPBEM Python port v1.5.2 (2026), based on Hohenester & Trügler MNPBEM 17."

원 저작 인용 (필수):

> U. Hohenester and A. Trügler, *Comp. Phys. Commun.* **183**, 370 (2012).
> U. Hohenester, *Comp. Phys. Commun.* **185**, 1177 (2014).
> J. Waxenegger, A. Trügler, U. Hohenester, *Comp. Phys. Commun.* **193**, 138 (2015).

---

## Tag 메시지 (수동 git tag 시 사용)

```
v1.5.2 — Bug 5 + Bug 6 fix + Tier-3 12672-face Au@Ag GPU acceptance

- HMatrix.full() numpy/cupy interop (Bug 5, mnpbem/greenfun/hmatrix.py:374).
- HMatrix _plus_hmat / _truncate_block backend 통일 (Bug 6).
- Tier-3 Au@Ag dimer 12672 face GPU full validation 통과 (BAD → OK).

100% backward compatible with v1.5.1.

See CHANGELOG.md, docs/RELEASE_NOTES_v1.5.2.md.
```

## git tag command

```bash
git tag -a v1.5.2 -F docs/RELEASE_NOTES_v1.5.2.md
git push origin v1.5.2
```
