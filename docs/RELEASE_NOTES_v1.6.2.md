# MNPBEM Python Port v1.6.2 — Internal Release

**Tag**: v1.6.2
**Date**: 2026-05-02
**Previous**: v1.6.1

## Highlights

- **VRAM share env vars wiring fix** — `MNPBEM_VRAM_SHARE_GPUS` /
  `MNPBEM_VRAM_SHARE_BACKEND` / `MNPBEM_VRAM_SHARE_DEVICE_IDS` 가 이제
  `lu_factor_dispatch` / `solve_dispatch` 에서 자동 인식됩니다.
  v1.6.0/v1.6.1 Tier-3 12672-face benchmark 시 single-GPU OOM 발생
  원인이 해소되었습니다.

## What's new

- `mnpbem/utils/gpu.py::_vram_share_env_defaults()` 헬퍼
- `lu_factor_dispatch(A)` / `solve_dispatch(A, b)` — `n_gpus` 미지정 시
  env var 자동 인식. 명시 kwarg 가 항상 우선
- `MNPBEM_VRAM_SHARE_DEVICE_IDS=0,1,2,3` 신규 지원
- `MNPBEM_VRAM_SHARE=0` master switch 로 강제 비활성
- `mnpbem/tests/test_vram_share_env_wiring.py` (15 tests)

## Root cause

C agent 발견:
- `MNPBEM_VRAM_SHARE_GPUS=4` 등 env vars 가 set 되었으나
- `bem_ret_iter.py:363`, `bem_stat_layer.py:66`, `bem_ret_layer.py:253`
  등에서 `lu_factor_dispatch(A)` 만 호출 (kwarg 미전달)
- 기존 `lu_factor_dispatch` 가 `n_gpus = int(kwargs.pop('n_gpus', 1))`
  로 default 1 → cuSolverMg 분기 fall-through
- 12672-face dense LU (49 GB) 가 single GPU 에 fit 안 되어 OOM

fix 방향: 옵션 A (`gpu.py` 자체에서 env var 자동 인식). BEM solver
변경 최소화. `bem_ret.py` 의 기존 `_vram_share_lu_kwargs` 명시 전달
경로는 kwarg 우선이므로 동작 변경 없음.

## Verified

- 4× RTX A6000 host 에서 `MNPBEM_VRAM_SHARE_GPUS=2` 만 set 후
  `lu_factor_dispatch(np.random.randn(256,256) + 1j*...)` →
  pkg tag `'mgpu'` 진입, `lu_solve_dispatch` residual 9.3e-16
- `MNPBEM_VRAM_SHARE=0` master switch 시 mgpu 분기 비활성, CPU 분기 진입
- `MNPBEM_VRAM_SHARE_GPUS=1` 또는 invalid value 시 안전하게 단일-GPU
  분기로 fall-through

## Tests

15 신규 테스트 (`mnpbem/tests/test_vram_share_env_wiring.py`):
- env var 헬퍼 5건 (unset / n_only / full set / master off / n=1 / invalid)
- `lu_factor_dispatch` 라우팅 6건 (no-env / env-only / kwarg overrides /
  kwarg n=1 forces off / master off)
- `lu_solve_dispatch` end-to-end 1건
- `solve_dispatch` env routing 2건

기존 `test_gpu_cupy_consistency.py` 14건 regression 없음 (9 PASS,
5 SKIP — cupy device path 는 GPU 환경에서만 실행).

## Backward compatibility

100% backward compatible:
- 기존 명시 `n_gpus` kwarg 호출 (예: `bem_ret.py::_vram_share_lu_kwargs`)
  이 항상 env 보다 우선
- env var 미설정 시 default `n_gpus=1` 유지 (single-GPU/CPU 분기)
- `MNPBEM_VRAM_SHARE=0` 옵션으로 사용자가 wiring 자체를 무력화 가능

## Known limits / Follow-up

- Tier-3 12672-face cuSolverMg 정식 batch benchmark 권장 (RTX A6000
  4× pooled VRAM 196 GB 활용 가능 검증). M5+ 또는 별도 lane 작업.
- `bem_ret_iter.py` 의 추가 LU 호출 지점 (`Delta_lu`, `Sigma_lu` 등) 도
  이제 동일 env var 자동 인식 — 명시 wiring 추가 작업 불필요.

## Migration

기존 v1.6.1 코드 변경 X. env var 만 export 하면 동작.

```bash
export MNPBEM_GPU=1
export MNPBEM_VRAM_SHARE_GPUS=4
export MNPBEM_VRAM_SHARE_BACKEND=cusolvermg
# (옵션) export MNPBEM_VRAM_SHARE_DEVICE_IDS=0,1,2,3
python my_simulation.py
```

## Citing

본 release 인용 시 v1.0.0 인용 형식과 동일하게 처리한다 (저장소 저자,
MNPBEM Python port 1.6.2, 2026, internal).

## git tag command

```bash
git tag -a v1.6.2 -F docs/RELEASE_NOTES_v1.6.2.md
git push origin v1.6.2
```
