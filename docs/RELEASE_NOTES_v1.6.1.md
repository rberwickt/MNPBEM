# MNPBEM Python Port v1.6.1 — Internal Release

**Tag**: v1.6.1
**Date**: 2026-05-02
**Previous**: v1.6.0

## Highlights

- **BEM assembly Numba JIT 가속** (Agent A, `mnpbem/greenfun/_numba_refine.py`)
  - Per-face Python loops (refine_diagonal, refine_offdiagonal, _refine_greenstat) → Numba @njit kernels
  - 3k+ face mesh assembly ~70% time saving (estimate, profile-driven)
  - bit-identical numerical contract (fastmath=False)
  - 자동 활성 (numba 가용 시), `MNPBEM_NUMBA_REFINE=0` 으로 비활성 가능
- **compgreen_ret_layer multi-particle indexing fix** (Agent B)
  - Au@Ag core-shell on substrate shape mismatch (116×116 vs 232×232) → `np.ix_(ind1, ind2)` sub-block 추출
  - 사용자 substrate 위 core-shell dimer 시뮬 정상 동작
  - 4 신규 test PASS

## What's new

- mnpbem: `mnpbem/greenfun/_numba_refine.py` — Numba @njit kernels (refine_diagonal / refine_offdiagonal / _refine_greenstat)
- mnpbem: `mnpbem/greenfun/greenret_refined.py` — Numba kernel 디스패치 통합
- mnpbem: `mnpbem/greenfun/compgreen_ret_layer.py:651` `np.ix_(ind1, ind2)` sub-block 추출
- 신규 tests: `mnpbem/tests/test_compgreen_ret_layer_multi.py` (4 테스트)

## Backward compatibility

100% backward compatible. Numba JIT 자동 활성, 결과 bit-identical.

## Performance

- BEM assembly (3k+ face): ~70% 절감 (Numba 적용)
- Tier-3 12672-face timing 정식 측정은 v1.6.x 후속 batch (concurrent CPU 경합 없는 상태에서)

## Known limits / Follow-up

- VRAM share env vars wiring 미완 (C agent 발견): `MNPBEM_VRAM_SHARE_*` set 되지만 `lu_factor_dispatch` 가 `n_gpus=N` kwarg 명시 전달 X → cusolverMg 활성화 실패. v1.6.x 후속 wiring fix 필요.
- Tier-3 timing benchmark 정식 측정 미완 (concurrent processes + assembly bottleneck 영향). A 의 Numba fix 적용 후 batch 재시도 필요.

## Migration

기존 v1.6.0 코드 변경 X.

## Citing

본 release 인용 시 v1.0.0 인용 형식과 동일하게 처리한다 (저장소 저자, MNPBEM Python port 1.6.1, 2026, internal).

## git tag command

```bash
git tag -a v1.6.1 -F docs/RELEASE_NOTES_v1.6.1.md
git push origin v1.6.1
```
