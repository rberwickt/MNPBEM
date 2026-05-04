# MNPBEM Python Port v1.6.0 — Internal Release

**Tag**: v1.6.0
**Date**: 2026-05-02
**Previous**: v1.5.2

## Highlights

- **B-Schur full coverage** — `BEMRetIter + schur=True + hmatrix=True` 60-face nonlocal core-shell GMRES 25분 hang → **6:51 수렴**.
  - 통찰: operator-form Schur 재구현 불필요. lu_dense threshold 500 → 4096 상향으로 해결.
  - SchurIterOperator 에 `eps_form='pointwise'|'operator'` 분기 + 자동 결정.
- **BEMRetLayerIter operator-form fix** — substrate + iter + multi-material drift 해결 (사용자 substrate use case 영향).
  - scalar eps fast path + non-scalar operator form (β v1.5.1 패턴 동일).
- **pymnpbem CLI `--str-conf` + `--sim-conf` + `--verbose`** — mnpbem_simulation MATLAB wrapper 호환. 모든 compute 파라미터 sim_conf 안 컨트롤.
  - 기존 `--config YAML` 도 backward-compat.
- **mesh_density 우선순위** — pymnpbem cube builder 가 `mesh_density` (nm) 를 `n_per_edge` (integer) 보다 우선 사용. core size 기준 변환 (mnpbem_simulation 의미와 일치).

## What's new

- mnpbem: `mnpbem/bem/schur_iter_helpers.py` `eps_form` 분기 + auto threshold 4096
- mnpbem: `mnpbem/bem/bem_ret_iter.py` `schur_eps_form='auto'` 옵션
- mnpbem: `mnpbem/bem/bem_ret_layer_iter.py` `_afun / _init_precond / _mfun` operator-form
- pymnpbem: `pymnpbem_simulation/cli.py` --str-conf/--sim-conf
- pymnpbem: `pymnpbem_simulation/structures/advanced_monomer_cube.py::_resolve_n_per_edge` mesh_density 우선
- 신규 tests: `test_b_schur.py`, `test_iter_convergence_layer.py`, `test_mesh_density_priority.py`, `test_cli_str_sim.py`

## Backward compatibility

100% backward compatible. 모든 신규 옵션 default OFF 또는 'auto'.

## Performance

- 60-face nonlocal+schur+iter+hmat: 25 min hang → **6:51 PASS** (410.7s)
- 사용자 use case (Au@Ag dimer 12672 face) 정식 통과: VRAM share 4 GPU 권장 (v1.5.2 부터)

## Known limits / Follow-up

- **BEM assembly perf bottleneck** (C agent 발견): `solve_spectrum_multi_gpu` 의 BEM matrix assembly 가 single-thread CPU bound. GPU 활용 0%. v1.6.x 후속 prof + numba JIT 검토 필요.
- **compgreen_ret_layer multi-particle layer indexing** (B agent 발견): `compgreen_ret_layer.py:651` shape mismatch (Au@Ag core-shell on substrate). v1.6.x 후속.
- Tier-3 timing benchmark 측정 미완 (assembly bottleneck + concurrent CPU 경합) — batch job 으로 분리 재시도 권장.

## Migration

기존 v1.5.2 코드 변경 X. 신규 옵션은 모두 'auto' default. `--str-conf/--sim-conf` 새 CLI 도 기존 `--config YAML` 과 병행 사용 가능.

## Citing

본 release 인용 시 v1.0.0 인용 형식과 동일하게 처리한다 (저장소 저자, MNPBEM Python port 1.6.0, 2026, internal).

## git tag command

```bash
git tag -a v1.6.0 -F docs/RELEASE_NOTES_v1.6.0.md
git push origin v1.6.0
```
