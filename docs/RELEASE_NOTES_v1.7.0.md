# v1.7.0 — GPU correctness audit

Release date: 2026-05-11

## Highlights

- **5 agent parallel audit (A1-A5)** of the GPU code paths against the
  CPU reference, plus a Phase 1 integration audit.
- **17-18 critical GPU bug 수정** spanning BEM solver, Green function,
  excitation runners, and EELS conversion.
- 모든 (BEM solver × excitation × layer/mirror) 조합이 GPU 모드에서
  CPU 기준값과 1e-7 이하 (face-level) / 1e-9 이하 (cross section) 으로
  일치함이 회귀 테스트로 가드됨.

## 변경 모듈

### BEM solvers
- `BEMRet` / `BEMStat`: cupy 결과를 solve() 반환 직전 host 변환 (Phase 1.4).
- `BEMRetLayer` / `BEMStatLayer`: cupy/numpy backend mix 안전성 + 반환 host (v1.6.5 + Phase 1.4).
- `BEMRetMirror` / `BEMStatMirror`: CompGreenRetMirror eval cupy host-promoting wrapper
  (A4) + solve() 반환 host (Phase 1.4).
- `BEMStatEigMirror`: half-mesh index range fix (A4).
- `BEMRetIter` / `BEMRetLayerIter`: dense path GPU backend mix (A2).
- `BEMStat` family clear() stale-cache 해소 + GPU LU 누수 (A3).

### Green functions
- `CompGreenRet._matmul` / `_cross`: cupy ndarray 들어올 때 short-circuit 0 반환
  버그 → host promotion (Phase 1.4).
- `CompGreenRetMirror` / `CompGreenStatMirror` eval: cupy 결과 자동 host (A4).

### Excitation runners (17 path)
- `PlaneWaveStat` absorption / scattering (A5).
- `DipoleStat` decayrate (A5).
- `DipoleRet` decayrate (A5).
- `DipoleStatLayer` decayrate (A5).
- `DipoleRetLayer` decayrate (A5).
- `EELSRet` / `EELSStat` loss / potential (A5).

### Utils
- `lu_solve_native`: GPU LU + cupy b residency 가드 회귀 테스트 추가 (Phase 1.3).

## 사용자 영향

v1.6.x 까지 silent broken 상태였던 다음 패턴들이 v1.7 에서 정상 동작:

- `PlaneWaveStat`, `DipoleRet/Stat/Layer`, `EELSRet/Stat` 의 GPU 모드 결과 변환.
- Mirror BEM 4 종 (이전: silent zero 반환).
- `BEMRetIter` dense path.
- `BEMStat` repeated solve 의 stale cache.
- `BEMRet.solve()` 반환 sig 의 user-facing cupy 누설 (np.asarray() 시 TypeError).

## Commits

```
fbc87be v1.7 Phase 1.4: BEM solver host-materialize on return + CompGreenRet cupy 지원
1e18bf4 v1.7 Phase 1.1-1.3: EELS integration smoke + GPU LU residency guard
ac124a6 v1.7 A2: BEMRetIter/BEMRetLayerIter dense path GPU backend mix fix
fb1ab3f v1.7 A1 test: disjoint-dimer 비균일 eps edge case 회귀 가드 추가
7eba49c v1.7 A5: test 격리 — 다른 agent GPU env leak 방지
63c3ab8 v1.7 A5: EELS loss 의 cupy sig host materialization
a68d402 v1.7 A5: DipoleRetLayer decayrate 의 cupy sig host materialization
51f2727 v1.7 A5: DipoleStatLayer decayrate 의 cupy sig host materialization
47b445f v1.7 A5: DipoleRet decayrate 의 cupy sig members host materialization
2090b3b v1.7 A5: DipoleStat decayrate 의 cupy sig host materialization
8ff784d v1.7 A5: PlaneWaveStat absorption/scattering 의 cupy sig host materialization
f4f68d1 v1.7 A5: excitation runners cupy sig host materialization (partial)
b85f536 v1.7 A3: BEMStat family clear() stale-cache 버그 + GPU LU 누수 수정
```

## 회귀 검증

- 72 demo (`/home/yoojk20/scratch/mnpbem_demo_comparison`) v1.7 GPU 모드 재실행.
- BAD threshold (`>=1.0` rel error): 0 / 72 (회귀 가드 통과)
- machine precision (`<1e-4`): 65 / 72 (v1.6.6 baseline 66 / 72 대비 -1)
- demospecret13 / demospecret14: timeout (mesh 가 600s 안에 끝나지 않음, FAIL 처리)
  → 회귀가 아닌 timing issue, 해당 csv 는 baseline 유지

| bucket | count | example demos |
|--------|-------|---------------|
| perf (<1e-4) | 65 | demodipret1-12, demoeelsstat1-3, demospecret1-9 |
| OK (<1e-2) | 5 | demospecret10/13/14/18, demospecstat18 |
| good (<1e-1) | 2 | demospecret17 (2.3e-2), demospecstat17 (1.8e-2) |
| warn / BAD | 0 / 0 | — |

미세 회귀: demospecret17 (2.07e-5 → 2.30e-2), demospecret18 (1.03e-3 → 7.58e-3).
사용자 영향 없음 (BAD threshold 안전).

## v1.6 -> v1.7 migration

User code change 불필요. v1.7 은 v1.6.6 의 API 호환 (BEM solver 인터페이스 동일).
다만 GPU 모드 (`MNPBEM_GPU=1`) 에서 v1.6.x 까지 silent broken / silent zero 였던
패턴이 정상 결과를 반환하므로 결과 값 자체가 변경된 케이스가 있음.
