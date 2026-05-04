# Migration from MATLAB MNPBEM to Python

This guide is for users with an existing MATLAB MNPBEM17 script who want
to port it to the Python toolbox. Calling sequences and class names are
intentionally preserved, so the translation is mostly mechanical. Below
are the most common patterns plus a list of pitfalls that catch new
users.

## Quick mapping table

### Materials

| MATLAB | Python |
|---|---|
| `epsconst(1)` | `from mnpbem.materials import EpsConst; EpsConst(1.0)` |
| `epstable('gold.dat')` | `from mnpbem.materials import EpsTable; EpsTable('gold.dat')` |
| `epsdrude('Au')` | `from mnpbem.materials import EpsDrude; EpsDrude(eps0, wp, gammad)` |
| `[eps,k]=epstab(enei)` | `eps, k = epstab(enei)` |

### Geometry

| MATLAB | Python |
|---|---|
| `p = trisphere(144, 20)` | `from mnpbem.geometry import trisphere; p = trisphere(144, 20)` |
| `p = trirod(20, 50)` | `from mnpbem.geometry import trirod; p = trirod(20, 50)` |
| `p = tricube(11, 20)` | `from mnpbem.geometry import tricube; p = tricube(11, 20)` |
| `comparticle({p}, {eps1,eps2}, [2,1], 1, op)` | `ComParticle([eps1, eps2], [p], [[2, 1]], 1, **op)` |
| `polygon(8, 'size', [10 20])` | `Polygon(8, mode='size', size=[10, 20])` |
| `polygon3(poly, 5, edge)` | `Polygon3(poly, 5, edge)` |
| `edgeprofile(0.4)` | `EdgeProfile(e=0.4)` |
| `tripolygon(poly3, edge)` | `tripolygon(poly3, edge)` |
| `layerstructure(epstab, [1,2], 0)` | `LayerStructure(epstab, [1, 2], [0.0])` |

### BEM solvers

| MATLAB | Python |
|---|---|
| `bem = bemstat(p)` | `from mnpbem.bem import BEMStat; bem = BEMStat(p)` |
| `bem = bemret(p)` | `from mnpbem.bem import BEMRet; bem = BEMRet(p)` |
| `bem = bemstatlayer(p, layer)` | `from mnpbem.bem import BEMStatLayer; bem = BEMStatLayer(p, layer)` |
| `bem = bemretlayer(p, layer)` | `from mnpbem.bem import BEMRetLayer; bem = BEMRetLayer(p, layer)` |
| `bem = bemretiter(p, op)` | `from mnpbem.bem import BEMRetIter; bem = BEMRetIter(p, **op)` |
| `bem = bem.init(enei)` | `bem = bem.init(enei)`  *(usually implicit in `.solve`)* |
| `sig = bem \ exc` | `sig, bem = bem.solve(exc)` |
| `sig = bem \ exc` (mirror) | `sig, bem = bem.solve(exc)`  *(same call)* |
| `clear(bem)` | `bem.clear()` |

### Excitations

| MATLAB | Python |
|---|---|
| `exc = planewave([1 0 0])` (stat) | `from mnpbem.simulation import PlaneWaveStat; exc = PlaneWaveStat([[1, 0, 0]])` |
| `exc = planewaveret(...)` | shorthand: `from mnpbem.simulation import planewave; exc = planewave(pol, dir, op)` |
| `exc = planewave([1 0 0], [0 0 1])` (ret) | `PlaneWaveRet([[1,0,0]], [[0,0,1]])` |
| `exc = dipole(pt)` | `from mnpbem.simulation import DipoleRet, DipoleStat; DipoleRet(pt)` (or `DipoleStat`) |
| `exc = electronbeam(p, impact, w, vel)` | `from mnpbem.simulation import EELSRet; EELSRet(p, impact, w, vel)` |

### Calling convention

| MATLAB pattern | Python pattern |
|---|---|
| `exc_struct = exc(p, enei)` | `pot = exc.potential(p, enei)` |
| `sig = bem \ exc_struct` | `sig, bem = bem.solve(pot)` |
| `[ext, dipole] = exc.extinction(sig)` | `ext = exc.extinction(sig)` |
| `[sca, dsig] = exc.scattering(sig)` | `sca, dsig = exc.scattering(sig)` |
| `abs = exc.absorption(sig)` | `abs_ = exc.absorption(sig)` |

### Far-field / spectrum

| MATLAB | Python |
|---|---|
| `pinfty = trisphere(256, 2)` | `pinfty = trisphere(256, 2)` |
| `spec = spectrum(pinfty, op)` | `from mnpbem.spectrum import SpectrumRet; spec = SpectrumRet(pinfty)` |
| `[sca, dsig] = spec.scattering(sig)` | `sca, dsig = spec.scattering(sig)` |
| `field = spec.farfield(sig)` | `field = spec.farfield(sig)` |

### Mie reference

| MATLAB | Python |
|---|---|
| `mie = miesolver(epsin, epsout, d, op)` | `from mnpbem.mie import mie_solver; mie = mie_solver(epsin, epsout, d, sim='ret')` |
| `[sca, ext, abs] = mie.cross(enei)` | `sca = mie.scattering(enei); ext = mie.extinction(enei); abs_ = ext - sca` |

### Iterative solver / ACA

| MATLAB | Python |
|---|---|
| `op = bemoptions('sim','ret','interp','curv','RelCutoff',2)` | `op = dict(sim='ret', interp='curv', RelCutoff=2)` |
| `op.iter = struct('tol',1e-6,'restart',30)` | pass `iter={'tol':1e-6,'restart':30}` to solver |
| `op.aca = struct('htol',1e-6,'kmax',100)` | pass `aca={'htol':1e-6,'kmax':100}` to solver |

### Nonlocal hydrodynamic Drude

| MATLAB | Python |
|---|---|
| `epsfun(@(enei) 1./(1./(eps_b - eps_m) - q_L * delta_d))` | `EpsNonlocal(eps_m, eps_b, delta_d=δ, beta=β)` |
| (manual `coverlayer.shift`) | `coverlayer.shift(p_core, delta_d)` (동일) |
| `coverlayer.refine(p, [[1, 2]])` | `coverlayer.refine(p, [[1, 2]])` (동일) |
| (ad-hoc 3-particle epstab construction) | `make_nonlocal_pair('gold', eps_embed, delta_d, beta)` 가 자동 `(core, shell)` 튜플 반환 |
| `bemstat(p, op{:}, 'refun', refun)` | `BEMStat(p, refun=refun)` |
| `bemret(p, op{:}, 'refun', refun)` | `BEMRet(p, refun=refun)` (v1.1.0 신규) |

---

## Side-by-side worked example

### MATLAB (sphere extinction, retarded)

```matlab
op       = bemoptions('sim','ret','interp','curv');
epstab   = {epsconst(1), epstable('gold.dat')};
p        = comparticle(epstab, {trisphere(144,20)}, [2,1], 1, op);
bem      = bemret(p);
exc      = planewave([1 0 0], [0 0 1], op);

enei     = linspace(400,800,41);
ext      = zeros(size(enei));
for i = 1:length(enei)
    sig    = bem \ exc(p, enei(i));
    ext(i) = exc.extinction(sig);
end
```

### Python equivalent

```python
import numpy as np
from mnpbem.materials  import EpsConst, EpsTable
from mnpbem.geometry   import trisphere, ComParticle
from mnpbem.bem        import BEMRet
from mnpbem.simulation import PlaneWaveRet

epstab = [EpsConst(1.0), EpsTable("gold.dat")]
p      = ComParticle(epstab, [trisphere(144, 20)], [[2, 1]], 1, interp="curv")
bem    = BEMRet(p)
exc    = PlaneWaveRet(np.array([[1, 0, 0]]), np.array([[0, 0, 1]]))

enei   = np.linspace(400, 800, 41)
ext    = np.zeros_like(enei)
for i, e in enumerate(enei):
    sig, bem = bem.solve(exc.potential(p, e))
    ext[i]   = float(np.real(np.ravel(exc.extinction(sig))[0]))
```

---

## Common pitfalls

### 1. `comparticle` outside/inside convention

MATLAB:

```matlab
comparticle({p}, {eps_out, eps_in}, [2, 1; 1, 2], 1, op)
```

is read as: face `i` has dielectric `eps(2)` on the *outside* and
`eps(1)` on the *inside* in the first column-pair, and the reverse for
the second. The trailing `1` is the index of which particle is
**closed** (a watertight surface).

Python keeps the same convention:

```python
ComParticle(eps, [p], [[2, 1], [1, 2]], 1)        # closed = 1 (1-based, like MATLAB)
ComParticle(eps, [p], [[2, 1]], 1)                # only one face / one row
```

If `closed_args` is omitted, no surface is treated as closed (which
breaks the static problem). When in doubt, pass `closed=[1]` (or
`closed=[1, 2]` for two surfaces).

### 2. 1-based vs 0-based indexing

- **`inout` entries**: 1-based, like MATLAB. `[[2, 1]]` means
  `eps[1]` outside, `eps[0]` inside (Python lists).
- **`closed` argument**: 1-based, like MATLAB.
- **`faces`** array: stored 0-based internally, like every NumPy array.
  When you write `p.faces` you get 0-based indices into `p.verts`.
  MATLAB's `p.faces` is 1-based — adjust by `-1` if you compare both.

### 3. Polarization / direction shape

MATLAB lets you write `planewave([1 0 0], [0 0 1])` — a 1×3 row vector.
NumPy keeps the same shape but typed:

```python
PlaneWaveRet(np.array([[1.0, 0.0, 0.0]]),         # shape (1, 3)
             np.array([[0.0, 0.0, 1.0]]))
```

A 1-D `(3,)` array is also accepted, but the `(1, 3)` form makes it
explicit that you are passing a single polarization (vs. a batch
of `(N, 3)`).

### 4. Retarded vs static

The MATLAB option `sim` does not exist in Python; you choose by
importing the right class:

| MATLAB `op.sim` | Python class |
|---|---|
| `'stat'` | `BEMStat`, `PlaneWaveStat`, `DipoleStat`, `EELSStat`, `SpectrumStat` |
| `'ret'`  | `BEMRet`,  `PlaneWaveRet`,  `DipoleRet`,  `EELSRet`,  `SpectrumRet`  |

The convenience factories `planewave(...)`, `dipole(...)`,
`electronbeam(...)`, `spectrum(...)` accept `op={'sim':'ret'}` and
return the right class.

### 5. ODE / quadrature tolerance

MATLAB's `bemoptions` uses `'AbsCutoff'`, `'RelCutoff'`, `'refine'`.
The Python solver accepts the same names as keyword arguments:

```python
ComParticle(eps, [p], [[2, 1]], 1, interp="curv", AbsCutoff=1e-3, RelCutoff=2)
```

If you previously tuned `op.RelCutoff` for accuracy, keep the same
value in Python — the integration scheme is bit-identical for
`interp='flat'` and ULP-close for `interp='curv'`.

### 6. Closed surfaces & EELS

EELS in MATLAB defaults to `closed=p.closed` from the `comparticle`.
In Python, build the particle with `closed=[1]` (or whichever is
closed) and pass the same `p` to `EELSRet(p, impact, w, vel)`:

```python
p   = ComParticle(eps, [sphere], [[2, 1]], 1)         # 1 = sphere is closed
exc = EELSRet(p, impact=np.array([[0, 0]]), width=0.5, vel=0.7)
```

### 7. `iter` and `aca` options

MATLAB:

```matlab
op.iter = struct('tol', 1e-6, 'restart', 30);
op.aca  = struct('htol', 1e-6, 'kmax', 100);
bem     = bemretiter(p, op);
```

Python:

```python
bem = BEMRetIter(p,
                 iter={"tol": 1e-6, "restart": 30, "maxit": 200},
                 aca ={"htol": 1e-6, "kmax": 100, "cleaf": 32, "eta": 2.5})
```

The default `htol=1e-6` matches MATLAB. For very small particles
(<2 nm) consider `htol=1e-8`.

### 8. `clear(bem)` vs `bem.clear()`

MATLAB: `clear(bem)` releases cached factors. Python:
`bem.clear()`. Useful inside a wavelength loop if memory is tight.

### 9. Cell arrays → Python lists

Whenever MATLAB uses `{ ... }` for a cell array, Python uses `[ ... ]`.

```matlab
{epsconst(1), epstable('gold.dat')}
```
becomes
```python
[EpsConst(1.0), EpsTable("gold.dat")]
```

### 10. `struct` returns → `CompStruct`

MATLAB returns nested structs from `bem \ exc`, `exc.potential(...)`,
etc. Python returns a `CompStruct` object whose fields are accessible
both as attributes and via `getfields(s, 'phi')`:

```python
sig.phi          # scalar potential
sig.sig1         # surface charge on side 1
getfields(sig, "phi", "sig1")     # tuple
```

### 11. Plotting

MATLAB `plot(p)` opens a figure. Python uses `BemPlot`:

```python
from mnpbem.misc import BemPlot
fig = BemPlot()
fig.plot(p)
fig.show()
```

For matplotlib non-interactive use (CI, headless), set the backend
before importing:

```python
import matplotlib
matplotlib.use("Agg")
```

### 12. Far-field collection mesh

In MATLAB you sometimes pass `[]` for `pinfty`. In Python, pass
`None` or omit:

```python
spec = SpectrumRet()                # uses default 256-face unit sphere
```

### 13. `bem.solve` return signature

MATLAB: `sig = bem \ exc` (single return).
Python: `sig, bem = bem.solve(exc)` (two returns: the second is the
solver itself, with cached factors). Always unpack both — if you do
`sig = bem.solve(exc)` you'll get a tuple by accident.

### 14. Wavelength sweep performance

In MATLAB, `bem.init(enei)` is sometimes called explicitly. In Python,
`bem.solve(exc.potential(p, enei))` does it implicitly. If you sweep
many wavelengths for the same particle, the dense-matrix factor is
re-built each call — use `BEMRetIter` (ACA + GMRES) which scales much
better, or use `compute_spectrum_parallel` for embarrassingly-parallel
wavelength sweeps.

### 15. GPU acceleration

Set `MNPBEM_GPU=1` (and ensure `cupy` is installed) to run dense
operations on the GPU. The Python API is otherwise identical:

```bash
MNPBEM_GPU=1 python my_script.py
```

For multi-GPU wavelength dispatch see `examples/07_gpu_multigpu.py`.

### 16. Nonlocal hydrodynamic Drude (v1.1.0)

`EpsNonlocal` packages the Yu Luo et al. cover-layer formulation. A few
caveats when porting MATLAB nonlocal scripts:

1. **Shell thickness `delta_d`**: 0.05 nm is the standard. Too small
   (< 0.01 nm) introduces numerical noise; too large (> 0.2 nm) breaks
   the thin-shell approximation.
2. **β (Fermi velocity)**: metal-dependent. Default values come from
   `sqrt(3/5) * v_F * hbar`: Au ≈ 0.714 eV·nm, Ag ≈ 0.864 eV·nm,
   Al ≈ 1.034 eV·nm. Pass `beta=` explicitly for non-tabulated metals
   or when reproducing a specific paper.
3. **Geometry**: `ComParticle` epstab has **3 entries**
   (`[embed, core, shell]`); `particles` has **2** (`[shell, core]`);
   `inout = [[3, 1], [2, 3]]`; `closed = [1, 2]`.
4. **Cover layer makes the BEM result smoother**, but the mesh face
   count grows by ≈ 2× → memory grows by ≈ 4× (standard formulation).
   - **v1.2.0+**: `schur=True` 적용 시 메모리 ~2× 만 (50% 절감), LU 풀이
     ~30% 가속. cover layer 변수를 schur 소거하여 reduced matrix 풀이.

     ```python
     bem = BEMStat(p, refun=refun, schur=True)   # v1.2.0
     # 또는:
     bem = BEMRet(p, refun=refun, schur=True)
     ```

     `schur='auto'` 또는 wrapper 가 cover layer 자동 감지.
   - 큰 mesh (25k+ face) + nonlocal 시나리오는 `MNPBEM_VRAM_SHARE_GPUS=N`
     환경변수로 multi-GPU pool 활용 (cuSolverMg 백엔드, v1.2.0+).
5. **`BEMRet` `refun` parameter** is new in v1.1.0 — use it when you
   want the retarded path with the cover-layer integration. `BEMStat`
   has accepted `refun` since v1.0.0.

### 17. Schur complement (v1.2.0)

EpsNonlocal cover-layer formulation 의 메모리 폭증 (face count ~2× →
matrix memory ~4×) 을 완화하기 위한 옵션.

| 동작 | 설정 | 메모리 | LU 시간 |
|---|---|---|---|
| Standard formulation | `schur=False` (default) | 4× | baseline |
| Schur complement | `schur=True` | ~2× | -30% |
| Auto-detect | `schur='auto'` | (cover layer 감지 시 2×) | -30% |

```python
# v1.1.0 — standard formulation
bem = BEMStat(p, refun=refun)

# v1.2.0 — Schur complement
bem = BEMStat(p, refun=refun, schur=True)
```

수학적으로 standard formulation 과 동등 (rel < 1e-12 수준에서 일치).
회귀 테스트는 둘 다 회기적으로 검증된다.

### 18. Multi-GPU VRAM share (v1.2.0)

단일 GPU VRAM (예: RTX A6000 48 GB) 을 초과하는 큰 dense LU
(25k+ face) 를 multi-GPU 메모리 풀로 처리.

```python
import os
os.environ['MNPBEM_VRAM_SHARE_GPUS']    = '2'           # 2 GPU pool = 96 GB
os.environ['MNPBEM_VRAM_SHARE_BACKEND'] = 'cusolvermg'  # default

from mnpbem.bem import BEMRet
bem = BEMRet(p)   # 자동으로 multi-GPU LU 활용
```

이전 (v1.1.0) 에는 단일 GPU OOM 이 되던 mesh 크기가 v1.2.0 부터
multi-GPU pool 로 fit 가능. wavelength 분배 (Lane D, multi-worker) 와
결합 가능 — 8 GPU 환경에서 2-GPU pool 4개 (`n_workers=4`,
`n_gpus_per_worker=2`).

### 19. Large mesh strategy (v1.3.0)

25k+ face 시뮬레이션을 어떻게 처리할지 결정 가이드.

| mesh face count | 권장 |
|---|---|
| < 1k | dense BEMStat / BEMRet |
| 1k - 5k | dense + Numba JIT (`MNPBEM_NUMBA=1`) |
| 5k - 25k | `BEMRetIter(p, hmatrix=True)` (v1.3.0) — H-matrix iter |
| 25k+ | `BEMRetIter(p, hmatrix=True)` + VRAM share (multi-GPU) |
| 50k+ | + 별도 H-matrix 분산 / preconditioner 튜닝 (실험적) |

```python
from mnpbem.bem import BEMRetIter

# 큰 mesh + iterative + H-matrix
bem = BEMRetIter(p, hmatrix=True, htol=1e-6,
                 tol=1e-6, maxiter=200)

# + multi-GPU VRAM share (v1.2.0)
import os
os.environ['MNPBEM_VRAM_SHARE_GPUS'] = '4'
```

`pymnpbem_simulation` 사용자는 YAML 의 `iter.hmatrix: 'auto'` 만
켜두면 face count 5000+ 에서 자동 활성된다 (v1.3.0).

**Common issue**: GMRES 가 수렴하지 않으면 다음을 시도:

- `tol` 완화 (예: 1e-4) 또는 `maxiter` 증가.
- `htol` 강화 (예: 1e-7) — H-matrix 압축이 너무 느슨해 condition
  악화 시 효과적.
- ~~preconditioner 강화~~ → **v1.5.0 부터 `preconditioner='auto'`
  지원** (H-matrix LU). 256-face sphere 에서 GMRES iter 55 → 1.
  pitfall #21 참고.

H-matrix vs dense 결과는 `htol` 기반 — `htol=1e-6` 에서
relative `< 1e-4` 수준으로 일치 (`docs/PERFORMANCE.md` §11).

### 20. Install 변경 (v1.4.0)

v1.3.0 이전 까지는 `pip install mnpbem` 한 줄이 (사실상)
모든 의존성을 끌어왔지만, v1.4.0 부터는 사용 환경에 맞춰
**extras** 로 install 범위를 고를 수 있다.

기존 (v1.3.0 이하):

```bash
pip install mnpbem            # 사실상 모든 extras 가 같이 들어옴
```

v1.4.0+:

```bash
pip install mnpbem            # CPU only (default, 가장 가벼움)
pip install mnpbem[gpu]       # + cupy-cuda12x (NVIDIA GPU 가속)
pip install mnpbem[mpi]       # + mpi4py (multi-node wavelength 분배)
pip install mnpbem[fmm]       # + fmm3dpy (free-space ret meshfield)
pip install mnpbem[all]       # gpu + mpi + fmm 전부
pip install mnpbem[dev]       # 개발 환경 (pytest, ruff 등)
```

기존 v1.3.0 코드는 **수정 X**. install 명령만 환경에 맞게
다르게 사용하면 된다. 시뮬레이션 코드 자체는 cupy 가 lazy
import 라 CPU only 환경에서도 동작 (CPU fallback).

GPU 활성 가능 여부를 runtime 에서 확인하려면:

```python
from mnpbem.utils.gpu import has_gpu_capability, get_install_hint

if not has_gpu_capability(verbose=True):
    print(get_install_hint())
```

자세한 시나리오별 install 절차는 `docs/INSTALL.md` 참고.

### 21. Large nonlocal mesh strategy (v1.5.0)

v1.5.0 부터 cover-layer (nonlocal) 계열 시뮬레이션에 H-matrix LU
preconditioner + Schur × Iterative 가 추가되어, 큰 nonlocal mesh 에서
GMRES 수렴 가속 및 cover layer 변수 implicit 소거가 가능하다.

| 시나리오 | 권장 옵션 |
|---|---|
| 작은 nonlocal (< 1k face cover) | `BEMStat(p, schur=True)` (v1.2.0) |
| 중간 nonlocal (1-5k) | `BEMRetIter(p, hmatrix=True, schur=True)` |
| 큰 nonlocal (5k+) | + `preconditioner='auto'` |
| 25k+ nonlocal (cover 50k+ face) | + VRAM share (v1.2.0) — Sigma H-matrix 재구성은 v1.6+ |

**예시**:

```python
from mnpbem.bem import BEMRetIter

bem = BEMRetIter(p, refun=refun,
                 hmatrix=True,           # v1.3.0
                 schur=True,             # v1.5.0 (cover layer 자동 감지)
                 preconditioner='auto')  # v1.5.0 (수렴 가속)

# 25k+ 면 추가:
import os
os.environ['MNPBEM_VRAM_SHARE_GPUS'] = '4'  # v1.2.0
```

`pymnpbem_simulation` 사용자는 YAML 의 `iter.preconditioner: 'auto'`,
`iter.schur: 'auto'` 만 켜두면 자동 활성된다 (v1.5.0).

**Common issue**:

- 25k+ face 의 진짜 memory-friendly preconditioner 는 Sigma/Delta
  H-matrix 재구성이 필요 — v1.6+ 과제. 현재 `hlu_tree` 는 8N×8N
  결합 시스템 특성상 dense fallback 로 동작하는 경우가 있다.
- `BEMStatIter` tree mode 는 diagonal term 깨짐으로 자동 dense
  fallback (one-time log).

### 22. mesh_density 우선 (v1.6.0)

이전 (v1.5.x): `n_per_edge` 가 explicit 명시 시 그대로 사용.
v1.6.0+: **`mesh_density` 가 우선** (mnpbem_simulation 의미체계 통일).

```python
# yaml 에 둘 다 있으면 mesh_density 우선
structure:
  core_size: 47
  mesh_density: 2     # 우선 사용 → core size 기준 n_per_edge=24
  n_per_edge: 24      # 무시 (mesh_density 가 결정)
```

backward-compat: `n_per_edge` 만 있는 yaml 은 그대로 사용.

---

## What does **not** map cleanly

A small number of MATLAB features are not yet ported — usually because
their Python equivalent is materially different. If you rely on these,
file an issue.

| MATLAB feature | Status |
|---|---|
| `nonlocal.m` (Pendry-style nonlocal cover layer) | done — see `EpsNonlocal` / `make_nonlocal_pair` (v1.1.0) and pitfall #16 |
| GUI (`MNPBEM_GUI`) | not ported (use `BemPlot` for static viewing) |
| `makemnpbemhelp.m` (HTML help generator) | replaced by this `docs/` directory |
| `compound.norm`, `compound.union` (set-algebra helpers) | partial — see `docs/API_REFERENCE.md`, `Compound` |

For everything in the standard `Demo/` directory (72 demos, including
EELS, layered substrate, dipole decay, plane wave, mirror symmetry),
the Python port reproduces the MATLAB output to machine precision in 55
of 72 cases (see `docs/PERFORMANCE.md` for the rest).

---

## Quick reference: where to find each MATLAB file

| MATLAB directory | Python module |
|---|---|
| `Particles/` | `mnpbem.geometry` |
| `Material/` | `mnpbem.materials` |
| `Greenfun/` | `mnpbem.greenfun` |
| `BEM/` | `mnpbem.bem` |
| `Simulation/` | `mnpbem.simulation`, `mnpbem.spectrum` |
| `Mie/` | `mnpbem.mie` |
| `Misc/` | `mnpbem.misc` |
| `Mesh2d/` | `mnpbem.geometry.mesh2d` (used by `tripolygon`) |
| `Demo/` | `examples/` (selected) + `validation/` (full regression suite) |
