import sys
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    from joblib import Parallel, delayed
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False

try:
    from threadpoolctl import threadpool_limits
    HAS_THREADPOOLCTL = True
except ImportError:
    HAS_THREADPOOLCTL = False


def _is_stat_solver(bem: Any) -> bool:
    needs = getattr(bem, 'needs', {})
    return needs.get('sim', '') == 'stat'


def _is_ret_solver(bem: Any) -> bool:
    needs = getattr(bem, 'needs', {})
    return needs.get('sim', '') == 'ret'


def _collect_cross_sections(
        exc: Any,
        sig: Any,
        is_ret: bool) -> Dict[str, Any]:
    result = {}

    if hasattr(exc, 'extinction'):
        result['extinction'] = exc.extinction(sig)

    if hasattr(exc, 'scattering'):
        sca_val = exc.scattering(sig)
        if is_ret and isinstance(sca_val, tuple):
            result['scattering'] = sca_val[0]
        else:
            result['scattering'] = sca_val

    if hasattr(exc, 'absorption'):
        result['absorption'] = exc.absorption(sig)

    return result


def _compute_single_wavelength_stat(
        bem: Any,
        exc: Any,
        particle: Any,
        wavelength: float) -> Dict[str, Any]:
    pot = exc(particle, wavelength)
    sig, _ = bem.solve(pot)
    return _collect_cross_sections(exc, sig, is_ret = False)


def _compute_single_wavelength_ret(
        exc: Any,
        particle: Any,
        wavelength: float,
        bem_class: type,
        bem_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    # RET: Green function이 k에 의존하므로 각 worker에서 bem 객체를 새로 생성
    bem_local = bem_class(particle, **bem_kwargs)
    pot = exc(particle, wavelength)
    sig, _ = bem_local.solve(pot)
    return _collect_cross_sections(exc, sig, is_ret = True)


def compute_spectrum(
        bem: Any,
        exc: Any,
        particle: Any,
        wavelengths: Union[np.ndarray, List[float]],
        **kwargs: Any) -> Dict[str, np.ndarray]:
    wavelengths = np.asarray(wavelengths)
    n_wl = len(wavelengths)
    is_ret = _is_ret_solver(bem)

    results_list = []  # type: List[Dict[str, Any]]
    for wl in wavelengths:
        pot = exc(particle, wl)
        sig, _ = bem.solve(pot)
        cs = _collect_cross_sections(exc, sig, is_ret = is_ret)
        results_list.append(cs)

    return _aggregate_results(results_list, n_wl)


def compute_spectrum_parallel(
        bem: Any,
        exc: Any,
        particle: Any,
        wavelengths: Union[np.ndarray, List[float]],
        n_jobs: int = -1,
        backend: str = 'loky',
        **kwargs: Any) -> Dict[str, np.ndarray]:
    wavelengths = np.asarray(wavelengths)
    n_wl = len(wavelengths)

    if not HAS_JOBLIB:
        warnings.warn('[info] joblib 미설치: 순차 계산으로 fallback')
        return compute_spectrum(bem, exc, particle, wavelengths, **kwargs)

    is_stat = _is_stat_solver(bem)
    is_ret = _is_ret_solver(bem)

    blas_threads = kwargs.get('blas_threads', 1)

    if is_stat:
        # STAT: Green function이 파장 무관 → bem 객체를 공유하고 solve만 병렬화
        def _worker_stat(wl: float) -> Dict[str, Any]:
            if HAS_THREADPOOLCTL:
                with threadpool_limits(limits = blas_threads, user_api = 'blas'):
                    return _compute_single_wavelength_stat(bem, exc, particle, wl)
            else:
                return _compute_single_wavelength_stat(bem, exc, particle, wl)

        results_list = Parallel(n_jobs = n_jobs, backend = backend)(
            delayed(_worker_stat)(wl) for wl in wavelengths
        )

    elif is_ret:
        # RET: Green function이 k에 의존 → 각 worker에서 독립 계산
        bem_class = type(bem)
        bem_kwargs = kwargs.get('bem_kwargs', {})

        def _worker_ret(wl: float) -> Dict[str, Any]:
            if HAS_THREADPOOLCTL:
                with threadpool_limits(limits = blas_threads, user_api = 'blas'):
                    return _compute_single_wavelength_ret(exc, particle, wl, bem_class, bem_kwargs)
            else:
                return _compute_single_wavelength_ret(exc, particle, wl, bem_class, bem_kwargs)

        results_list = Parallel(n_jobs = n_jobs, backend = backend)(
            delayed(_worker_ret)(wl) for wl in wavelengths
        )

    else:
        raise ValueError('[error] 알 수 없는 solver 타입: <{}>'.format(getattr(bem, 'needs', {})))

    return _aggregate_results(results_list, n_wl)


def _aggregate_results(
        results_list: List[Dict[str, Any]],
        n_wl: int) -> Dict[str, np.ndarray]:
    if n_wl == 0:
        return {}

    keys = list(results_list[0].keys())
    output = {}  # type: Dict[str, np.ndarray]

    for key in keys:
        first_val = results_list[0][key]

        if np.isscalar(first_val) or (isinstance(first_val, np.ndarray) and first_val.ndim == 0):
            arr = np.empty(n_wl, dtype = float)
            for i in range(n_wl):
                arr[i] = float(results_list[i][key])
            output[key] = arr

        elif isinstance(first_val, np.ndarray) and first_val.ndim == 1:
            # 다중 편광: (n_wl, npol)
            npol = first_val.shape[0]
            arr = np.empty((n_wl, npol), dtype = float)
            for i in range(n_wl):
                arr[i, :] = results_list[i][key]
            output[key] = arr

        else:
            # 기타 형태: 리스트로 저장
            arr_list = [results_list[i][key] for i in range(n_wl)]
            output[key] = arr_list

    return output
