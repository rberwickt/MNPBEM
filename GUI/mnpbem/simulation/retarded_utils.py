import os
import sys

from typing import List, Dict, Tuple, Optional, Union, Any, Callable

import numpy as np

from ..greenfun import CompStruct
from ..misc.math_utils import inner


def scattering(field: CompStruct,
        medium: Optional[Any] = None) -> Union[np.ndarray, Tuple[np.ndarray, CompStruct]]:

    # MATLAB: Simulation/retarded/scattering.m
    # Radiated power for electromagnetic fields

    pinfty = field.p
    e = field.e
    h = field.h

    squeeze_needed = False
    if e.ndim == 2:
        e = e[:, :, np.newaxis]
        h = h[:, :, np.newaxis]
        squeeze_needed = True

    # Poynting vector in direction of outer surface normal
    # dsca = 0.5 * real(inner(nvec, cross(e, conj(h), 2)))
    poynting = np.cross(e, np.conj(h), axis = 1)  # (ndir, 3, ...)
    dsca_arr = 0.5 * np.real(inner(pinfty.nvec, poynting))  # (ndir, ...)

    area = pinfty.area.copy()

    # Filter by medium if specified
    if medium is not None and hasattr(pinfty, 'inout'):
        inout = pinfty.inout
        if hasattr(inout, '__len__') and len(inout) > 0:
            last_col = inout[:, -1] if inout.ndim > 1 else inout
            mask = (last_col == medium)
            area[mask] = 0

    # Total radiated power: sca = squeeze(matmul(area', dsca))
    sca = np.tensordot(area, dsca_arr, axes = ([0], [0]))
    sca = np.squeeze(sca)

    return sca


def extinction(field: CompStruct,
        infield: CompStruct) -> np.ndarray:

    # MATLAB: Simulation/retarded/extinction.m
    # Extinction cross section from electromagnetic fields

    pinfty = field.p
    e = field.e
    h = field.h
    ein = infield.e
    hin = infield.h

    if e.ndim == 2:
        e = e[:, :, np.newaxis]
        h = h[:, :, np.newaxis]
    if ein.ndim == 2:
        ein = ein[:, :, np.newaxis]
        hin = hin[:, :, np.newaxis]

    # Background dielectric constant
    if hasattr(pinfty, 'eps') and hasattr(pinfty, 'inout'):
        inout = pinfty.inout
        last_idx = inout[-1] if inout.ndim == 1 else inout[:, -1][-1]
        nb = np.sqrt(pinfty.eps[last_idx](field.enei)[0])
    elif hasattr(pinfty, 'eps'):
        eps_list = pinfty.eps
        nb = np.sqrt(eps_list[-1](field.enei)[0])
    else:
        nb = 1.0

    # Extinction: -1/nb * nvec . real(cross(e, conj(hin)) + cross(conj(ein), h))
    cross1 = np.cross(e, np.conj(hin), axis = 1)
    cross2 = np.cross(np.conj(ein), h, axis = 1)
    integrand = np.real(cross1 + cross2)

    dext = -(1.0 / nb) * inner(pinfty.nvec, integrand)

    ext = np.tensordot(pinfty.area, dext, axes = ([0], [0]))
    ext = np.squeeze(ext)

    return ext


def absorption(field: CompStruct,
        infield: CompStruct) -> np.ndarray:

    # MATLAB: Simulation/retarded/absorption.m
    # Absorption cross section from electromagnetic fields

    return extinction(field, infield) - scattering(field)
