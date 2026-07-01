import os
import sys

from typing import List, Dict, Tuple, Optional, Union, Any, Callable

from ..bem.bembase import BemBase
from .spectrum_stat import SpectrumStat
from .spectrum_ret import SpectrumRet
from .spectrum_stat_layer import SpectrumStatLayer
from .spectrum_ret_layer import SpectrumRetLayer


# -- registry entries --------------------------------------------------------

class _SpectrumStatEntry(BemBase):
    name = 'spectrum'
    needs = [{'sim': 'stat'}]

class _SpectrumRetEntry(BemBase):
    name = 'spectrum'
    needs = [{'sim': 'ret'}]

class _SpectrumStatLayerEntry(BemBase):
    name = 'spectrum'
    needs = [{'sim': 'stat'}, 'layer']

class _SpectrumRetLayerEntry(BemBase):
    name = 'spectrum'
    needs = [{'sim': 'ret'}, 'layer']


_ENTRY_TO_CLASS = {
    _SpectrumStatEntry: SpectrumStat,
    _SpectrumRetEntry: SpectrumRet,
    _SpectrumStatLayerEntry: SpectrumStatLayer,
    _SpectrumRetLayerEntry: SpectrumRetLayer,
}


def spectrum(op: Optional[Dict[str, Any]] = None,
        **kwargs: Any) -> Any:

    if op is None:
        op = {}

    merged = dict(op)
    merged.update(kwargs)

    entry = BemBase.find('spectrum', merged)
    if entry is None:
        raise ValueError('[error] no spectrum class found for given <options>')

    cls = _ENTRY_TO_CLASS.get(entry, None)
    if cls is None:
        raise ValueError('[error] no spectrum class mapped for entry <{}>'.format(entry.__name__))

    return cls(op, **kwargs)
