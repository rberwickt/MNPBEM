import os
import sys

from typing import List, Dict, Tuple, Optional, Union, Any, Callable

from ..bem.bembase import BemBase
from .eels_stat import EELSStat
from .eels_ret import EELSRet


# -- registry entries --------------------------------------------------------

class _EELSStatEntry(BemBase):
    name = 'eels'
    needs = [{'sim': 'stat'}]

class _EELSRetEntry(BemBase):
    name = 'eels'
    needs = [{'sim': 'ret'}]


_ENTRY_TO_CLASS = {
    _EELSStatEntry: EELSStat,
    _EELSRetEntry: EELSRet,
}


def electronbeam(p: Any,
        op: Optional[Dict[str, Any]] = None,
        **kwargs: Any) -> Any:

    if op is None:
        op = {}

    merged = dict(op)
    merged.update(kwargs)

    entry = BemBase.find('eels', merged)
    if entry is None:
        raise ValueError('[error] no EELS class found for given <options>')

    cls = _ENTRY_TO_CLASS.get(entry, None)
    if cls is None:
        raise ValueError('[error] no EELS class mapped for entry <{}>'.format(entry.__name__))

    return cls(p, op, **kwargs)
