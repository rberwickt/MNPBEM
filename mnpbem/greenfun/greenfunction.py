import os
import sys

from typing import List, Dict, Tuple, Optional, Union, Any, Callable

from ..bem.bembase import BemBase
from .compgreen_stat import CompGreenStat
from .compgreen_ret import CompGreenRet
from .compgreen_stat_mirror import CompGreenStatMirror
from .compgreen_ret_mirror import CompGreenRetMirror
from .compgreen_stat_layer import CompGreenStatLayer
from .compgreen_ret_layer import CompGreenRetLayer


# -- registry entries --------------------------------------------------------

class _CompGreenStatEntry(BemBase):
    name = 'greenfunction'
    needs = [{'sim': 'stat'}]

class _CompGreenRetEntry(BemBase):
    name = 'greenfunction'
    needs = [{'sim': 'ret'}]

class _CompGreenStatMirrorEntry(BemBase):
    name = 'greenfunction'
    needs = [{'sim': 'stat'}, 'sym']

class _CompGreenRetMirrorEntry(BemBase):
    name = 'greenfunction'
    needs = [{'sim': 'ret'}, 'sym']

class _CompGreenStatLayerEntry(BemBase):
    name = 'greenfunction'
    needs = [{'sim': 'stat'}, 'layer']

class _CompGreenRetLayerEntry(BemBase):
    name = 'greenfunction'
    needs = [{'sim': 'ret'}, 'layer']


_ENTRY_TO_CLASS = {
    _CompGreenStatEntry: CompGreenStat,
    _CompGreenRetEntry: CompGreenRet,
    _CompGreenStatMirrorEntry: CompGreenStatMirror,
    _CompGreenRetMirrorEntry: CompGreenRetMirror,
    _CompGreenStatLayerEntry: CompGreenStatLayer,
    _CompGreenRetLayerEntry: CompGreenRetLayer,
}


def greenfunction(p1: Any,
        p2: Any,
        op: Optional[Dict[str, Any]] = None,
        **kwargs: Any) -> Any:

    if op is None:
        op = {}

    merged = dict(op)
    merged.update(kwargs)

    entry = BemBase.find('greenfunction', merged)
    if entry is None:
        raise ValueError('[error] no green function class found for given <options>')

    cls = _ENTRY_TO_CLASS.get(entry, None)
    if cls is None:
        raise ValueError('[error] no green function class mapped for entry <{}>'.format(entry.__name__))

    return cls(p1, p2, op, **kwargs)
