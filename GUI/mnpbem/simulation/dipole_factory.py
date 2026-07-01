import os
import sys

from typing import List, Dict, Tuple, Optional, Union, Any, Callable

from ..bem.bembase import BemBase
from .dipole_stat import DipoleStat
from .dipole_ret import DipoleRet
from .dipole_stat_mirror import DipoleStatMirror
from .dipole_ret_mirror import DipoleRetMirror
from .dipole_stat_layer import DipoleStatLayer
from .dipole_ret_layer import DipoleRetLayer


# -- registry entries --------------------------------------------------------

class _DipoleStatEntry(BemBase):
    name = 'dipole'
    needs = [{'sim': 'stat'}]

class _DipoleRetEntry(BemBase):
    name = 'dipole'
    needs = [{'sim': 'ret'}]

class _DipoleStatMirrorEntry(BemBase):
    name = 'dipole'
    needs = [{'sim': 'stat'}, 'sym']

class _DipoleRetMirrorEntry(BemBase):
    name = 'dipole'
    needs = [{'sim': 'ret'}, 'sym']

class _DipoleStatLayerEntry(BemBase):
    name = 'dipole'
    needs = [{'sim': 'stat'}, 'layer']

class _DipoleRetLayerEntry(BemBase):
    name = 'dipole'
    needs = [{'sim': 'ret'}, 'layer']


_ENTRY_TO_CLASS = {
    _DipoleStatEntry: DipoleStat,
    _DipoleRetEntry: DipoleRet,
    _DipoleStatMirrorEntry: DipoleStatMirror,
    _DipoleRetMirrorEntry: DipoleRetMirror,
    _DipoleStatLayerEntry: DipoleStatLayer,
    _DipoleRetLayerEntry: DipoleRetLayer,
}


def dipole(pt: Any,
        op: Optional[Dict[str, Any]] = None,
        **kwargs: Any) -> Any:

    if op is None:
        op = {}

    merged = dict(op)
    merged.update(kwargs)

    entry = BemBase.find('dipole', merged)
    if entry is None:
        raise ValueError('[error] no dipole class found for given <options>')

    cls = _ENTRY_TO_CLASS.get(entry, None)
    if cls is None:
        raise ValueError('[error] no dipole class mapped for entry <{}>'.format(entry.__name__))

    return cls(pt, op, **kwargs)
