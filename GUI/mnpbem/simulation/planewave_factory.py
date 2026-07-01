import os
import sys

from typing import List, Dict, Tuple, Optional, Union, Any, Callable

from ..bem.bembase import BemBase
from .planewave_stat import PlaneWaveStat
from .planewave_ret import PlaneWaveRet
from .planewave_stat_mirror import PlaneWaveStatMirror
from .planewave_ret_mirror import PlaneWaveRetMirror
from .planewave_stat_layer import PlaneWaveStatLayer
from .planewave_ret_layer import PlaneWaveRetLayer


# -- registry entries --------------------------------------------------------

class _PlaneWaveStatEntry(BemBase):
    name = 'planewave'
    needs = [{'sim': 'stat'}]

class _PlaneWaveRetEntry(BemBase):
    name = 'planewave'
    needs = [{'sim': 'ret'}]

class _PlaneWaveStatMirrorEntry(BemBase):
    name = 'planewave'
    needs = [{'sim': 'stat'}, 'sym']

class _PlaneWaveRetMirrorEntry(BemBase):
    name = 'planewave'
    needs = [{'sim': 'ret'}, 'sym']

class _PlaneWaveStatLayerEntry(BemBase):
    name = 'planewave'
    needs = [{'sim': 'stat'}, 'layer']

class _PlaneWaveRetLayerEntry(BemBase):
    name = 'planewave'
    needs = [{'sim': 'ret'}, 'layer']


_ENTRY_TO_CLASS = {
    _PlaneWaveStatEntry: PlaneWaveStat,
    _PlaneWaveRetEntry: PlaneWaveRet,
    _PlaneWaveStatMirrorEntry: PlaneWaveStatMirror,
    _PlaneWaveRetMirrorEntry: PlaneWaveRetMirror,
    _PlaneWaveStatLayerEntry: PlaneWaveStatLayer,
    _PlaneWaveRetLayerEntry: PlaneWaveRetLayer,
}


def planewave(pol: Any,
        dir: Any = None,
        op: Optional[Dict[str, Any]] = None,
        **kwargs: Any) -> Any:

    if op is None:
        op = {}

    merged = dict(op)
    merged.update(kwargs)

    entry = BemBase.find('planewave', merged)
    if entry is None:
        raise ValueError('[error] no planewave class found for given <options>')

    cls = _ENTRY_TO_CLASS.get(entry, None)
    if cls is None:
        raise ValueError('[error] no planewave class mapped for entry <{}>'.format(entry.__name__))

    if dir is not None:
        return cls(pol, dir, op, **kwargs)
    else:
        return cls(pol, op, **kwargs)
