import os
import sys

from typing import List, Dict, Tuple, Optional, Union, Any, Callable

from .bem_stat import BEMStat
from .bem_ret import BEMRet
from .bem_stat_iter import BEMStatIter
from .bem_ret_iter import BEMRetIter
from .bem_stat_mirror import BEMStatMirror
from .bem_ret_mirror import BEMRetMirror
from .bem_stat_layer import BEMStatLayer
from .bem_ret_layer import BEMRetLayer
from .bem_ret_layer_iter import BEMRetLayerIter


# solver class lookup table
# key: (mode, layer, mirror, iterative)
_SOLVER_MAP = {
    ('stat', False, False, False): BEMStat,
    ('stat', False, False, True):  BEMStatIter,
    ('stat', False, True,  False): BEMStatMirror,
    ('stat', True,  False, False): BEMStatLayer,
    ('ret',  False, False, False): BEMRet,
    ('ret',  False, False, True):  BEMRetIter,
    ('ret',  False, True,  False): BEMRetMirror,
    ('ret',  True,  False, False): BEMRetLayer,
    ('ret',  True,  False, True):  BEMRetLayerIter,
}

# iterative fallback lookup: iterative key -> dense fallback key
_ITERATIVE_FALLBACK = {
    ('stat', True,  False, True):  ('stat', True,  False, False),  # BEMStatLayer (no BEMStatLayerIter)
    ('stat', False, True,  True):  ('stat', False, True,  False),  # BEMStatMirror (no BEMStatMirrorIter)
    ('ret',  False, True,  True):  ('ret',  False, True,  False),  # BEMRetMirror (no BEMRetMirrorIter)
    ('ret',  True,  True,  True):  ('ret',  True,  True,  False),  # layer+mirror iterative -> dense
    ('stat', True,  True,  True):  ('stat', True,  True,  False),  # layer+mirror iterative -> dense
    ('ret',  True,  True,  False): None,  # layer+mirror dense -- may not exist
    ('stat', True,  True,  False): None,  # layer+mirror dense -- may not exist
}


def create_solver(
        particle: Any,
        mode: str = 'stat',
        layer: Optional[Any] = None,
        mirror: Optional[Any] = None,
        threshold: int = 1000,
        force: Optional[str] = None,
        **kwargs: Any) -> Any:

    assert mode in {'stat', 'ret'}, '[error] <mode> must be "stat" or "ret", got <{}>'.format(mode)
    assert force is None or force in {'dense', 'iterative'}, \
        '[error] <force> must be None, "dense", or "iterative", got <{}>'.format(force)

    has_layer = layer is not None
    has_mirror = mirror is not None

    # nfaces
    nfaces = particle.nfaces if hasattr(particle, 'nfaces') else particle.n

    # iterative decision
    if force == 'dense':
        use_iterative = False
    elif force == 'iterative':
        use_iterative = True
    else:
        use_iterative = nfaces > threshold

    key = (mode, has_layer, has_mirror, use_iterative)

    # direct lookup
    solver_cls = _SOLVER_MAP.get(key, None)

    # fallback: iterative not available -> try dense
    if solver_cls is None and use_iterative:
        fallback_key = _ITERATIVE_FALLBACK.get(key, None)
        if fallback_key is not None and fallback_key in _SOLVER_MAP:
            solver_cls = _SOLVER_MAP[fallback_key]
            print('[info] iterative solver not available for ({}, layer={}, mirror={}), '
                  'falling back to dense: {}'.format(mode, has_layer, has_mirror, solver_cls.__name__))
        else:
            # try without iterative flag as last resort
            dense_key = (mode, has_layer, has_mirror, False)
            solver_cls = _SOLVER_MAP.get(dense_key, None)
            if solver_cls is not None:
                print('[info] iterative solver not available for ({}, layer={}, mirror={}), '
                      'falling back to dense: {}'.format(mode, has_layer, has_mirror, solver_cls.__name__))

    if solver_cls is None:
        raise ValueError('[error] no solver available for (mode={}, layer={}, mirror={}, iterative={})'.format(
            mode, has_layer, has_mirror, use_iterative))

    # build constructor arguments
    # layer solvers take (p, layer, ...) while others take (p, ...)
    if has_layer and has_mirror:
        # layer + mirror: not supported for any solver currently
        raise ValueError('[error] layer + mirror combination is not supported')
    elif has_layer:
        solver = solver_cls(particle, layer, **kwargs)
    else:
        solver = solver_cls(particle, **kwargs)

    print('[info] create_solver: {} (nfaces={}, mode={}, layer={}, mirror={}, iterative={})'.format(
        solver_cls.__name__, nfaces, mode, has_layer, has_mirror, use_iterative))

    return solver
