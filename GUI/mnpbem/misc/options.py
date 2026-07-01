"""
BEM options handling for MNPBEM.

MATLAB: Misc/bemoptions.m, Misc/getbemoptions.m, Misc/getfields.m
"""

from typing import Any, Dict, List, Optional


def bemoptions(op: Optional[Dict[str, Any]] = None,
        **kwargs: Any) -> Dict[str, Any]:
    """
    MATLAB: Misc/bemoptions.m

    Set standard options for MNPBEM simulation.

    Parameters
    ----------
    op : dict, optional
        Option dictionary from previous call
    **kwargs : dict
        Additional property name-value pairs

    Returns
    -------
    op : dict
        Dictionary with standard or user-defined options
    """
    if op is None:
        op = {}
        op['sim'] = 'ret'
        op['waitbar'] = 1
        op['RelCutoff'] = 3
        op['order'] = 5
        op['interp'] = 'flat'

    op.update(kwargs)
    return op


def getbemoptions(*args: Any,
        **kwargs: Any) -> Dict[str, Any]:
    """
    MATLAB: Misc/getbemoptions.m

    Get options for MNPBEM simulation.

    Processes arguments in order: dicts are merged, lists of strings trigger
    substructure extraction, and keyword pairs are added directly.

    Parameters
    ----------
    *args : dicts, lists, or key-value pairs
    **kwargs : additional keyword arguments

    Returns
    -------
    op : dict
    """
    op = {}
    sub = None
    it = 0

    while it < len(args):
        arg = args[it]

        if isinstance(arg, dict):
            op.update(arg)
            if sub is not None:
                op = _extract_subs(op, sub)
            it += 1

        elif isinstance(arg, str):
            if it + 1 < len(args):
                op[arg] = args[it + 1]
                it += 2
            else:
                it += 1

        elif isinstance(arg, (list, tuple)):
            sub = arg
            op = _extract_subs(op, sub)
            it += 1

        else:
            it += 1

    op.update(kwargs)
    return op


def _extract_subs(op: Dict[str, Any],
        sub: List[str]) -> Dict[str, Any]:
    """
    MATLAB: getbemoptions/subs

    Extract fields from substructures.
    """
    for name in sub:
        if name in op and isinstance(op[name], dict):
            for key, val in op[name].items():
                op[key] = val
    return op


def getfields(param: Dict[str, Any],
        *names: str) -> Dict[str, Any]:
    """
    MATLAB: Misc/getfields.m

    Extract fields from a dictionary. If names are provided,
    only extract those fields.

    Parameters
    ----------
    param : dict
        Source dictionary
    *names : str
        Field names to extract (all if empty)

    Returns
    -------
    fields : dict
        Extracted fields
    """
    if len(names) == 0:
        return dict(param)
    else:
        return {k: v for k, v in param.items() if k in names}
