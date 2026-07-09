from typing import Any


class BEMLayerMirror(object):
    """Dummy class: BEM solvers for layer and mirror symmetry not implemented.

    MATLAB: @bemlayermirror
    """

    name = 'bemsolver'
    needs = {'sim': True, 'layer': True, 'sym': True}

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError(
            '[error] BEM solvers for layers and mirror symmetry not implemented')
