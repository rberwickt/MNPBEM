import os
import sys

from typing import List, Dict, Tuple, Optional, Union, Any, Callable, Type


class BemBase(object):

    name: str = ''
    needs: List[Any] = []

    _registry: List[Type['BemBase']] = []

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if cls.name and cls.needs:
            BemBase._registry.append(cls)

    @staticmethod
    def find(name: str,
            op: Optional[Dict[str, Any]] = None,
            **kwargs: Any) -> Optional[Type['BemBase']]:

        if op is None:
            op = {}

        op = dict(op)
        op.update(kwargs)

        best_cls: Optional[Type['BemBase']] = None
        best_count: int = 0

        for cls in BemBase._registry:
            if cls.name != name:
                continue

            n_match = 0
            all_ok = True

            for need in cls.needs:
                if isinstance(need, str):
                    if need not in op or op[need] is None:
                        all_ok = False
                        break
                    n_match += 1

                elif isinstance(need, dict):
                    for fname, fval in need.items():
                        if fname not in op or op[fname] != fval:
                            all_ok = False
                            break
                    if not all_ok:
                        break
                    n_match += 1

                else:
                    all_ok = False
                    break

            if all_ok and n_match > best_count:
                best_count = n_match
                best_cls = cls

        return best_cls
