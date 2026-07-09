from __future__ import annotations

import numpy as np
from typing import Any, List, Optional, Sequence, Tuple, Union


class Compound(object):

    def __init__(self,
            eps: List[Any],
            p: List[Any],
            inout: Union[Sequence[int], np.ndarray]) -> None:

        self.eps = list(eps)
        self.p = list(p)

        inout_arr = np.asarray(inout, dtype = int)
        if inout_arr.ndim == 0:
            inout_arr = inout_arr.reshape(1)
        self.inout = inout_arr

        self._mask = list(range(len(self.p)))
        self._rebuild_pc()

    def _rebuild_pc(self) -> None:
        active = [self.p[i] for i in self._mask]
        if len(active) == 0:
            self.pc = None
        elif len(active) == 1:
            self.pc = active[0]
        else:
            result = active[0]
            for item in active[1:]:
                result = result + item
            self.pc = result

    @property
    def mask(self) -> List[int]:
        return list(self._mask)

    @mask.setter
    def mask(self,
            ind: Optional[Union[int, Sequence[int]]]) -> None:
        self._set_mask_impl(ind)

    def _set_mask_impl(self,
            ind: Optional[Union[int, Sequence[int]]]) -> None:
        if ind is None:
            self._mask = list(range(len(self.p)))
        else:
            if np.isscalar(ind):
                ind_list = [int(ind)]
            else:
                ind_list = [int(i) for i in np.asarray(ind).ravel()]
            self._mask = ind_list
        self._rebuild_pc()

    # MATLAB: @compound/mask.m
    def set_mask_matlab(self,
            ind: Optional[Union[int, Sequence[int]]] = None) -> 'Compound':
        if ind is None or (hasattr(ind, '__len__') and len(ind) == 0):
            self._set_mask_impl(None)
        else:
            if np.isscalar(ind):
                zero_idx = [int(ind) - 1]
            else:
                zero_idx = [int(i) - 1 for i in np.asarray(ind).ravel()]
            self._set_mask_impl(zero_idx)
        return self

    # MATLAB: @compound/size (subsref)
    @property
    def size(self) -> np.ndarray:
        return np.array([self.p[i].size if hasattr(self.p[i], 'size') and not callable(getattr(self.p[i], 'size', None))
                         else self._nsize(self.p[i]) for i in self._mask], dtype = int)

    @staticmethod
    def _nsize(part: Any) -> int:
        if hasattr(part, 'n'):
            return int(part.n)
        if hasattr(part, 'nfaces'):
            return int(part.nfaces)
        if hasattr(part, 'pos'):
            return int(np.asarray(part.pos).shape[0])
        raise AttributeError('[error] particle has no <size>/<n>/<nfaces>/<pos> attribute')

    @property
    def n(self) -> int:
        return int(np.sum(self._sizes_masked()))

    @property
    def np(self) -> int:
        return len(self._mask)

    def _sizes_masked(self) -> np.ndarray:
        return np.array([self._nsize(self.p[i]) for i in self._mask], dtype = int)

    # MATLAB: @compound/dielectric.m
    def dielectric(self,
            enei: float,
            inout: int) -> List[Any]:
        eps_table = [eps_fn(enei) for eps_fn in self.eps]

        # When inout has 2 columns (particles), pick the requested column
        # (inout is 1-indexed: 1 = inside, 2 = outside)
        if self.inout.ndim == 2 and self.inout.size != len(self.p):
            col = int(inout) - 1
            media = self.inout[self._mask, col]
            return [eps_table[int(m) - 1] for m in media]

        if self.inout.ndim == 2 and self.inout.shape[1] >= 2:
            col = int(inout) - 1
            media = self.inout[self._mask, col]
            return [eps_table[int(m) - 1] for m in media]

        # Compound of points: single medium per group
        media = np.atleast_1d(self.inout)[self._mask]
        return [eps_table[int(m) - 1] for m in media]

    # MATLAB: @compound/index.m
    def index(self,
            ipart: Union[int, Sequence[int]]) -> np.ndarray:
        sizes = self._sizes_masked()
        cum = np.zeros(len(sizes) + 1, dtype = int)
        cum[1:] = np.cumsum(sizes)

        if np.isscalar(ipart):
            ipart_list = [int(ipart)]
        else:
            ipart_list = [int(i) for i in np.asarray(ipart).ravel()]

        total = sum(int(sizes[i - 1]) for i in ipart_list)
        out = np.empty(total, dtype = int)
        offset = 0
        for i in ipart_list:
            start = int(cum[i - 1])
            stop = int(cum[i])
            seg = stop - start
            out[offset:offset + seg] = np.arange(start, stop, dtype = int)
            offset += seg
        return out

    # MATLAB: @compound/ipart.m
    def ipart(self,
            ind: Union[int, Sequence[int]]) -> Tuple[np.ndarray, np.ndarray]:
        sizes = self._sizes_masked()
        cum = np.zeros(len(sizes) + 1, dtype = int)
        cum[1:] = np.cumsum(sizes)

        ind_arr = np.atleast_1d(np.asarray(ind, dtype = int)).ravel()

        ipart_out = np.empty(len(ind_arr), dtype = int)
        for k, i in enumerate(ind_arr):
            # MATLAB: find( i > siz, 1, 'last' )
            # siz = [0, cumsum(...)]; last index where i (1-indexed) > siz value
            mask = i > cum
            if not np.any(mask):
                raise IndexError('[error] index {} out of range for <ipart>'.format(i))
            ipart_out[k] = int(np.max(np.where(mask)[0])) + 1

        rel = ind_arr - cum[ipart_out - 1]
        return ipart_out, rel

    # MATLAB: @compound/expand.m
    def expand(self,
            val: Any) -> np.ndarray:
        sizes = self._sizes_masked()
        total = int(np.sum(sizes))

        if isinstance(val, (list, tuple)):
            assert len(val) == len(sizes), \
                '[error] <val> length {} does not match masked particle count {}'.format(len(val), len(sizes))
            first = np.asarray(val[0])
            if first.ndim == 0:
                out = np.empty(total, dtype = first.dtype)
                offset = 0
                for v, s in zip(val, sizes):
                    out[offset:offset + s] = v
                    offset += s
                return out
            else:
                cols = first.shape[0] if first.ndim == 1 else first.shape[-1]
                out = np.empty((total, cols), dtype = first.dtype)
                offset = 0
                for v, s in zip(val, sizes):
                    v_arr = np.asarray(v).reshape(-1)
                    out[offset:offset + s] = np.tile(v_arr, (s, 1))
                    offset += s
                return out

        arr = np.asarray(val)
        if arr.ndim == 0:
            return np.full(total, arr.item(), dtype = arr.dtype)
        return np.tile(arr.reshape(1, -1), (total, 1))

    def eps1(self,
            enei: float) -> List[Any]:
        return self.expand(self.dielectric(enei, 1))

    def eps2(self,
            enei: float) -> List[Any]:
        return self.expand(self.dielectric(enei, 2))

    # MATLAB: @compound/set.m
    def set(self,
            **kwargs: Any) -> 'Compound':
        for key, value in kwargs.items():
            setattr(self.pc, key, value)
        return self

    # MATLAB: @compound/eq.m
    def __eq__(self,
            other: Any) -> bool:
        if not isinstance(other, Compound) and not hasattr(other, 'pc'):
            return NotImplemented
        try:
            a = np.asarray(self.pc.pos).ravel()
            b = np.asarray(other.pc.pos).ravel()
        except AttributeError:
            return NotImplemented
        if a.size != b.size:
            return False
        return bool(np.all(a == b))

    # MATLAB: @compound/ne.m
    def __ne__(self,
            other: Any) -> bool:
        result = self.__eq__(other)
        if result is NotImplemented:
            return NotImplemented
        return not result

    def __hash__(self) -> int:
        return id(self)

    # MATLAB: @compound/subsref.m -- Python attribute access
    # Implement as __getattr__ fallback so obj.<prop> delegates to pc.
    def __getattr__(self,
            name: str) -> Any:
        if name.startswith('_') or name in {'eps', 'p', 'inout', 'mask', 'pc',
                                            'size', 'n', 'np', 'dielectric',
                                            'index', 'ipart', 'expand',
                                            'eps1', 'eps2', 'set'}:
            raise AttributeError(name)
        pc = self.__dict__.get('pc', None)
        if pc is not None and hasattr(pc, name):
            return getattr(pc, name)
        raise AttributeError('[error] Compound has no attribute <{}>'.format(name))

    def __repr__(self) -> str:
        return 'Compound(np = {}, n = {})'.format(self.np, self.n)
