"""
Value array for plotting within MNPBEM.

MATLAB: @valarray/
"""

import numpy as np
from typing import Any, Dict, Optional


class ValArray(object):
    """
    Value array for plotting on particle surface.

    MATLAB: @valarray

    Parameters
    ----------
    p : particle object
        Discretized particle
    val : ndarray or None
        Value array
    truecolor : bool
        If True, treat val as RGB truecolor array

    Methods
    -------
    init2(val, truecolor) -> None
    get_val(opt) -> ndarray
    plot(opt, **kwargs) -> None
    isbase(p) -> bool
    ispage() -> bool
    pagesize(val) -> tuple
    depends(*properties) -> bool
    """

    def __init__(self, p: object,
            val: Optional[np.ndarray] = None,
            truecolor: bool = False) -> None:
        self.p = p
        self.truecolor = False
        self.h = None

        if val is not None:
            # expand to full size or interpolate from faces to vertices
            if val.shape[0] == 1:
                val = np.tile(val, (p.nverts, 1))
            if val.shape[0] == p.n and hasattr(p, 'interp'):
                val = p.interp(val)
            self.val = val
            self.truecolor = truecolor
        else:
            # default golden color
            self.val = np.tile([1.0, 0.7, 0.0], (p.nverts, 1))
            self.truecolor = True

    def init2(self, val: Optional[np.ndarray] = None,
            truecolor: bool = False) -> None:
        """
        MATLAB: @valarray/init2.m

        Re-initialization of value array.
        """
        if val is not None:
            if val.shape[0] == 1:
                val = np.tile(val, (self.p.nverts, 1))
            if val.shape[0] == self.p.n and hasattr(self.p, 'interp'):
                val = self.p.interp(val)
            self.val = val
            self.truecolor = truecolor
        else:
            self.val = np.tile([1.0, 0.7, 0.0], (self.p.nverts, 1))
            self.truecolor = True

    def get_val(self, opt: Dict[str, Any]) -> np.ndarray:
        """
        MATLAB: @valarray/subsref.m (() case)

        Get value array with applied plot options.
        """
        if self.truecolor:
            return self.val
        elif not self.ispage():
            return opt['fun'](self.val)
        else:
            return opt['fun'](self.val[:, opt['ind']])

    def plot(self, opt: Dict[str, Any], **kwargs: Any) -> None:
        """
        MATLAB: @valarray/plot.m

        Plot value array using matplotlib.
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        val = self.get_val(opt)
        face_alpha = kwargs.get('FaceAlpha', 1.0)

        if self.h is None:
            verts = self.p.verts
            faces = self.p.faces

            # build polygon collection
            polys = []
            face_colors = []
            for i in range(faces.shape[0]):
                face = faces[i]
                valid = face[~np.isnan(face)].astype(int)
                poly = verts[valid]
                polys.append(poly)
                # face color from vertex average
                if val.ndim == 1:
                    face_colors.append(np.mean(val[valid]))
                else:
                    face_colors.append(np.mean(val[valid], axis = 0))

            face_colors = np.array(face_colors)

            ax = plt.gca()
            if not hasattr(ax, 'get_zlim'):
                fig = plt.gcf()
                ax = fig.add_subplot(111, projection = '3d')

            pc = Poly3DCollection(polys, alpha = face_alpha)

            if self.truecolor:
                pc.set_facecolors(np.clip(face_colors, 0, 1))
            else:
                pc.set_array(face_colors.ravel())

            pc.set_edgecolor('none')
            ax.add_collection3d(pc)
            self.h = pc
        else:
            # update existing plot
            face_colors = []
            faces = self.p.faces
            for i in range(faces.shape[0]):
                face = faces[i]
                valid = face[~np.isnan(face)].astype(int)
                if val.ndim == 1:
                    face_colors.append(np.mean(val[valid]))
                else:
                    face_colors.append(np.mean(val[valid], axis = 0))

            face_colors = np.array(face_colors)
            if self.truecolor:
                self.h.set_facecolors(np.clip(face_colors, 0, 1))
            else:
                self.h.set_array(face_colors.ravel())

    def isbase(self, p: object) -> bool:
        """
        MATLAB: @valarray/isbase.m

        Check if particle is same as stored particle.
        """
        if not hasattr(p, 'verts'):
            return False
        if p.verts.shape[0] != self.p.verts.shape[0]:
            return False
        return np.all(p.verts == self.p.verts)

    def ispage(self) -> bool:
        """
        MATLAB: @valarray/ispage.m

        True if value array is multi-dimensional.
        """
        if self.truecolor:
            return False
        return self.val.ndim > 1 and self.val.shape[1] != 1

    def pagesize(self, val: Optional[np.ndarray] = None) -> tuple:
        """
        MATLAB: @valarray/pagesize.m

        Paging size of value array.
        """
        if val is None:
            val = self.val
        return val.shape[1:]

    def depends(self, *properties: str) -> bool:
        """
        MATLAB: @valarray/depends.m

        Check whether object uses one of the given properties.
        """
        if 'ind' in properties and self.ispage():
            return True
        if 'fun' in properties and not self.truecolor:
            return True
        return False

    def min_val(self, opt: Dict[str, Any]) -> Optional[float]:
        """
        MATLAB: @valarray/min.m
        """
        if self.truecolor:
            return None
        return np.min(self.get_val(opt))

    def max_val(self, opt: Dict[str, Any]) -> Optional[float]:
        """
        MATLAB: @valarray/max.m
        """
        if self.truecolor:
            return None
        return np.max(self.get_val(opt))

    def __repr__(self) -> str:
        return 'ValArray(n={}, truecolor={})'.format(
            self.val.shape[0], self.truecolor)
