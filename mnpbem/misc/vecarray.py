"""
Vector array for plotting within MNPBEM.

MATLAB: @vecarray/
"""

import numpy as np
from typing import Any, Dict, Optional


class VecArray(object):
    """
    Vector array for plotting vector fields.

    MATLAB: @vecarray

    Parameters
    ----------
    pos : ndarray, shape (n, 3)
        Vector positions
    vec : ndarray, shape (n, 3) or (n, 3, ...)
        Vector array
    mode : str
        'cone' or 'arrow'

    Methods
    -------
    init2(vec, mode) -> None
    get_vec(opt) -> ndarray
    plot(opt, **kwargs) -> None
    isbase(pos) -> bool
    ispage() -> bool
    pagesize(vec) -> tuple
    depends(*properties) -> bool
    """

    def __init__(self, pos: np.ndarray, vec: np.ndarray,
            mode: str = 'cone') -> None:
        self.pos = pos
        self.vec = vec
        self.mode = mode
        self.h = None
        self.color = 'b'

    def init2(self, vec: np.ndarray,
            mode: Optional[str] = None) -> None:
        """
        MATLAB: @vecarray/init2.m

        Re-initialization of vector array.
        """
        self.vec = vec
        if mode is not None:
            self.mode = mode

    def get_vec(self, opt: Dict[str, Any]) -> np.ndarray:
        """
        MATLAB: @vecarray/subsref.m (() case)

        Get vector array with applied plot options.
        """
        if self.ispage():
            return opt['fun'](self.vec[:, :, opt['ind']])
        else:
            return opt['fun'](self.vec)

    def plot(self, opt: Dict[str, Any], **kwargs: Any) -> None:
        """
        MATLAB: @vecarray/plot.m

        Plot vector array using matplotlib.
        """
        import matplotlib.pyplot as plt

        vec = self.get_vec(opt)
        # vector length
        vec_len = np.sqrt(np.sum(np.abs(vec) ** 2, axis = 1))

        # apply scaling
        max_len = np.max(vec_len) if np.max(vec_len) > 0 else 1.0
        if opt['scale'] > 0:
            scale = opt['scale'] * opt['sfun'](vec_len / max_len)
        else:
            scale = -opt['scale'] * opt['sfun'](vec_len)

        # delete previous plot
        if self.h is not None:
            try:
                self.h.remove()
            except Exception:
                pass

        if self.mode == 'arrow':
            color = kwargs.get('color', self.color)
            self.color = color
            # scale vectors
            scaled_vec = vec * scale[:, np.newaxis]

            ax = plt.gca()
            if not hasattr(ax, 'get_zlim'):
                fig = plt.gcf()
                ax = fig.add_subplot(111, projection = '3d')

            self.h = ax.quiver(
                self.pos[:, 0], self.pos[:, 1], self.pos[:, 2],
                scaled_vec[:, 0], scaled_vec[:, 1], scaled_vec[:, 2],
                color = color)

        elif self.mode == 'cone':
            self.h = _cone_plot(self.pos, vec, vec_len, scale)

    def isbase(self, pos: object) -> bool:
        """
        MATLAB: @vecarray/isbase.m

        Check if positions match.
        """
        if not isinstance(pos, np.ndarray):
            return False
        if pos.shape[0] != self.pos.shape[0]:
            return False
        return np.all(pos == self.pos)

    def ispage(self) -> bool:
        """
        MATLAB: @vecarray/ispage.m

        True if vector array is multi-dimensional.
        """
        return self.vec.ndim > 2

    def pagesize(self, vec: Optional[np.ndarray] = None) -> tuple:
        """
        MATLAB: @vecarray/pagesize.m

        Paging size of vector array.
        """
        if vec is None:
            vec = self.vec
        if vec.ndim <= 2:
            return (1,)
        return vec.shape[2:]

    def depends(self, *properties: str) -> bool:
        """
        MATLAB: @vecarray/depends.m

        Check whether object uses one of the given properties.
        """
        if 'ind' in properties and self.ispage():
            return True
        if 'fun' in properties or 'scale' in properties or 'sfun' in properties:
            return True
        return False

    def min_val(self, opt: Dict[str, Any]) -> float:
        """
        MATLAB: @vecarray/min.m
        """
        vec = self.get_vec(opt)
        return np.min(np.sqrt(np.sum(np.abs(vec) ** 2, axis = 1)))

    def max_val(self, opt: Dict[str, Any]) -> float:
        """
        MATLAB: @vecarray/max.m
        """
        vec = self.get_vec(opt)
        return np.max(np.sqrt(np.sum(np.abs(vec) ** 2, axis = 1)))

    def __repr__(self) -> str:
        return 'VecArray(n={}, mode={})'.format(
            self.pos.shape[0], self.mode)


def _cone_plot(pos: np.ndarray, vec: np.ndarray,
        vec_len: np.ndarray, scale: np.ndarray) -> object:
    """
    MATLAB: @vecarray/plot.m/coneplot (embedded function)

    Cone plot for vectors at positions using matplotlib.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    # cone geometry: cylindrical profile
    n_sides = 20
    theta_vals = np.linspace(0, 2 * np.pi, n_sides + 1)
    radii = np.array([0.0, 0.6, 0.3, 0.3, 0.0])
    heights = np.array([2.0, 0.0, 0.0, -1.0, -1.0])

    # base cone vertices (n_profile x n_sides+1)
    n_prof = len(radii)
    cone_x = np.outer(radii, np.cos(theta_vals))
    cone_y = np.outer(radii, np.sin(theta_vals))
    cone_z = np.tile(heights[:, np.newaxis], (1, n_sides + 1))

    # flatten to vertex list
    base_verts = np.column_stack([cone_x.ravel(), cone_y.ravel(), cone_z.ravel()])
    n_base = base_verts.shape[0]

    # build faces from quad strips
    base_faces = []
    for i in range(n_prof - 1):
        for j in range(n_sides):
            v0 = i * (n_sides + 1) + j
            v1 = i * (n_sides + 1) + j + 1
            v2 = (i + 1) * (n_sides + 1) + j + 1
            v3 = (i + 1) * (n_sides + 1) + j
            base_faces.append([v0, v1, v2, v3])
    base_faces = np.array(base_faces)

    # rotation matrices for each vector
    all_polys = []
    all_colors = []

    for i in range(pos.shape[0]):
        if vec_len[i] < 1e-30:
            continue

        # spherical coordinates
        phi = np.arctan2(vec[i, 1], vec[i, 0])
        r_xy = np.sqrt(vec[i, 0] ** 2 + vec[i, 1] ** 2)
        theta_val = np.arctan2(r_xy, vec[i, 2])

        # rotation matrices Ry(-theta) * Rz(-phi)
        cp, sp = np.cos(phi), np.sin(phi)
        ct, st = np.cos(theta_val), np.sin(theta_val)

        # combined rotation
        rot = np.array([
            [ct * cp, -sp, st * cp],
            [ct * sp, cp, st * sp],
            [-st, 0.0, ct]])

        # scale and rotate
        v = scale[i] * (base_verts @ rot.T) + pos[i]

        # build polygons
        for face in base_faces:
            all_polys.append(v[face])
            all_colors.append(vec_len[i])

    if len(all_polys) == 0:
        return None

    ax = plt.gca()
    if not hasattr(ax, 'get_zlim'):
        fig = plt.gcf()
        ax = fig.add_subplot(111, projection = '3d')

    pc = Poly3DCollection(all_polys)
    pc.set_array(np.array(all_colors))
    pc.set_edgecolor('none')
    ax.add_collection3d(pc)

    return pc
