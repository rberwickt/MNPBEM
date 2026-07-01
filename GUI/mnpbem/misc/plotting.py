"""
Plotting utilities for MNPBEM.

MATLAB: Misc/arrowplot.m, Misc/coneplot.m, Misc/coneplot2.m,
        Misc/mycolormap.m, Misc/particlecursor.m
"""

import numpy as np
from typing import Any, Optional, Tuple, Union

from .bemplot import BemPlot
from .math_utils import vec_norm


def arrowplot(pos: np.ndarray, vec: np.ndarray,
        **kwargs: Any) -> BemPlot:
    """
    MATLAB: Misc/arrowplot.m

    Plot vectors at given positions using arrows.

    Parameters
    ----------
    pos : ndarray, shape (n, 3)
        Positions where arrows are plotted
    vec : ndarray, shape (n, 3)
        Vectors to be plotted
    **kwargs
        fun, scale, sfun - passed to BemPlot

    Returns
    -------
    bp : BemPlot
    """
    bp = BemPlot.get(**kwargs)
    bp.plotarrow(pos, vec, **kwargs)
    return bp


def coneplot(pos: np.ndarray, vec: np.ndarray,
        **kwargs: Any) -> BemPlot:
    """
    MATLAB: Misc/coneplot.m

    Plot vectors at given positions using cones via BemPlot.

    Parameters
    ----------
    pos : ndarray, shape (n, 3)
        Positions where cones are plotted
    vec : ndarray, shape (n, 3)
        Vectors to be plotted
    **kwargs
        fun, scale, sfun - passed to BemPlot

    Returns
    -------
    bp : BemPlot
    """
    bp = BemPlot.get(**kwargs)
    bp.plotcone(pos, vec, **kwargs)
    return bp


def coneplot2(pos: np.ndarray, vec: np.ndarray,
        **kwargs: Any) -> object:
    """
    MATLAB: Misc/coneplot2.m

    Plot vectors at given positions using cones.

    Parameters
    ----------
    pos : ndarray, shape (n, 3)
        Positions where cones are plotted
    vec : ndarray, shape (n, 3)
        Vectors to be plotted
    **kwargs
        scale, sfun - scaling options

    Returns
    -------
    h : plot handle
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from .vecarray import _cone_plot

    scale_val = kwargs.get('scale', 1.0)
    sfun = kwargs.get('sfun', lambda x: x)

    # vector length
    v_len = vec_norm(vec)
    max_len = np.max(v_len) if np.max(v_len) > 0 else 1.0
    scale = scale_val * sfun(v_len / max_len)

    h = _cone_plot(pos, vec, v_len, scale)

    ax = plt.gca()
    ax.set_axis_off()

    return h


def mycolormap(key: str, n: int = 100) -> np.ndarray:
    """
    MATLAB: Misc/mycolormap.m

    Load custom colormaps.

    Parameters
    ----------
    key : str
        Colormap name: 'std:1'-'std:10', 'cen:1'-'cen:5', 'con:1'-'con:7'
    n : int
        Number of colors to interpolate

    Returns
    -------
    cmap : ndarray, shape (n, 3)
        RGB colormap values in [0, 1]
    """
    colormaps = {
        'std:1': np.array([
            [0, 0, 0], [69, 0, 0], [139, 0, 0], [197, 82, 0],
            [247, 154, 0], [255, 204, 0], [255, 249, 0], [255, 255, 111],
            [255, 255, 239]], dtype = float),
        'std:2': np.array([
            [0, 0, 0], [47, 47, 47], [95, 95, 95], [142, 142, 142],
            [184, 184, 184], [204, 204, 204], [220, 220, 220],
            [236, 236, 236], [252, 252, 252]], dtype = float),
        'std:3': np.array([
            [139, 0, 0], [255, 0, 0], [255, 165, 0], [255, 255, 0],
            [31, 255, 0], [0, 31, 223], [0, 0, 153], [65, 0, 131],
            [217, 113, 224]], dtype = float),
        'std:4': np.array([
            [0, 0, 128], [0, 0, 191], [0, 0, 255], [0, 0, 127], [0, 0, 15],
            [111, 0, 0], [239, 0, 0], [204, 0, 0], [146, 0, 0]], dtype = float),
        'std:5': np.array([
            [0, 0, 128], [0, 0, 191], [0, 0, 255], [127, 127, 127],
            [239, 239, 15], [255, 143, 0], [255, 15, 0], [204, 0, 0],
            [146, 0, 0]], dtype = float),
        'std:6': np.array([
            [0, 100, 0], [0, 177, 0], [0, 255, 0], [0, 127, 0], [0, 15, 0],
            [111, 0, 0], [239, 0, 0], [204, 0, 0], [146, 0, 0]], dtype = float),
        'std:7': np.array([
            [0, 100, 0], [0, 177, 0], [0, 255, 0], [127, 255, 0],
            [239, 255, 0], [255, 143, 0], [255, 15, 0], [204, 0, 0],
            [146, 0, 0]], dtype = float),
        'std:8': np.array([
            [0, 0, 139], [0, 0, 197], [0, 0, 255], [0, 0, 127], [0, 0, 15],
            [0, 111, 0], [0, 239, 0], [0, 187, 0], [0, 109, 0]], dtype = float),
        'std:9': np.array([
            [0, 0, 139], [0, 0, 197], [0, 0, 255], [127, 127, 127],
            [239, 239, 15], [143, 255, 0], [15, 255, 0], [0, 187, 0],
            [0, 109, 0]], dtype = float),
        'std:10': np.array([
            [0, 0, 255], [0, 50, 127], [0, 100, 0], [0, 177, 0],
            [0, 245, 0], [111, 143, 0], [239, 15, 0], [204, 0, 0],
            [146, 0, 0]], dtype = float),
        'cen:1': np.array([
            [179, 88, 6], [224, 130, 20], [253, 184, 99], [254, 224, 182],
            [247, 247, 247], [219, 221, 236], [182, 176, 213], [134, 122, 176],
            [89, 48, 140]], dtype = float),
        'cen:2': np.array([
            [140, 81, 10], [191, 129, 45], [223, 194, 125], [246, 232, 195],
            [245, 245, 245], [204, 235, 231], [136, 208, 197], [62, 157, 149],
            [7, 108, 100]], dtype = float),
        'cen:3': np.array([
            [178, 24, 43], [214, 96, 77], [244, 165, 130], [253, 219, 199],
            [247, 247, 247], [213, 231, 240], [153, 201, 224], [76, 153, 198],
            [37, 107, 174]], dtype = float),
        'cen:4': np.array([
            [178, 24, 43], [214, 96, 77], [244, 165, 130], [253, 219, 199],
            [255, 255, 255], [227, 227, 227], [190, 190, 190], [141, 141, 141],
            [84, 84, 84]], dtype = float),
        'cen:5': np.array([
            [215, 48, 39], [244, 109, 67], [253, 174, 97], [254, 224, 139],
            [255, 255, 223], [221, 241, 149], [172, 219, 110], [110, 192, 99],
            [35, 156, 82]], dtype = float),
        'con:1': np.array([
            [255, 247, 251], [236, 231, 242], [208, 209, 230], [166, 189, 219],
            [122, 171, 208], [61, 147, 193], [11, 116, 178], [4, 92, 145],
            [2, 60, 94]], dtype = float),
        'con:2': np.array([
            [247, 252, 253], [229, 245, 249], [204, 236, 230], [153, 216, 201],
            [108, 196, 168], [69, 176, 123], [38, 143, 75], [4, 112, 47],
            [0, 73, 29]], dtype = float),
        'con:3': np.array([
            [255, 255, 217], [237, 248, 176], [199, 233, 180], [127, 205, 187],
            [72, 184, 194], [33, 149, 192], [33, 100, 171], [36, 57, 150],
            [11, 31, 95]], dtype = float),
        'con:4': np.array([
            [255, 247, 236], [254, 232, 200], [253, 212, 158], [253, 187, 132],
            [252, 146, 94], [240, 106, 74], [218, 54, 36], [183, 6, 3],
            [133, 0, 0]], dtype = float),
        'con:5': np.array([
            [255, 255, 204], [255, 237, 160], [254, 217, 118], [254, 178, 76],
            [253, 145, 62], [252, 85, 44], [230, 32, 29], [193, 3, 36],
            [177, 0, 38]], dtype = float),
        'con:6': np.array([
            [255, 255, 229], [255, 247, 188], [254, 227, 145], [254, 196, 79],
            [254, 158, 45], [238, 117, 22], [208, 80, 4], [159, 55, 3],
            [108, 38, 5]], dtype = float),
        'con:7': np.array([
            [247, 252, 253], [224, 236, 244], [191, 211, 230], [158, 188, 218],
            [142, 154, 200], [140, 112, 179], [136, 70, 159], [129, 21, 128],
            [83, 1, 81]], dtype = float),
    }

    if key not in colormaps:
        raise ValueError('[error] Colormap <{}> not known!'.format(key))

    raw = colormaps[key]

    # interpolate to n colors
    x_old = np.linspace(0, 1, raw.shape[0])
    x_new = np.linspace(0, 1, n)
    cmap = np.empty((n, 3))
    for ch in range(3):
        cmap[:, ch] = np.interp(x_new, x_old, raw[:, ch])

    cmap = cmap / 255.0

    return cmap


def particlecursor(p: object) -> None:
    """
    MATLAB: Misc/particlecursor.m

    Interactive face selection on particle surface.
    This is a placeholder for the MATLAB interactive tool.
    In Python, use matplotlib pick events instead.
    """
    print('[info] particlecursor is not available in Python.')
    print('[info] Use matplotlib pick events for interactive selection.')
