"""
Just planning for now: Using PyVista (import pyvista as pv
from pyvistaqt import QtInteractor) to display the structure in 3D

It might be better to have this be a blocking pop-up since it will be hardware accelerated, and thus should probably die when the user closes it
    otherwise it might stick around in memory while the simulation runs which isn't really ideal 
    if it works really well, could try to revamp the matplotlib 3D field plot to be faster

Modeling substrate might be an issue, but we can leave it for now I guess.
"""