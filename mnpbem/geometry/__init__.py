"""
Geometry and mesh generation module.

Classes:
- Particle: Basic particle with triangular mesh
- ComParticle: Compound particle with multiple materials
- ComParticleMirror: Compound particle with mirror symmetry
- CompStructMirror: Structure for compound with mirror symmetry
- Point: Single collection of points in space
- ComPoint: Compound of points in a dielectric environment
- EdgeProfile: Edge rounding profile for nanostructures
- Polygon: 2D polygon for mesh generation
- Polygon3: 3D polygon for extrusion of particles

Functions:
- trisphere: Generate triangulated sphere
- trirod: Generate triangulated nanorod
- tricube: Generate triangulated nanocube with rounded edges
- tritorus: Generate triangulated torus
- trispheresegment: Generate triangulated sphere segment
- trispherescale: Scale a sphere to create ellipsoid
- tripolygon: Generate 3D particle from 2D polygon + edge profile
- fvgrid: Convert parametric surface to face-vertex structure
- connect: Compute connectivity between particles
"""

from .particle import Particle
from .compound import Compound
from .comparticle import ComParticle
from .comparticle_mirror import ComParticleMirror, CompStructMirror
from .compoint import Point, ComPoint
from .mesh_generators import (
    trisphere,
    trirod,
    tricube,
    tritorus,
    trispheresegment,
    trispherescale,
    tripolygon,
    fvgrid,
)
from .edgeprofile import EdgeProfile
from .polygon import Polygon
from .polygon3 import Polygon3
from .connect import connect
from .layer_structure import LayerStructure

__all__ = [
    "Particle",
    "Compound",
    "ComParticle",
    "ComParticleMirror",
    "CompStructMirror",
    "Point",
    "ComPoint",
    "EdgeProfile",
    "Polygon",
    "Polygon3",
    "trisphere",
    "trirod",
    "tricube",
    "tritorus",
    "trispheresegment",
    "trispherescale",
    "tripolygon",
    "fvgrid",
    "connect",
    "LayerStructure",
]
