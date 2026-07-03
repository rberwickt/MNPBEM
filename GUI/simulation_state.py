# stores all of the data so that it can be easily passed between screens/functions
from dataclasses import dataclass, field
from typing import Any, Optional, Callable
# NOTE: add validation of simulation state before running the simulation (ex. environment material is set)
@dataclass
class SimulationState:
    # filename (str) => A callable that takes a float (enei) 
    #        and returns a tuple of (complex_eps, float_k)
    loaded_dielectrics: dict[str, Callable[[float], tuple[complex, float]]] = field(default_factory=dict) # all material functions
    #loaded_calculations: dict[str, Callable] = field(default_factory=dict) # unsure what the callable will be for this
    raw_results: Optional[Any] = None                                   # Simulation output (Sigma)

    solver: str = "Retarded"

    # Structure (Geometry) Settings ===================================
    structure: str = "Sphere" # not sure how this will work with multiple structures....

    # Material Settings ===============================================
    material_count: int = 0 # should be set to one as soon as the material_settings widget loads
    materials: Optional[Callable[[float], tuple[complex, float]]] = field(default_factory=list)

    # Excitation Setttings ============================================
    excitation_source: str = "Plane Wave"

        # Plane Wave Settings =========
    polarization: str = "p"
    polarization_angle: int = 15 # (degrees)
            # Jones Vectors and direction
    jones_ex: int = 1
    jones_ey: int = 0
    jones_ez: int = 0

    dir_x: int = 0
    dir_y: int = 0
    dir_z: int = 1

        # Electron Beam Settings =======
    kinetic_energy: float = 8e+04 # (eV)
    beam_width: int = 1 # (nm)

        # Dipole Settings ==============
    oscillation_dir: str = "x"
    dipole_x: int = 0 # (nm)
    dipole_y: int = 0 # (nm)
    dipole_z: int = 0 # (nm)