# stores all of the data so that it can be easily passed between screens/functions
from dataclasses import dataclass, field
from typing import Any, Optional

@dataclass
class SimulationState:
    user_functions: Optional[Any] = None                                # User functions for both materials (EpsFun) and post-processing
    dat_tables: dict[str, Any] = field(default_factory=dict)            # For materials
    raw_results: Optional[Any] = None                                   # Simulation output (Sigma)

    # Excitation Setttings ============================================
    excitation_source: str = "Plane Wave"

        # Plane Wave Settings =========
    polarization: str = "p"
    polarization_angle: int = 15 # (degrees)
            # Jones Vectors
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