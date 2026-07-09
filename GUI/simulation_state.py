# stores all of the data so that it can be easily passed between screens/functions
from dataclasses import dataclass, field
from typing import Any, Optional, Callable
from .mnpbem.misc import EV2NM
#import numpy as np
# NOTE: add validation of simulation state before running the simulation (ex. environment material is set)
@dataclass
class SimulationState:
    # filename (str) => A callable that takes a float (enei) 
    #        and returns a tuple of (complex_eps, float_k)
    loaded_dielectrics: dict[str, Callable[[float], tuple[complex, float]]] = field(default_factory=dict) # all material functions
    #loaded_calculations: dict[str, Callable] = field(default_factory=dict) # unsure what the callable will be for this
    raw_results: Optional[Any] = None                                   # Simulation output (Sigma)

    solver: str = "Retarded"

    # Energy Range Settings ===========================================
    energy_in_nm: bool = True
    energy_min: float = 100.0
    energy_max: float = 400.0
    energy_steps: int = 10
    rel_cutoff: int = 3 # higher is slower, NOTE: not changed by the user right now

    # Structure and Material Settings ===================================
    structure: str = "Sphere" # shape
    use_substrate: bool = False
    environment_material: Callable[[float], tuple[complex, float]] = None
    substrate_material: Callable[[float], tuple[complex, float]] = None

    mesh_density: int = 3 # nm density, sim code said to not let end user change this, but we can leave it in for now
    refine: int = 2 # same as mesh density
    interp: str = "curv" # was present in many structure configs, unsure of use so leaving it in the state for edge cases

    materials: list[Callable[[float], tuple[complex, float]]] = field(default_factory=list) # should just be one until we have shells/dimers
    # but it never hurts to future proof

        # Shape Specific Settings =====
            # sphere
    diameter: float = 50.0 # shared with rod
            # rod
    horizontal: bool = False # only true when rod(?) - code was unclear
    height: float = 80.0
            # cube
    size: float = 30.0
    cube_e: float = 0.25 # TODO: look into meaning of this
    n_per_edge: int = 16 # cube equivalent of mesh density (bears same restriction worries)
    

    # Excitation Setttings ============================================
    excitation_source: str = "Plane Wave"

        # Plane Wave Settings =========
    # deprecated for now (was in the MATLAB, but not in pymnpbem-sim)
    #polarization: str = "p"
    #polarization_angle: int = 15 # (degrees)
            # Polarization Vectors and Direction
    pol_x: int = 1
    pol_y: int = 0
    pol_z: int = 0

    pol_dir_x: int = 0
    pol_dir_y: int = 0
    pol_dir_z: int = 1

        # Electron Beam Settings =======
    beam_energy: float = 200e3 # (eV)
    beam_width: int = 0.5 # (nm)
    impact_parameter: float = 5.0 # (nm)

        # Dipole Settings ==============
    dipole_moment_x: int = 1 
    dipole_moment_y: int = 0 
    dipole_moment_z: int = 0 

    dipole_pos_x: int = 0 # (nm)
    dipole_pos_y: int = 0 # (nm)
    dipole_pos_z: int = 20 # (nm)

    def to_dict(self) -> dict:
        """Convert GUI state to dictionary for PyMNPBEM_simulation"""
        simulation = {}
        # solver -> simulation_type
        simulation['type'] = "ret"
        if self.solver == "Quasistatic":
            simulation['type'] = "stat"
        #elif self.solver == "Iterative Retarded":
            #simulation['simulation_type'] = "ret_iter" # might mess with the compute config (maybe disable for now?)
            

        # excitation
        if self.excitation_source == "Plane Wave":
            # polarizations, propagation_dirs
            simulation['excitation'] = "planewave"
            simulation['polarizations'] = [[self.pol_x, self.pol_y, self.pol_z]]
            simulation['propagation_dirs'] = [[self.pol_dir_x, self.pol_dir_y, self.pol_dir_z]]
        elif self.excitation_source == "Electron Beam":
            # impact_parameter, beam_energy, beam_width
            simulation['excitation'] = "eels"
            simulation['impact_parameter'] = self.impact_parameter
            simulation['beam_energy'] = self.beam_energy
            simulation['beam_width'] = self.beam_width
        elif self.excitation_source == "Dipole":
            #dipole_position, dipole_moment
            simulation['excitation'] = "dipole"
        
        # wavelength (range)
        nm_min = self.energy_min
        nm_max = self.energy_max
        if self.energy_in_nm is False:
            nm_min /= EV2NM
            nm_max /= EV2NM
        simulation["enei_min"] = nm_min
        simulation['enei_max'] = nm_max
        simulation['n_wavelengths'] = self.energy_steps

        # user will select calculations in post-processing
        simulation['calculate_cross_sections'] = False
        simulation['calculate_fields'] = False
        
        # example TEMP TEMP TEMP TEMP TEMP TEMP TEMP
        cfg = {
            'structure': {
                'type': 'sphere',
                'diameter': 20.0,
                'mesh_density': 144,
                'refine': 2,
                'interp': 'curv'
            },
            'simulation': {
                'type': 'ret',
                'excitation': 'planewave',
                'enei_min': 450,
                'enei_max': 750,
                'n_wavelengths': 2,
                'polarizations': [[1,0,0]],
                'propagation_dirs': [[0,0,-1]],
                'calculate_cross_sections': True,
                'calculate_fields': False
            },
            'materials': {
                'medium': 'vacuum',
                'particle': 'gold'
            },
            'compute': {
                'n_workers': 1,
                'n_threads': 4,
                'n_gpus_per_worker': 0,
                'multi_node': False
            },
            'output': {
                'dir': './tmp',   # can be a temp dir or wherever
                'name': 'sim_run',
                'formats': [], # disable automatic NPZ/JSON saving
                'save_plots': False
            }
        }
        """substrate example piece (under materials)
        'use_substrate': True,
        'substrate': {
            'material': 'glass',   # name resolved via refractive_index_paths or treated as builtin
            'gap': 0.001
        },
        'refractive_index_paths': {
            'glass': {'type': 'constant', 'epsilon': 2.25}
        }"""
        # TODO: create function to wrap in substrate layer (might be able to reuse sim code?)
        return cfg