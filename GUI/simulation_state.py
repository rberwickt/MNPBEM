# stores all of the data so that it can be easily passed between screens/functions
from dataclasses import dataclass, field
from typing import Any, Optional, Callable, Dict
from mnpbem.misc import EV2NM
import traceback
from pathlib import Path
import threading

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
    materials: list[str] = field(default_factory=list)            # particle material names (core->shell)
    environment_material: Optional[str] = None                    # e.g., 'vacuum' or 'water' or '/path/to/file.dat'
    substrate_material: Optional[str] = None

    mesh_density: int = 3 # nm density, sim code said to not let end user change this, but we can leave it in for now
    refine: int = 2 # same as mesh density
    interp: str = "curv" # was present in many structure configs, unsure of use so leaving it in the state for edge cases

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

    

    def validate_state(self) -> tuple[bool, str]:
        """Validate that the simulation state is ready to run.
        
        Returns (is_valid, error_message_or_empty_string)
        """
        if not self.materials:
            return False, "No particle material selected"
        
        if not self.environment_material:
            return False, "Environment material not selected"
        
        if self.energy_min >= self.energy_max:
            return False, "Energy min must be less than energy max"
        
        if self.energy_steps < 1:
            return False, "Energy steps must be at least 1"
        
        return True, ""

    def to_dict(self, output_dir: Optional[str] = None, output_name: str = "sim_run") -> dict:
        """Convert GUI state to a pymnpbem_simulation-compatible config dict.

        This produces a dict with sections:
          - structure
          - simulation
          - materials
          - compute
          - output

        The result is intentionally similar to the YAML structure expected by
        pymnpbem_simulation.config.apply_defaults/validate_config and can be passed
        into the programmatic runner below.
        """
        # map GUI solver to pymnpbem type
        sim_type = "ret"
        if str(self.solver).lower().startswith("quasi") or "quasistatic" in str(self.solver).lower():
            sim_type = "stat"

        # structure type mapping
        s_type = str(self.structure).lower()
        if "sphere" in s_type:
            structure_type = "sphere"
        elif "rod" in s_type:
            structure_type = "rod"
        elif "cube" in s_type:
            structure_type = "cube"
        elif "ellipsoid" in s_type:
            structure_type = "ellipsoid"
        else:
            # fallback: use lowercase raw value
            structure_type = s_type

        # wavelengths: pymnpbem uses nm grid; GUI may provide eV depending on energy_in_nm
        nm_min = float(self.energy_min)
        nm_max = float(self.energy_max)
        if not self.energy_in_nm:
            # energy provided in eV: convert to nm with EV2NM helper
            nm_min = float(nm_min / EV2NM)
            nm_max = float(nm_max / EV2NM)

        # material resolution: try to pick a name (string) for materials where possible
        particle_name = None
        if len(self.materials) > 0:
            particle_name = self.materials[0]

        medium_name = self.environment_material or "vacuum"

        # build structure block
        struct_block: dict[str, Any] = {"type": structure_type, "mesh_density": float(self.mesh_density), "refine": int(self.refine), "interp": self.interp}
        # populate shape-specific params
        if structure_type == "sphere":
            struct_block["diameter"] = float(self.diameter)
        elif structure_type == "rod":
            struct_block["diameter"] = float(self.diameter)
            struct_block["height"] = float(self.height)
            struct_block["horizontal"] = bool(self.horizontal)
        elif structure_type == "cube":
            struct_block["size"] = float(self.size)
            struct_block["n_per_edge"] = int(self.n_per_edge)
        else:
            # generic: include some fields if present
            struct_block["diameter"] = float(self.diameter)

        # Resolve output directory
        if output_dir is None:
            output_dir = str(Path(".") / "tmp")
        else:
            output_dir = str(Path(output_dir))

        # Build base simulation config
        sim_config: dict[str, Any] = {
            "type": sim_type,
            "enei_min": float(nm_min),
            "enei_max": float(nm_max),
            "n_wavelengths": int(self.energy_steps),
            "calculate_cross_sections": True,
            "calculate_fields": False,
            "interp": self.interp,
            "relcutoff": int(self.rel_cutoff)
        }

        # excitation mapping - add excitation-specific parameters
        if self.excitation_source == "Plane Wave":
            sim_config["excitation"] = "planewave"
            sim_config["polarizations"] = [[self.pol_x, self.pol_y, self.pol_z]]
            sim_config["propagation_dirs"] = [[self.pol_dir_x, self.pol_dir_y, self.pol_dir_z]]
        elif self.excitation_source == "Electron Beam":
            sim_config["excitation"] = "eels"
            sim_config["impact_parameter"] = float(self.impact_parameter)
            sim_config["beam_energy"] = float(self.beam_energy)
            sim_config["beam_width"] = float(self.beam_width)
        elif self.excitation_source == "Dipole":
            sim_config["excitation"] = "dipole"
            sim_config["dipole_position"] = [self.dipole_pos_x, self.dipole_pos_y, self.dipole_pos_z]
            sim_config["dipole_moment"] = [self.dipole_moment_x, self.dipole_moment_y, self.dipole_moment_z]
        else:
            # default to planewave
            sim_config["excitation"] = "planewave"
            sim_config["polarizations"] = [[self.pol_x, self.pol_y, self.pol_z]]
            sim_config["propagation_dirs"] = [[self.pol_dir_x, self.pol_dir_y, self.pol_dir_z]]

        cfg = {
            "structure": struct_block,
            "simulation": sim_config,
            "materials": {
                "medium": medium_name,
                "materials": [particle_name] if particle_name is not None else []
            },
            "compute": {
                "n_workers": 1,
                "n_threads": 1,
                "n_gpus_per_worker": 0,
                "multi_node": False
            },
            "output": {
                "dir": output_dir,
                "name": output_name,
                "formats": [],   # disable automatic saving by default (GUI controls saving)
                "save_plots": False
            }
        }

        # substrate support
        if self.use_substrate and (self.substrate_material is not None):
            cfg["materials"]["use_substrate"] = True
            cfg["materials"]["substrate"] = {"material": self.substrate_material, "gap": 0.001}

        return cfg

    def save_config_yaml(self, path: str) -> None:
        """
        Save the current to_dict() result as a YAML snapshot using pymnpbem_simulation.config.save_yaml.
        """
        output_path = Path(path)
        cfg = self.to_dict(
            output_dir=str(output_path.parent),
            output_name=output_path.stem
        )
        try:
            from pymnpbem_simulation.config import save_yaml
            save_yaml(str(output_path), cfg)
        except Exception as exc:
            raise RuntimeError(f"Failed to save YAML config: {exc}")

    def run_simulation(self,
            output_dir: Optional[str] = None,
            output_name: str = "sim_run",
            save_outputs: bool = False,
            n_wavelengths_override: Optional[int] = None,
            compute_overrides: Optional[Dict[str, int]] = None,
            progress_callback: Optional[Callable[[str], None]] = None) -> dict:
        """
        Run pymnpbem_simulation programmatically from the GUI state.

        - output_dir: where to save outputs (if save_outputs True). Defaults to ./tmp.
        - save_outputs: if True, the runner will persist spectrum/field files using pymnpbem IO helpers.
        - compute_overrides: optional dict to override compute.n_workers/n_threads/n_gpus_per_worker
        - progress_callback: optional callable that will be called with simple status messages (str).
        Returns the simulation result dict (same as dispatch_single_node returns).
        
        NOTE: This function sets up environment variables and imports before running.
        It can safely be called from a background thread.
        """
        def _report(msg: str):
            if progress_callback:
                try:
                    progress_callback(msg)
                except Exception:
                    pass

        # Build config dict
        cfg = self.to_dict(output_dir=output_dir, output_name=output_name)

        # apply compute overrides if provided
        if compute_overrides:
            cfg.setdefault("compute", {})
            for k, v in compute_overrides.items():
                cfg["compute"][k] = v

        # optionally override wavelength count
        if n_wavelengths_override is not None:
            cfg.setdefault("simulation", {})
            cfg["simulation"]["n_wavelengths"] = int(n_wavelengths_override)

        # Import the minimal pymnpbem_simulation helpers and run programmatically.
        # Important: set up environment BEFORE importing heavy mnpbem modules.
        try:
            _report("Preparing environment")
            from pymnpbem_simulation.env_setup import assert_pre_import, setup_env
            from pymnpbem_simulation.config import apply_defaults, validate_config
            from pymnpbem_simulation.util import print_info, ensure_dir
            # Ensure env setup hasn't been bypassed
            assert_pre_import()

            # ensure compute block defaulting, then apply defaults
            cfg = apply_defaults(cfg)
            validate_config(cfg)

            n_threads = int(cfg.get("compute", {}).get("n_threads", 1))
            n_gpus = int(cfg.get("compute", {}).get("n_gpus_per_worker", 0))

            _report(f"Setting environment: n_threads={n_threads}, n_gpus={n_gpus}")
            setup_env(n_threads, n_gpus)

            # Now import structure/build + dispatch (after env is set)
            _report("Building structure")
            from pymnpbem_simulation.structures import build_structure
            from pymnpbem_simulation.dispatch import dispatch_single_node
            from pymnpbem_simulation.io import save_spectrum, save_field, save_run_metadata
            import numpy as np
            import time

            output_path = Path(cfg["output"]["dir"]) / cfg["output"]["name"]
            ensure_dir(str(output_path))
            
            # build structure (returns p, epstab, nfaces)
            p, epstab, nfaces = build_structure(cfg["structure"], cfg.get("materials", {}))
            # build wavelength grid using the same logic as the CLI helper (but minimal here)
            sim = cfg["simulation"]
            e_min = float(sim["enei_min"])
            e_max = float(sim["enei_max"])
            n_wl = int(sim["n_wavelengths"])
            enei = np.linspace(e_min, e_max, n_wl)

            _report("Dispatching simulation")
            t0 = time.time()
            result = dispatch_single_node(cfg, p, epstab, enei)
            total_s = time.time() - t0
            _report(f"Simulation finished in {total_s:.1f}s")

            # store raw results in state for later postprocessing
            self.raw_results = result

            # Optionally persist outputs using pymnpbem IO helpers
            if save_outputs:
                _report(f"Saving outputs to {output_path}")
                save_run_metadata(str(output_path), cfg, nfaces)
                if result.get("kind", None) == "field":
                    save_field(str(output_path), result)
                else:
                    save_spectrum(str(output_path), result)

            return result

        except Exception as exc:
            # capture and re-raise (GUI should display a user-friendly message)
            tb = traceback.format_exc()
            _report(f"Simulation failed: {exc}\n{tb}")
            raise

    def run_simulation_threaded(self,
            on_success: Callable[[dict], None],
            on_error: Callable[[Exception], None],
            on_progress: Optional[Callable[[str], None]] = None,
            output_dir: Optional[str] = None,
            output_name: str = "sim_run",
            save_outputs: bool = False,
            n_threads: int = 1) -> threading.Thread:
        """
        Run simulation in a background thread to avoid blocking the GUI.
        
        The simulation itself can use multiple threads via pymnpbem's internal parallelization.
        This threading is only to keep the GUI responsive.
        
        Args:
            on_success: Callback when simulation completes. Called with result dict.
            on_error: Callback when simulation fails. Called with Exception.
            on_progress: Optional callback for progress messages.
            output_dir: Output directory (defaults to ./tmp).
            output_name: Name for output folder.
            save_outputs: Whether to save output files.
            n_threads: Number of threads for the simulation (default 1).
        
        Returns:
            The Thread object (already started). You can call .join() on it if needed.
        """
        # Prepare compute overrides to allow multi-threading
        compute_overrides = None
        if n_threads > 1:
            compute_overrides = {"n_threads": n_threads}
        
        def _worker():
            try:
                result = self.run_simulation(
                    output_dir=output_dir,
                    output_name=output_name,
                    save_outputs=save_outputs,
                    progress_callback=on_progress,
                    compute_overrides=compute_overrides
                )
                on_success(result)
            except Exception as e:
                on_error(e)
        
        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()
        return thread
