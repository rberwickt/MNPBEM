# stores all of the data so that it can be easily passed between screens/functions
from dataclasses import dataclass, field
from typing import Any, Optional, Callable, Dict

import traceback
import ctypes
from pathlib import Path
import threading
import shutil
import copy


@dataclass
class SimulationState:
    # UI-only: names shown in dropdowns
    loaded_dielectrics: list[str] = field(default_factory=list)

    # authoritative material source for simulation
    material_descriptors: dict[str, dict[str, Any]] = field(default_factory=dict)

    # loaded_calculations should be addresed at some point

    raw_results: Optional[Any] = None               # Simulation output (Sigma)

    solver: str = "Retarded"
    gpu_precision: str = "fp64" # will either be 'fp64' or 'fp32', only used if GPU is enabled

    calc_fields: bool = True
    calc_cross_sections: bool = True

    # Runtime environment setup (must be configured before mnpbem import)
    env_n_workers: int = 1
    env_n_threads: int = 4
    env_n_gpus_per_worker: int = 0

    # Energy Range Settings ===========================================
    energy_in_nm: bool = True
    energy_min: float = 300.0
    energy_max: float = 1200.0
    energy_steps: int = 100
    rel_cutoff: int = 3 # higher is slower, NOTE: recommended not to be changed by user, but can for testing

    # Field grid sampling (rectangular). Volumetric output requires
    # non-collapsed sampling along all 3 axes (especially z).
    field_x_min: float = -150.0
    field_x_max: float = 150.0
    field_y_min: float = -150.0
    field_y_max: float = 150.0
    field_z_min: float = 0.0
    field_z_max: float = 0.0
    field_nx: int = 50
    field_ny: int = 50
    field_nz: int = 1

    # Structure and Material Settings ===================================
    structure: str = "Sphere" # shape
    use_substrate: bool = False
    materials: list[str] = field(default_factory=list)            # particle material names (core->shell)
    environment_material: Optional[str] = None                    # 
    substrate_material: Optional[str] = None
    substrate_gap: float = 0.001 # (nm)

    sphere_n_verts: int = 256 # sphere discretization target (trisphere vertex count)
    mesh_element_size_nm: float = 5.0 # rod/cube element size in nm (smaller = finer mesh / more faces)
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
    cube_e: float = 0.25 
    n_per_edge: int = 16 # cube equivalent of mesh density (seems to be overridden by mesh density?)
    

    # Excitation Setttings ============================================
    excitation_source: str = "Plane Wave"

        # Plane Wave Settings =========
    # deprecated for now (was in the MATLAB, but not in pymnpbem-sim)
    #polarization: str = "p"
    #polarization_angle: int = 15 # (degrees)
            # Polarization Vectors and Direction
    plane_wave_polarizations: list[list[int]] = field(
        default_factory = lambda: [[1, 0, 0]]
    )
    plane_wave_propagation_dirs: list[list[int]] = field(
        default_factory = lambda: [[0, 0, 1]]
    )


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

    def get_plane_wave_polarizations(self) -> list[list[int]]:
        pols = getattr(self, "plane_wave_polarizations", None)
        if isinstance(pols, list) and len(pols) > 0:
            normalized = []
            for pol in pols:
                if isinstance(pol, (list, tuple)) and len(pol) >= 3:
                    normalized.append([
                        int(pol[0]),
                        int(pol[1]),
                        int(pol[2]),
                    ])
            if len(normalized) > 0:
                return normalized

        return [[int(self.pol_x), int(self.pol_y), int(self.pol_z)]]

    def get_plane_wave_propagation_dirs(self) -> list[list[int]]:
        dirs = getattr(self, "plane_wave_propagation_dirs", None)
        if isinstance(dirs, list) and len(dirs) > 0:
            normalized = []
            for direction in dirs:
                if isinstance(direction, (list, tuple)) and len(direction) >= 3:
                    normalized.append([
                        int(direction[0]),
                        int(direction[1]),
                        int(direction[2]),
                    ])
            if len(normalized) > 0:
                return normalized

        return [[int(self.pol_dir_x), int(self.pol_dir_y), int(self.pol_dir_z)]]

    def set_plane_wave_polarizations(self, polarizations: list[list[int]]) -> None:
        normalized = []
        for pol in polarizations:
            if isinstance(pol, (list, tuple)) and len(pol) >= 3:
                normalized.append([
                    int(pol[0]),
                    int(pol[1]),
                    int(pol[2]),
                ])

        if len(normalized) == 0:
            normalized = [[1, 0, 0]]

        self.plane_wave_polarizations = normalized

        first = normalized[0]
        self.pol_x = int(first[0])
        self.pol_y = int(first[1])
        self.pol_z = int(first[2])

    def set_plane_wave_propagation_dirs(self, propagation_dirs: list[list[int]]) -> None:
        normalized = []
        for direction in propagation_dirs:
            if isinstance(direction, (list, tuple)) and len(direction) >= 3:
                normalized.append([
                    int(direction[0]),
                    int(direction[1]),
                    int(direction[2]),
                ])

        if len(normalized) == 0:
            normalized = [[0, 0, 1]]

        self.plane_wave_propagation_dirs = normalized

        first = normalized[0]
        self.pol_dir_x = int(first[0])
        self.pol_dir_y = int(first[1])
        self.pol_dir_z = int(first[2])

    def _plane_wave_pairs(self) -> list[tuple[list[int], list[int]]]:
        polarizations = self.get_plane_wave_polarizations()
        propagation_dirs = self.get_plane_wave_propagation_dirs()

        if len(propagation_dirs) < len(polarizations):
            last_dir = propagation_dirs[-1] if len(propagation_dirs) > 0 else [0, 0, 1]
            propagation_dirs = propagation_dirs + [list(last_dir) for _ in range(len(polarizations) - len(propagation_dirs))]
        elif len(propagation_dirs) > len(polarizations):
            propagation_dirs = propagation_dirs[:len(polarizations)]

        return list(zip(polarizations, propagation_dirs))


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

        if self.field_x_min >= self.field_x_max:
            return False, "Field grid x_min must be less than x_max"

        if self.field_y_min >= self.field_y_max:
            return False, "Field grid y_min must be less than y_max"

        if self.field_nx < 1 or self.field_ny < 1 or self.field_nz < 1:
            return False, "Field grid points (nx, ny, nz) must be at least 1"

        if self.field_nz > 1 and self.field_z_min >= self.field_z_max:
            return False, "Field grid z_min must be less than z_max when nz > 1"

        if self.excitation_source == "Plane Wave" and len(self.get_plane_wave_polarizations()) < 1:
            return False, "At least one plane-wave polarization must be configured"

        if self.excitation_source == "Plane Wave":
            for idx, (polarization, direction) in enumerate(self._plane_wave_pairs(), start = 1):
                if polarization == [0, 0, 0]:
                    return False, "Plane-wave polarization {} cannot be the zero vector".format(idx)
                if direction == [0, 0, 0]:
                    return False, "Plane-wave propagation direction {} cannot be the zero vector".format(idx)
                if sum(int(a) * int(b) for a, b in zip(polarization, direction)) != 0:
                    return False, (
                        "Plane-wave polarization {} must be orthogonal to its propagation direction"
                        .format(idx)
                    )

        if self.excitation_source == "Dipole":
            dipole_moment = [
                float(self.dipole_moment_x),
                float(self.dipole_moment_y),
                float(self.dipole_moment_z)]

            if dipole_moment == [0.0, 0.0, 0.0]:
                return False, "Dipole moment cannot be the zero vector"

            # GUI dipole path targets embedding medium (medium index 1).
            # Prevent the common crash case where the source point is placed
            # inside the particle, yielding an empty ComPoint group.
            px = float(self.dipole_pos_x)
            py = float(self.dipole_pos_y)
            pz = float(self.dipole_pos_z)
            s_type = str(self.structure).lower()

            if "sphere" in s_type:
                radius = 0.5 * float(self.diameter)
                if (px * px + py * py + pz * pz) < (radius * radius):
                    return False, (
                        "Dipole position is inside the sphere. Move it outside "
                        "the particle surface for GUI dipole runs."
                    )
            elif "cube" in s_type:
                half = 0.5 * float(self.size)
                if (abs(px) < half) and (abs(py) < half) and (abs(pz) < half):
                    return False, (
                        "Dipole position is inside the cube. Move it outside "
                        "the particle surface for GUI dipole runs."
                    )
            elif "rod" in s_type:
                radius = 0.5 * float(self.diameter)
                half_h = 0.5 * float(self.height)
                if bool(self.horizontal):
                    inside_core = ((py * py + pz * pz) < (radius * radius)) and (abs(px) < half_h)
                else:
                    inside_core = ((px * px + py * py) < (radius * radius)) and (abs(pz) < half_h)
                if inside_core:
                    return False, (
                        "Dipole position is inside the rod. Move it outside "
                        "the particle surface for GUI dipole runs."
                    )
        
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
        into the runner below.
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
        #elif "ellipsoid" in s_type:
            #structure_type = "ellipsoid"
        else:
            # fallback: use lowercase raw value
            structure_type = s_type

        from mnpbem.misc import EV2NM
        nm_min = float(self.energy_min)
        nm_max = float(self.energy_max)
        if not self.energy_in_nm:
            # convert to nm with EV2NM helper
            nm_min = float(nm_min / EV2NM)
            nm_max = float(nm_max / EV2NM)

        # material resolution
        particle_name = None
        if len(self.materials) > 0:
            particle_name = self.materials[0]

        medium_name = self.environment_material or "vacuum"

        # build structure block
        struct_block: dict[str, Any] = {"type": structure_type, "refine": int(self.refine), "interp": self.interp}
        # shape-specific params
        if structure_type == "sphere":
            struct_block["diameter"] = float(self.diameter)
            struct_block["n_verts"] = int(self.sphere_n_verts)
        elif structure_type == "rod":
            struct_block["diameter"] = float(self.diameter)
            struct_block["height"] = float(self.height)
            struct_block["horizontal"] = bool(self.horizontal)
            struct_block["mesh_density"] = float(self.mesh_element_size_nm)
        elif structure_type == "cube":
            struct_block["size"] = float(self.size)
            struct_block["n_per_edge"] = int(self.n_per_edge)
            struct_block["mesh_density"] = float(self.mesh_element_size_nm)
        else:
            # generic
            struct_block["diameter"] = float(self.diameter)
            struct_block["mesh_density"] = float(self.mesh_element_size_nm)

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
            "calculate_cross_sections": self.calc_cross_sections,
            "calculate_fields": self.excitation_source == "Plane Wave" and self.calc_fields,  # only calculate fields for plane wave excitation (only one supported out of the three)
            "interp": self.interp,
            "relcutoff": int(self.rel_cutoff),
            "grid": {
                "type": "rectangular",
                "x_range": [float(self.field_x_min), float(self.field_x_max)],
                "y_range": [float(self.field_y_min), float(self.field_y_max)],
                "z_range": [float(self.field_z_min), float(self.field_z_max)],
                "n_points": [int(self.field_nx), int(self.field_ny), int(self.field_nz)]
            }
        }

        plane_wave_pairs = self._plane_wave_pairs()
        plane_wave_polarizations = [list(pol) for pol, _direction in plane_wave_pairs]
        propagation_dirs = [list(direction) for _pol, direction in plane_wave_pairs]

        # excitation-specific parameters
        if self.excitation_source == "Plane Wave":
            sim_config["excitation"] = "planewave"
            sim_config["polarizations"] = plane_wave_polarizations
            sim_config["propagation_dirs"] = propagation_dirs
        elif self.excitation_source == "Electron Beam":
            sim_config["excitation"] = "eels"
            sim_config["impact_parameter"] = float(self.impact_parameter)
            sim_config["beam_energy"] = float(self.beam_energy)
            sim_config["beam_width"] = float(self.beam_width)
        elif self.excitation_source == "Dipole":
            sim_config["excitation"] = "dipole"
            dipole_position = [
                float(self.dipole_pos_x),
                float(self.dipole_pos_y),
                float(self.dipole_pos_z)]
            dipole_moment = [
                float(self.dipole_moment_x),
                float(self.dipole_moment_y),
                float(self.dipole_moment_z)]

            # Keep both formats for compatibility across simulation runners:
            # - dipole_ret/dipole_stat read simulation.dipole.{position,orientation,medium}
            # - *_layer/*_mirror dipole runners read flat dipole_position/moment keys.
            sim_config["dipole"] = {
                "position": dipole_position,
                "orientation": dipole_moment,
                "medium": 1
            }
            sim_config["dipole_position"] = dipole_position
            sim_config["dipole_moment"] = dipole_moment
        else:
            # default to planewave
            sim_config["excitation"] = "planewave"
            sim_config["polarizations"] = plane_wave_polarizations
            sim_config["propagation_dirs"] = propagation_dirs

        cfg = {
            "structure": struct_block,
            "simulation": sim_config,
            "materials": {
                "medium": medium_name,
                "materials": [particle_name] if particle_name is not None else [],
                "refractive_index_paths": self.material_descriptors
            },
            "compute": {
                "n_workers": max(1, int(self.env_n_workers)),
                "n_threads": max(1, int(self.env_n_threads)),
                "n_gpus_per_worker": max(0, int(self.env_n_gpus_per_worker)),
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
            cfg["materials"]["substrate"] = {"material": self.substrate_material, "gap": self.substrate_gap}

        if self.env_n_gpus_per_worker > 0: # add the GPU precision setting if GPU is enabled
            cfg["compute"]["gpu_precision"] = str(self.gpu_precision).lower()
            
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
        
        NOTE: Environment must be set up BEFORE this is called (done in gui_main.py).
        This function can safely be called from a background thread.
        """
        def _report(msg: str):
            if progress_callback:
                try:
                    progress_callback(msg)
                except Exception:
                    pass

        # Build config dict
        cfg = self.to_dict(output_dir=output_dir, output_name=output_name)

        sim_cfg = cfg.get("simulation", {}) if isinstance(cfg, dict) else {}

        # calculate_spectrum defaults to True in backend unless explicitly false.
        wants_fields = bool(sim_cfg.get("calculate_fields", False))
        wants_spectrum = bool(sim_cfg.get("calculate_spectrum", True))
        if "calculate_cross_sections" in sim_cfg:
            wants_spectrum = bool(sim_cfg.get("calculate_cross_sections", wants_spectrum))

        def _has_spectrum_payload(res: Any) -> bool:
            if not isinstance(res, dict):
                return False
            required = ("wavelength", "ext", "sca", "abs")
            return all(k in res for k in required)

        def _has_field_payload(res: Any) -> bool:
            if not isinstance(res, dict):
                return False
            required = ("wavelength", "pos", "e")
            return all(k in res for k in required)

        def _merge_result_payload(base: dict, extra: dict) -> dict:
            merged = dict(base)
            for key in ("wavelength", "pos", "e", "h", "grid_shape", "inout"):
                if key in extra:
                    merged[key] = extra[key]
            if "n_pol" in extra and "n_pol" not in merged:
                merged["n_pol"] = extra["n_pol"]
            merged["kind"] = "spectrum_field"
            return merged

        # apply compute overrides if provided
        if compute_overrides:
            cfg.setdefault("compute", {})
            for k, v in compute_overrides.items():
                cfg["compute"][k] = v

        # optionally override wavelength count
        if n_wavelengths_override is not None:
            cfg.setdefault("simulation", {})
            cfg["simulation"]["n_wavelengths"] = int(n_wavelengths_override)

        # Import pymnpbem_simulation helpers (environment already set up at GUI startup)
        try:
            _report("Preparing simulation")
            from pymnpbem_simulation.config import apply_defaults, validate_config
            from pymnpbem_simulation.util import print_info, ensure_dir

            # ensure compute block defaulting, then apply defaults
            cfg = apply_defaults(cfg)
            validate_config(cfg)

            _report("Building structure")
            from pymnpbem_simulation.structures import build_structure
            from pymnpbem_simulation.dispatch import dispatch_single_node
            from pymnpbem_simulation.io import save_spectrum, save_field, save_run_metadata
            import numpy as np
            import time

            output_path = Path(cfg["output"]["dir"]) / cfg["output"]["name"]
            ensure_dir(str(output_path))

            # Pre-run sigma cache clear (GUI policy): start each run from
            # a clean sigma cache to avoid stale-mode reuse across runs
            # (e.g. quasistatic cache accidentally consumed by retarded
            # field evaluation in the same output folder).
            try:
                from pymnpbem_simulation import sigma_cache as _sc

                sigma_root = Path(_sc.sigma_dir(str(output_path)))
                if sigma_root.exists():
                    shutil.rmtree(sigma_root, ignore_errors=True)
                    _report(
                        "Cleared pre-run sigma cache: {}".format(sigma_root)
                    )
            except Exception:
                pass
            
            # build structure (returns p, epstab, nfaces)
            p, epstab, nfaces = build_structure(cfg["structure"], cfg.get("materials", {}))
            # build wavelength grid using the same logic as the CLI helper
            sim = cfg["simulation"]
            e_min = float(sim["enei_min"])
            e_max = float(sim["enei_max"])
            n_wl = int(sim["n_wavelengths"])
            enei = np.linspace(e_min, e_max, n_wl)

            # For combined spectrum+field requests, run spectrum first.
            # dispatch_single_node routes calculate_fields=True to FieldCalculator,
            # which is much heavier for layered/substrate cases if sigma cache
            # is cold. Running spectrum first warms sigma cache and avoids the
            # expensive field fallback doing full BEM solves per wavelength.
            cfg_dispatch = cfg
            if wants_fields and wants_spectrum:
                _report("Dispatching spectrum pass")
                cfg_dispatch = copy.deepcopy(cfg)
                cfg_dispatch.setdefault("simulation", {})
                cfg_dispatch["simulation"]["calculate_spectrum"] = True
                cfg_dispatch["simulation"]["calculate_cross_sections"] = True
                cfg_dispatch["simulation"]["calculate_fields"] = False
            else:
                _report("Dispatching simulation")

            t0 = time.time()
            result = dispatch_single_node(cfg_dispatch, p, epstab, enei)

            if wants_fields and (not _has_field_payload(result)):
                _report("Running field follow-up pass for post-processing")
                cfg_field = copy.deepcopy(cfg)
                cfg_field.setdefault("simulation", {})
                cfg_field["simulation"]["calculate_spectrum"] = False
                cfg_field["simulation"]["calculate_fields"] = True
                # Keep sigma cache enabled so FieldCalculator can load the
                # spectrum-pass sigma instead of recomputing heavy layer BEM.
                cfg_field["simulation"]["save_sigma_cache"] = True
                field_result = dispatch_single_node(cfg_field, p, epstab, enei)
                result = _merge_result_payload(result, field_result)

            if wants_spectrum and (not _has_spectrum_payload(result)):
                _report("Running spectrum follow-up pass for post-processing")
                cfg_spec = copy.deepcopy(cfg)
                cfg_spec.setdefault("simulation", {})
                cfg_spec["simulation"]["calculate_spectrum"] = True
                cfg_spec["simulation"]["calculate_cross_sections"] = True
                cfg_spec["simulation"]["calculate_fields"] = False
                spec_result = dispatch_single_node(cfg_spec, p, epstab, enei)
                result = _merge_result_payload(spec_result, result)

            total_s = time.time() - t0
            _report(f"Simulation finished in {total_s:.1f}s")

            # store raw results in state for later postprocessing
            self.raw_results = result

            # Optionally save
            if save_outputs:
                _report(f"Saving outputs to {output_path}")
                save_run_metadata(str(output_path), cfg, nfaces)
                if result.get("kind", None) == "field":
                    save_field(str(output_path), result)
                else:
                    save_spectrum(str(output_path), result)

            return result

        except Exception as exc:
            # capture and re-raise
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
        # Always pass explicit thread override so runtime matches GUI request.
        compute_overrides = {"n_threads": max(1, int(n_threads))}
        
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

    def cancel_simulation_thread(self, thread: Optional[threading.Thread]) -> bool:
        """Force-stop the GUI simulation worker thread (best effort).

        This keeps the existing threaded pipeline intact while allowing a
        practical hard-cancel from the dialog.
        """
        if thread is None:
            return False

        if not thread.is_alive():
            return True

        thread_id = thread.ident
        if thread_id is None:
            return False

        try:
            async_exc = ctypes.pythonapi.PyThreadState_SetAsyncExc
            async_exc.argtypes = [ctypes.c_ulong, ctypes.py_object]
            async_exc.restype = ctypes.c_int

            result = async_exc(ctypes.c_ulong(thread_id), ctypes.py_object(SystemExit))
            if result == 0:
                return False
            if result > 1:
                # Revert if CPython reports multiple targets.
                async_exc(ctypes.c_ulong(thread_id), None)
                return False
            return True
        except Exception:
            return False
