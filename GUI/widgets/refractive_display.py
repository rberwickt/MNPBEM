from PySide6.QtWidgets import (QGroupBox, QFormLayout, QHBoxLayout, QCheckBox, QVBoxLayout, QLabel, QPushButton)
from PySide6.QtCore import Qt
import numpy as np
import importlib.util
import sys
from pathlib import Path

from mnpbem.materials import EpsTable
from mnpbem.misc import EV2NM
from ..simulation_state import SimulationState
from .calculation_figure import CalculationFigure
from matplotlib.figure import Figure
from .material_dropdown import MaterialComboBox


# choose the environment and substate options

class RefractiveIndexWidget(QGroupBox):
    def __init__(self, state: SimulationState, parent=None):
        super().__init__("Refractive Index", parent)
        self.state = state
        self.layout = QVBoxLayout(self)

        title_bar = QHBoxLayout()
        self.material_dropdown = MaterialComboBox(state)
        title_bar.addWidget(QLabel("Material:"))
        title_bar.addWidget(self.material_dropdown)

        self.refresh_btn = QPushButton("Refresh Plot")
        self.refresh_btn.clicked.connect(self.update_plot)
        title_bar.addWidget(self.refresh_btn)

        self.layout.addLayout(title_bar)

        self.figure = None # Placeholder for the CalculationFigure

        # lambda because I check the text in the case of refresh anyway
        self.material_dropdown.currentTextChanged.connect(lambda text: self.update_plot())
        self._resolved_material_cache: dict[str, tuple[str, object]] = {}

    def update_plot(self):
        # clear existing plot
        if self.figure is not None:
            self.layout.removeWidget(self.figure)
            self.figure.deleteLater()
            self.figure = None

        # create a new one
        selected_material = self.material_dropdown.currentText()
        if selected_material != "": # default value for empty material dropdown
            new_fig = self.construct_figure()
            self.figure = CalculationFigure(new_fig)
            self.layout.addWidget(self.figure)

    def construct_figure(self) -> Figure:
        selected_material = self.material_dropdown.currentText()
        wl_min = self.state.energy_min
        wl_max = self.state.energy_max
        if self.state.energy_in_nm is False:
            wl_min = EV2NM / wl_min
            wl_max = EV2NM / wl_max

        wavelengths = np.linspace(wl_min, wl_max, self.state.energy_steps) # same as the sim builder, just a bit early

        real, imag = self._get_eps_components(selected_material, wavelengths)

        # figure building
        fig = Figure()
        ax = fig.add_subplot(111)
        ax.plot(wavelengths, real, label="Eps1 (Real)")
        ax.plot(wavelengths, imag, label="Eps2 (Imag)")
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Refractive Index")
        ax.legend()
        ax.set_title(f"Refractive Index for {selected_material}")
        return fig

    def _resolve_material_from_descriptor(self, material_name: str): # changing everything to descriptors comes back to bite me...
        descriptor = self.state.material_descriptors.get(material_name)
        if descriptor is None:
            return None

        if callable(descriptor):
            return descriptor

        if not isinstance(descriptor, dict):
            return None

        dtype = str(descriptor.get("type", "")).lower()
        if dtype == "table":
            file_path = descriptor.get("file")
            if not file_path:
                return None
            return EpsTable(str(file_path))

        if dtype == "python_module":
            module_path = descriptor.get("module_path")
            factory_name = str(descriptor.get("factory", "generate_eps_func"))
            if not module_path:
                return None

            module_file = Path(str(module_path))
            if not module_file.exists():
                return None

            mod_name = "_gui_user_mat_{}_{}".format(module_file.stem, abs(hash(str(module_file.resolve()))))
            spec_obj = importlib.util.spec_from_file_location(mod_name, str(module_file))
            if spec_obj is None or spec_obj.loader is None:
                return None

            module = importlib.util.module_from_spec(spec_obj)
            sys.modules[mod_name] = module
            spec_obj.loader.exec_module(module)

            if not hasattr(module, factory_name):
                return None

            material_obj = getattr(module, factory_name)()
            if callable(material_obj):
                return material_obj
            return None

        if dtype == "constant":
            if "epsilon" not in descriptor:
                return None
            eps_const = complex(descriptor["epsilon"])

            def _const_eps(w):
                arr = np.asarray(w)
                if arr.shape == ():
                    return eps_const, 0.0
                return np.full(arr.shape, eps_const, dtype = np.complex128), np.zeros(arr.shape, dtype = np.complex128)

            return _const_eps

        return None

    def _get_material_callable(self, material_name: str):
        descriptor = self.state.material_descriptors.get(material_name)
        descriptor_key = repr(descriptor)

        if material_name in self._resolved_material_cache:
            cached_key, cached_obj = self._resolved_material_cache[material_name]
            if cached_key == descriptor_key:
                return cached_obj

        material_obj = self._resolve_material_from_descriptor(material_name)
        self._resolved_material_cache[material_name] = (descriptor_key, material_obj)
        return material_obj

    def _eval_eps_vector(self, material_fn, wl: np.ndarray) -> np.ndarray:
        try:
            eps_val, _ = material_fn(wl)
            eps_arr = np.asarray(eps_val, dtype = np.complex128)
            if eps_arr.size == 1:
                return np.full(wl.shape, complex(eps_arr.flat[0]), dtype = np.complex128)
            if eps_arr.shape != wl.shape:
                eps_arr = np.reshape(eps_arr, wl.shape)
            return eps_arr
        except Exception:
            eps_out = np.empty(wl.shape, dtype = np.complex128)
            for i in range(wl.size):
                eps_i, _ = material_fn(float(wl[i]))
                eps_out[i] = complex(np.asarray(eps_i).flat[0])
            return eps_out

    def _get_eps_components(self, material_name: str, wl: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Resolve selected material descriptor and return eps real/imag over wavelengths."""
        if not material_name:
            return np.zeros_like(wl), np.zeros_like(wl)

        material_fn = self._get_material_callable(material_name)
        if material_fn is None:
            return np.zeros_like(wl), np.zeros_like(wl)

        eps_complex = np.ascontiguousarray(self._eval_eps_vector(material_fn, wl).reshape(-1))
        eps1 = np.real(eps_complex)
        eps2 = np.imag(eps_complex)
        return eps1.reshape(wl.shape), eps2.reshape(wl.shape)