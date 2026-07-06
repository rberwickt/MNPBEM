from PySide6.QtWidgets import (QGroupBox, QFormLayout, QComboBox, QTabWidget, 
                               QWidget, QVBoxLayout, QHBoxLayout, QDoubleSpinBox, QSpinBox)
from PySide6.QtCore import Qt, Signal
#from PySide6.QtGui import QIntValidator
from ..simulation_state import SimulationState
class StructureSettingsWidget(QGroupBox):
    state_changed = Signal() # Haven't decided if this is going to be a useful signal, 
    #                           but I'll leave it here for now
    def __init__(self, state: SimulationState, parent=None):
        super().__init__("Structure Settings", parent)
        self.state = state  # Keep a reference to the data struct

        self.layout = QFormLayout(self)

        self.geo_combo = QComboBox()
        self.geo_combo.addItems(["Sphere", "Rod", "Cube", "Torus", "Ellipsoid"])
        self.geo_combo.setCurrentText(self.state.structure)
        self.geo_combo.currentTextChanged.connect(self._on_geo_changed)
        self.layout.addRow("Geometry:", self.geo_combo)


        # Sphere Settings =============================================
        # n, diameter=1.0 (faces ~ n, "n must be one of {144, 256, 484, 1024, ...} (precomputed meshes)")
 
        # Rod Settings ================================================
        # (diameter, height, n=None, triangles=False), little bit confused on n and triangles

        # Cube Settings ===============================================
        # (n, length=1.0, e=0.25) "n : int per edge, e edge rounding"


        # Allow for shells (use a system like the material_dropdown widgets)

        # Substrate (layered versions of functions) 
        # layer must be of form: LayerStructure(epstab, ind, z) 	epstab : list[Eps], ind : list[int], z : list[float] 	
        # Stratified medium definition (z[k] = interface, ind[k] = dielectric index).

    def _on_geo_changed(self, text: str):
        self.state.structure = text
        # might change the structure settings based on the structure (like excitation settings)
