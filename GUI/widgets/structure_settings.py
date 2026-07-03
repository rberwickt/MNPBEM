from PySide6.QtWidgets import (QGroupBox, QFormLayout, QComboBox, QStackedWidget, 
                               QWidget, QVBoxLayout, QHBoxLayout, QDoubleSpinBox, QSpinBox)
from PySide6.QtCore import Qt, Signal
#from PySide6.QtGui import QIntValidator
from ..simulation_state import SimulationState
class StructureSettingsWidget(QGroupBox):
    state_changed = Signal() # Haven't decided if this is going to be a useful signal, 
    #                           but I'll leave it here for now
    def __init__(self, state: SimulationState, parent=None):
        super().__init__("Structure Setings", parent)
        self.state = state  # Keep a reference to the data struct

        self.layout = QFormLayout(self)

        self.geo_combo = QComboBox()
        self.geo_combo.addItems(["Sphere", "Rod", "Cube", "Torus"])
        self.geo_combo.setCurrentText(self.state.structure)
        self.geo_combo.currentTextChanged.connect(self._on_geo_changed)
        self.layout.addRow("Structure:", self.geo_combo)

    def _on_geo_changed(self, text: str):
        self.state.structure = text
        # might change the structure settings based on the structure (like excitation settings)
