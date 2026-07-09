from PySide6.QtWidgets import (QGroupBox, QFormLayout, QHBoxLayout, QCheckBox)
from PySide6.QtCore import Qt
from ..simulation_state import SimulationState
from .material_dropdown import MaterialComboBox

# choose the environment and substate options

class MaterialOptionsWidget(QGroupBox):
    def __init__(self, state: SimulationState, parent=None):
        super().__init__("Environment Settings", parent)
        self.state = state
        self.layout = QFormLayout(self)

        self.medium_combo = MaterialComboBox(self.state)
        self.layout.addRow("Environment Material:", self.medium_combo)
        self.medium_combo.currentTextChanged.connect(lambda mat: setattr(self.state, 'environment_material', self.state.loaded_dielectrics[mat]))

        self.substrate_check = QCheckBox()
        checkbox_layout = QHBoxLayout()
        checkbox_layout.addStretch()         
        checkbox_layout.addWidget(self.substrate_check)
        checkbox_layout.addStretch()         
        self.substrate_check.checkStateChanged.connect(self._handle_check)
        self.layout.addRow("Enable Substrate:", checkbox_layout)
        
        self.substrate_combo = MaterialComboBox(self.state)
        self.substrate_combo.currentTextChanged.connect(lambda mat: setattr(self.state, 'substrate_material', self.state.loaded_dielectrics[mat]))
        self.layout.addRow("Substrate Material:", self.substrate_combo)
        self.layout.setRowVisible(self.substrate_combo, False) # hide since it is false by default

    def _handle_check(self, checkState):
        if checkState == Qt.CheckState.Checked:
            self.state.use_substrate = True
            self.layout.setRowVisible(self.substrate_combo, True)
        else:
            self.state.use_substrate = False
            self.layout.setRowVisible(self.substrate_combo, False)