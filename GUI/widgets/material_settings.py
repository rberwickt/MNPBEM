from PySide6.QtWidgets import (QGroupBox, QFormLayout, QSpinBox, QScrollArea, QWidget, QVBoxLayout)
from PySide6.QtCore import Qt
from ..simulation_state import SimulationState
from .material_dropdown import MaterialComboBox

# structure: choose environment, choose any number of extra materials and then combine into ComParticle at simulation time(?)
# TODO: allow for on the fly creation of EpsConst callables instead of having to make a file like vacuum.py each time

class MaterialOptionsWidget(QGroupBox):
    MAX_MATERIALS = 5
    def __init__(self, state: SimulationState, parent=None):
        super().__init__("Material Selection", parent)
        self.state = state
        self.layout = QVBoxLayout(self)
        self.material_selects = []

        # material count select
        self.mat_count = QSpinBox()
        self.mat_count.setRange(1, self.MAX_MATERIALS)
        toSet = 1
        if toSet < self.state.material_count: toSet = self.state.material_count
        self.mat_count.setValue(toSet) # should trigger the signal, generating selects
        mat_label = QFormLayout()
        mat_label.addRow("Number of Materials:", self.mat_count)
        self.layout.addLayout(mat_label)

        self.mat_scroll = QScrollArea()
        self.mat_scroll.setWidgetResizable(True)  # allows inner widgets to scale
        self.mat_scroll.setFixedHeight(300)       # fixes the height so it won't grow instead of the inner widgets
        self.mat_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.mat_content = QWidget()
        self.mat_layout = QVBoxLayout(self.mat_content)
        self.mat_layout.setAlignment(Qt.AlignTop)

        self._change_selects(1)
        self.mat_count.valueChanged.connect(self._change_selects)
        

        self.mat_scroll.setWidget(self.mat_content)
        self.layout.addWidget(self.mat_scroll)

    def _change_selects(self, val: int): # change the number of selections (and the state) based on the new mat_count
        val = val + 1 # leave room for environment
        if val > self.state.material_count: # adding selects
            for i in range(self.state.material_count, val):
                label = f"Structure {i}:"
                if i == 0:
                    label = "Environment:"
                mat_box = MaterialComboBox(self.state, i, label)
                self.material_selects.append(mat_box)
                self.mat_layout.addWidget(mat_box)
                self.state.materials.append(None) # will be filled in later (ideally) <- end goal is to have something verify all data before simulation
        elif val < self.state.material_count: # removing selects
            for i in range(self.state.material_count - val):
                to_remove = self.material_selects.pop() # remove the last element until they are equal again
                self.state.materials.pop() # do the same with the state
                self.mat_layout.removeWidget(to_remove)
                to_remove.hide()
                to_remove.deleteLater()
        self.state.material_count = val
