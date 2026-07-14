from PySide6.QtWidgets import (
    QGroupBox, QFormLayout, QSpinBox, QDoubleSpinBox,
    QPushButton)
from PySide6.QtCore import Signal
from ..simulation_state import SimulationState
from mnpbem.misc import EV2NM

class EnergyRangeWidget(QGroupBox):
    #range_changed = Signal() # notify the refractive index graph that the range has changed so it can update
    # decided to leave this commented for now (might end up being a performance issue if they change it by holding down the arrow keys)
    def __init__(self, state: SimulationState, parent=None):
        super().__init__("Wavelength Range", parent)
        self.state = state
        self.layout = QFormLayout(self)

        # start in nm by default
        self.max_box = QDoubleSpinBox()
        self.max_box.setRange(0.01, 2000.00)
        self.max_box.setSuffix(" nm")
        self.max_box.setValue(self.state.energy_max)
        self.max_box.valueChanged.connect(lambda val: setattr(self.state, 'energy_max', val))
        self.layout.addRow("Max:", self.max_box)

        self.min_box = QDoubleSpinBox()
        self.min_box.setRange(0.01, 2000.00)
        self.min_box.setSuffix(" nm")
        self.min_box.setValue(self.state.energy_min)
        self.min_box.valueChanged.connect(lambda val: setattr(self.state, 'energy_min', val))
        self.layout.addRow("Min:", self.min_box)

        # see above
        #self.max_box.valueChanged.connect(self.range_changed.emit)
        #self.min_box.valueChanged.connect(self.range_changed.emit)

        self.steps_box = QSpinBox()
        self.steps_box.setRange(1, 10000)
        self.steps_box.setValue(self.state.energy_steps)
        self.steps_box.valueChanged.connect(lambda val: setattr(self.state, 'energy_steps', val))
        self.layout.addRow("Steps:", self.steps_box)

        self.unit_button = QPushButton("Change to eV")
        self.unit_button.clicked.connect(self._convert_units)
        self.layout.addRow(self.unit_button)

    def _convert_units(self): # swap between nm and eV freely
        self.state.energy_max = EV2NM / self.state.energy_max
        self.state.energy_min = EV2NM / self.state.energy_min

        if self.state.energy_in_nm: # swap to eV
            self.max_box.setSuffix(" ev")
            self.min_box.setSuffix(" ev")
            self.setTitle("Energy Range")
            self.unit_button.setText("Change to nm")
        else: # swap to nm
            self.max_box.setSuffix(" nm")
            self.min_box.setSuffix(" nm")
            self.setTitle("Wavelength Range")
            self.unit_button.setText("Change to eV")

        self.min_box.setValue(self.state.energy_min)
        self.max_box.setValue(self.state.energy_max)
        self.state.energy_in_nm = not self.state.energy_in_nm

        
