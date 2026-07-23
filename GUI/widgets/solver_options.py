from .tooltip_combobox import ToolTipComboBox
from PySide6.QtWidgets import (
    QGroupBox,
    QVBoxLayout,
    QCheckBox,
    QHBoxLayout,
    QRadioButton,
    QLabel,
    QSpinBox,
    QDoubleSpinBox,
    QPushButton,
)
from ..simulation_state import SimulationState
from mnpbem.misc import EV2NM

# honestly this one is pretty small, but I made it it's own class for consistency
class SolverOptionsWidget(QGroupBox):
    def __init__(self, state: SimulationState, parent=None):
        super().__init__("Solver Selection", parent)
        self.state = state
        self.layout = QVBoxLayout(self)
        self.solver_combo = ToolTipComboBox()
        options = [("Retarded", "Solves for both charge and current, recommended"),
                   ("Quasistatic", "Quick, but only solves for charge (May impact post-processing)")]
        # ("Iterative Retarded", "Faster, but experimental")
        self.solver_combo.addItemsWithTooltips(options)
        self.solver_combo.currentTextChanged.connect(lambda solver: setattr(self.state, 'solver', solver))
        self.layout.addWidget(self.solver_combo)

        self.substrate_check = QCheckBox("Calculate Fields")
        self.substrate_check.setChecked(self.state.calc_fields)
        self.substrate_check.stateChanged.connect(lambda state: setattr(self.state, 'calc_fields', state == 2))
        self.layout.addWidget(self.substrate_check)

        if self.state.env_n_gpus_per_worker > 0:
                    radio_buttons = QHBoxLayout()
                    self.fp32 = QRadioButton("Single Precision (FP32)")
                    self.fp64 = QRadioButton("Double Precision (FP64)")
        
                    self.fp32.setChecked(self.state.gpu_precision == "fp32")
                    self.fp64.setChecked(self.state.gpu_precision == "fp64")
        
                    radio_buttons.addWidget(self.fp32)
                    radio_buttons.addWidget(self.fp64)
                    self.layout.addLayout(radio_buttons)
                    self.fp32.setToolTip("Single precision is faster and uses less memory, but may be less accurate for some simulations.")
                    self.fp64.setToolTip("Double precision is slower and uses more memory, but is more accurate for some simulations.")
        
                    self.fp32.toggled.connect(self._radio_toggled)
                    self.fp64.toggled.connect(self._radio_toggled)

        self._build_wavelength_controls()


    def _build_wavelength_controls(self):
        wavelength_row = QHBoxLayout()
        self.wavelength_group = QGroupBox("Wavelength Range")
        self.wavelength_group.setLayout(wavelength_row)
        min_label = QLabel("Min")
        wavelength_row.addWidget(min_label)

        self.min_box = QDoubleSpinBox()
        self.min_box.setRange(0.01, 2000.00)
        self.min_box.setSuffix(" nm")
        self.min_box.setValue(self.state.energy_min)
        self.min_box.valueChanged.connect(lambda val: setattr(self.state, 'energy_min', float(val)))
        wavelength_row.addWidget(self.min_box)

        max_label = QLabel("Max")
        wavelength_row.addWidget(max_label)

        self.max_box = QDoubleSpinBox()
        self.max_box.setRange(0.01, 2000.00)
        self.max_box.setSuffix(" nm")
        self.max_box.setValue(self.state.energy_max)
        self.max_box.valueChanged.connect(lambda val: setattr(self.state, 'energy_max', float(val)))
        wavelength_row.addWidget(self.max_box)

        steps_label = QLabel("Steps")
        wavelength_row.addWidget(steps_label)

        self.steps_box = QSpinBox()
        self.steps_box.setRange(1, 10000)
        self.steps_box.setValue(self.state.energy_steps)
        self.steps_box.valueChanged.connect(lambda val: setattr(self.state, 'energy_steps', int(val)))
        wavelength_row.addWidget(self.steps_box)

        self.layout.addWidget(self.wavelength_group)

        button_row = QHBoxLayout()
        button_row.addStretch() 
        self.unit_button = QPushButton("Change to eV")
        self.unit_button.clicked.connect(self._convert_units)
        button_row.addWidget(self.unit_button)
        button_row.addStretch()
        self.layout.addLayout(button_row)

    def _convert_units(self):
        self.state.energy_max = EV2NM / self.state.energy_max
        self.state.energy_min = EV2NM / self.state.energy_min

        if self.state.energy_in_nm:
            self.max_box.setSuffix(" ev")
            self.min_box.setSuffix(" ev")
            self.wavelength_group.setTitle("Energy Range")
            self.unit_button.setText("Change to nm")
        else:
            self.max_box.setSuffix(" nm")
            self.min_box.setSuffix(" nm")
            self.wavelength_group.setTitle("Wavelength Range")
            self.unit_button.setText("Change to eV")

        self.min_box.setValue(self.state.energy_min)
        self.max_box.setValue(self.state.energy_max)
        self.state.energy_in_nm = not self.state.energy_in_nm

    def _radio_toggled(self):
        if self.fp32.isChecked():
            self.state.gpu_precision = "fp32"
            self.fp64.setChecked(False)
        elif self.fp64.isChecked():
            self.state.gpu_precision = "fp64"
            self.fp32.setChecked(False)
