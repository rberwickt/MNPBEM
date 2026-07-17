from .tooltip_combobox import ToolTipComboBox
from PySide6.QtWidgets import (QGroupBox, QVBoxLayout, QCheckBox, QHBoxLayout, QRadioButton)
from ..simulation_state import SimulationState

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
    def _radio_toggled(self):
        if self.fp32.isChecked():
            self.state.gpu_precision = "fp32"
            self.fp64.setChecked(False)
        elif self.fp64.isChecked():
            self.state.gpu_precision = "fp64"
            self.fp32.setChecked(False)
