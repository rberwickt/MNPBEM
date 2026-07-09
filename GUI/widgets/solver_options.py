from .tooltip_combobox import ToolTipComboBox
from PySide6.QtWidgets import (QGroupBox, QVBoxLayout)
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
