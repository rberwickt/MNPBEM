from PySide6.QtWidgets import (QDialog, QLabel, QFormLayout, QGroupBox, QHBoxLayout, QVBoxLayout,
                               QLabel)
#from PySide6.QtCore import Qt
#from PySide6.QtGui import 
from ..simulation_state import SimulationState
class StateDebugDialog(QDialog):
    def __init__(self, state: SimulationState, parent=None):
        super().__init__(parent)
        self.state = state  # Keep a reference to the data struct
        self.setWindowTitle("DEBUG: State View")
        dlg_layout = QFormLayout(self)
        dlg_layout.addRow(QLabel("Close Debug Window to Continue Using Main Window (not multi-threaded)"))
        self.state_view = QGroupBox("Simulation State")

        state_columns = QHBoxLayout(self.state_view) # 5 each

        # col 1
        col_1 = QVBoxLayout()
        col_1.addWidget(QLabel(f"excitation_source: {self.state.excitation_source}"))
        col_1.addWidget(QLabel(f"polarization: {self.state.polarization}"))
        col_1.addWidget(QLabel(f"polarization_angle: {self.state.polarization_angle}"))
        col_1.addWidget(QLabel(f"jones_ex: {self.state.jones_ex}"))
        col_1.addWidget(QLabel(f"jones_ey: {self.state.jones_ey}"))
        # col 2
        col_2 = QVBoxLayout()
        col_2.addWidget(QLabel(f"jones_ez: {self.state.jones_ez}"))
        col_2.addWidget(QLabel(f"dir_x: {self.state.dir_x}"))
        col_2.addWidget(QLabel(f"dir_y: {self.state.dir_y}"))
        col_2.addWidget(QLabel(f"dir_z: {self.state.dir_z}"))
        col_2.addWidget(QLabel(f"kinetic_energy: {self.state.kinetic_energy}"))
        # col 3
        col_3 = QVBoxLayout()
        col_3.addWidget(QLabel(f"beam_width: {self.state.beam_width}"))
        col_3.addWidget(QLabel(f"oscillation_dir: {self.state.oscillation_dir}"))
        col_3.addWidget(QLabel(f"dipole_x: {self.state.dipole_x}"))
        col_3.addWidget(QLabel(f"dipole_y: {self.state.dipole_y}"))
        col_3.addWidget(QLabel(f"dipole_z: {self.state.dipole_z}"))

        # combining layouts
        state_columns.addLayout(col_1)
        state_columns.addLayout(col_2)
        state_columns.addLayout(col_3)

        dlg_layout.addRow(self.state_view)


