from PySide6.QtWidgets import QGroupBox, QFormLayout, QComboBox, QStackedWidget, QWidget, QFormLayout, QLabel, QHBoxLayout, QLineEdit
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QIntValidator
from ..simulation_state import SimulationState
class ExcitationSettingsWidget(QGroupBox):
    state_changed = Signal() # Haven't decided if this is going to be a useful signal, but I'll leave it here for now
    JONES_MIN = -10
    JONES_MAX = 10

    def __init__(self, state: SimulationState, parent=None):
        super().__init__("Excitation Source", parent)
        self.state = state  # Keep a reference to the data struct

        self.source_combo = QComboBox()
        self.source_combo.addItems(["Plane Wave", "Electron Beam", "Dipole"])
        self.source_combo.setCurrentText(self.state.excitation_source)
        
        self.stacked_widget = QStackedWidget()

        # Plane Wave ================================================
        self.plane_wave_settings = QWidget()
        plane_wave_layout = QFormLayout(self.plane_wave_settings)

        plane_wave_layout.addRow(QLabel("Plane Wave Settings"))

        self.polarization_combo = QComboBox()
        self.polarization_combo.addItems(["p", "s", "Polarization Vector", "Polarization Angle"])
        self.polarization_combo.setCurrentText(self.state.polarization)
        plane_wave_layout.addRow("Polarizaton", self.polarization_combo)

        self.jones_vectors = QGroupBox("Jones Vectors")
        self.jones_validator = QIntValidator(self.JONES_MIN, self.JONES_MAX, self.jones_vectors)
        jones_layout = QHBoxLayout(self.jones_vectors)
        # ex,ey,ez
        left_col = QFormLayout()
        self.ex = QLineEdit()
        self.ex.setValidator(self.jones_validator)
        left_col.addRow("Ex", self.ex)

        self.ey = QLineEdit()
        self.ey.setValidator(self.jones_validator)
        left_col.addRow("Ey", self.ey)

        self.ez = QLineEdit()
        self.ez.setValidator(self.jones_validator)
        left_col.addRow("Ez", self.ez)
        # direction
        right_col = QFormLayout()
        self.dir_x = QLineEdit()
        self.dir_x.setValidator(self.jones_validator)
        right_col.addRow("Dir_x", self.dir_x)
        
        self.dir_y = QLineEdit()
        self.dir_y.setValidator(self.jones_validator)
        right_col.addRow("Dir_y", self.dir_y)

        self.dir_z = QLineEdit()
        self.dir_z.setValidator(self.jones_validator)
        right_col.addRow("Dir_z", self.dir_z)

        jones_layout.addLayout(left_col)
        jones_layout.addLayout(right_col)
        plane_wave_layout.addRow(self.jones_vectors)

        # Electron Beam ============================================
        self.beam_settings = QWidget()
        beam_layout = QFormLayout(self.beam_settings)
        beam_layout.addRow(QLabel("Electron Beam Settings"))

        # Dipole ===================================================
        self.dipole_settings = QWidget()
        dipole_layout = QFormLayout(self.dipole_settings)
        dipole_layout.addRow(QLabel("Dipole Settings"))


        
        self.stacked_widget.addWidget(self.plane_wave_settings)
        self.stacked_widget.addWidget(self.beam_settings)
        self.stacked_widget.addWidget(self.dipole_settings)

        # layout
        layout = QFormLayout(self)
        layout.addRow("Excitation Source", self.source_combo)
        layout.addRow("Excitation Settings:", self.stacked_widget)
        
        # binding all the settings signals
        self.source_combo.currentTextChanged.connect(self._on_source_changed)

    # --- Signal handlers to change the state ---
    def _on_source_changed(self, text: str):
        # from ["Plane Wave", "Electron Beam", "Dipole"]
        self.state.excitation_source = text
        if text == "Plane Wave":
            self.stacked_widget.setCurrentIndex(0)
        elif text == "Electron Beam":
            self.stacked_widget.setCurrentIndex(1)
        elif text == "Dipole":
            self.stacked_widget.setCurrentIndex(2)
        self.state_changed.emit()