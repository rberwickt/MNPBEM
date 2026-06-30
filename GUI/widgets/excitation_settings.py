from PySide6.QtWidgets import (QGroupBox, QFormLayout, QComboBox, QStackedWidget, 
                               QWidget, QVBoxLayout, QLabel, QHBoxLayout, 
                               QLineEdit, QDoubleSpinBox, QSpinBox)
from PySide6.QtCore import Qt, Signal
#from PySide6.QtGui import QIntValidator
from ..simulation_state import SimulationState
class ExcitationSettingsWidget(QGroupBox):
    state_changed = Signal() # Haven't decided if this is going to be a useful signal, but I'll leave it here for now
    
    # Number Entry Ranges (if min isn't here, likely is 0)
    JONES_MIN = -100
    JONES_MAX = 100
    PW_DIR_MIN = -100
    PW_DIR_MAX = 100

    KINETIC_ENERGY_MAX = 1000000.00
    BEAM_WIDTH_MAX = 100 # (nm)

    def __init__(self, state: SimulationState, parent=None):
        super().__init__("Excitation Source", parent)
        self.state = state  # Keep a reference to the data struct

        self.source_combo = QComboBox()
        self.source_combo.addItems(["Plane Wave", "Electron Beam", "Dipole"])
        self.source_combo.setCurrentText(self.state.excitation_source)
        
        self.settings_group = QGroupBox("Excitation Settings (Plane Wave)")
        settings_layout = QVBoxLayout(self.settings_group)
        self.stacked_widget = QStackedWidget()
        settings_layout.addWidget(self.stacked_widget)

        # Plane Wave ================================================
        self.plane_wave_settings = QWidget()
        plane_wave_layout = QFormLayout(self.plane_wave_settings)

        self.polarization_combo = QComboBox()
        self.polarization_combo.addItems(["p", "s", "Polarization Vector", "Polarization Angle"])
        self.polarization_combo.setCurrentText(self.state.polarization)
        self.polarization_combo.currentTextChanged.connect(lambda pol: setattr(self.state, 'polarization', pol))
        plane_wave_layout.addRow("Polarization:", self.polarization_combo)

        self.jones_vectors = QGroupBox("Jones Vectors and Direction")
        jones_layout = QHBoxLayout(self.jones_vectors)

        # ex,ey,ez
        left_col = QFormLayout()
        self.ex = QSpinBox()
        self.ex.setRange(self.JONES_MIN, self.JONES_MAX)
        self.ex.setValue(self.state.jones_ex)
        self.ex.valueChanged.connect(lambda val: setattr(self.state, 'jones_ex', val))
        left_col.addRow("Ex:", self.ex)

        self.ey = QSpinBox()
        self.ey.setRange(self.JONES_MIN, self.JONES_MAX)
        self.ey.setValue(self.state.jones_ey)
        self.ey.valueChanged.connect(lambda val: setattr(self.state, 'jones_ey', val))
        left_col.addRow("Ey:", self.ey)

        self.ez = QSpinBox()
        self.ez.setRange(self.JONES_MIN, self.JONES_MAX)
        self.ez.setValue(self.state.jones_ez)
        self.ez.valueChanged.connect(lambda val: setattr(self.state, 'jones_ez', val))
        left_col.addRow("Ez:", self.ez)

        # direction
        right_col = QFormLayout()
        self.dir_x = QSpinBox()
        self.dir_x.setRange(self.PW_DIR_MIN, self.PW_DIR_MAX)
        self.dir_x.setValue(self.state.dir_x)
        self.dir_x.valueChanged.connect(lambda val: setattr(self.state, 'dir_x', val))
        right_col.addRow("Dir_x:", self.dir_x)
        
        self.dir_y = QSpinBox()
        self.dir_y.setRange(self.PW_DIR_MIN, self.PW_DIR_MAX)
        self.dir_y.setValue(self.state.dir_y)
        self.dir_y.valueChanged.connect(lambda val: setattr(self.state, 'dir_y', val))
        right_col.addRow("Dir_y:", self.dir_y)

        self.dir_z = QSpinBox()
        self.dir_z.setRange(self.PW_DIR_MIN, self.PW_DIR_MAX)
        self.dir_z.setValue(self.state.dir_z)
        self.dir_z.valueChanged.connect(lambda val: setattr(self.state, 'dir_z', val))
        right_col.addRow("Dir_z:", self.dir_z)

        jones_layout.addLayout(left_col)
        jones_layout.addLayout(right_col)
        plane_wave_layout.addRow(self.jones_vectors)

        # Electron Beam ============================================
        self.beam_settings = QWidget()
        beam_layout = QFormLayout(self.beam_settings)

        self.kinetic_energy = QDoubleSpinBox()
        self.kinetic_energy.setRange(1.0, self.KINETIC_ENERGY_MAX) 
        self.kinetic_energy.setDecimals(2)
        self.kinetic_energy.setSuffix(" eV")
        self.kinetic_energy.setValue(self.state.kinetic_energy)
        self.kinetic_energy.valueChanged.connect(lambda val: setattr(self.state, 'kinetic_energy', val))
        beam_layout.addRow("Kinetic Energy:", self.kinetic_energy)

        self.beam_width = QSpinBox()
        self.beam_width.setRange(1, self.BEAM_WIDTH_MAX)
        self.beam_width.setSuffix(" nm")
        self.beam_width.setValue(self.state.beam_width)
        self.beam_width.valueChanged.connect(lambda val: setattr(self.state, 'beam_width', val))
        beam_layout.addRow("Beam Width:", self.beam_width)

        # Dipole ===================================================
        self.dipole_settings = QWidget()
        dipole_layout = QFormLayout(self.dipole_settings)

        self.oscillation_combo = QComboBox()
        self.oscillation_combo.addItems(["x", "y", "z"])
        self.oscillation_combo.setCurrentText(self.state.oscillation_dir)
        self.oscillation_combo.currentTextChanged.connect(lambda pol: setattr(self.state, 'oscillation_dir', pol))
        dipole_layout.addRow("Oscillation Direction:", self.oscillation_combo)

        
        self.stacked_widget.addWidget(self.plane_wave_settings)
        self.stacked_widget.addWidget(self.beam_settings)
        self.stacked_widget.addWidget(self.dipole_settings)

        # layout
        layout = QFormLayout(self)
        layout.addRow(self.source_combo)
        layout.addRow(self.settings_group)
        
        # binding all the settings signals
        self.source_combo.currentTextChanged.connect(self._on_source_changed)

    # --- Signal handlers to change the state ---
    def _on_source_changed(self, text: str):
        # from ["Plane Wave", "Electron Beam", "Dipole"]
        self.state.excitation_source = text
        if text == "Plane Wave":
            self.settings_group.setTitle("Excitation Settings (Plane Wave)")
            self.stacked_widget.setCurrentIndex(0)
            setattr(self.state, 'excitation_source', "Plane Wave")
        elif text == "Electron Beam":
            self.settings_group.setTitle("Excitation Settings (Electron Beam)")
            self.stacked_widget.setCurrentIndex(1)
            setattr(self.state, 'excitation_source', "Electron Beam")
        elif text == "Dipole":
            self.settings_group.setTitle("Excitation Settings (Dipole)")
            self.stacked_widget.setCurrentIndex(2)
            setattr(self.state, 'excitation_source', "Dipole")
        self.state_changed.emit()