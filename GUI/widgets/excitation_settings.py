from PySide6.QtWidgets import (QGroupBox, QFormLayout, QComboBox, QStackedWidget, 
                               QWidget, QVBoxLayout, QHBoxLayout, QDoubleSpinBox, QSpinBox)
from PySide6.QtCore import Qt, Signal
#from PySide6.QtGui import QIntValidator
from ..simulation_state import SimulationState
class ExcitationSettingsWidget(QGroupBox):
    # Number Entry Ranges (if min isn't here, likely is 1)
    PW_POL_MIN = -100
    PW_POL_MAX = 100
    PW_DIR_MIN = -100
    PW_DIR_MAX = 100

    BEAM_ENERGY_MAX = 1000000.00 # (eV)
    BEAM_WIDTH_MAX = 100 # (nm)
    IMPACT_MAX = 100.0 # (nm)

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

        self.pol_vectors = QGroupBox("Polarization and Direction")
        pol_layout = QHBoxLayout(self.pol_vectors)

        # polarization x,y,z
        left_col = QFormLayout()
        self.pol_x = QSpinBox()
        self.pol_x.setRange(self.PW_POL_MIN, self.PW_POL_MAX)
        self.pol_x.setValue(self.state.pol_x)
        self.pol_x.valueChanged.connect(lambda val: setattr(self.state, 'pol_x', val))
        left_col.addRow("X:", self.pol_x)

        self.pol_y = QSpinBox()
        self.pol_y.setRange(self.PW_POL_MIN, self.PW_POL_MAX)
        self.pol_y.setValue(self.state.pol_y)
        self.pol_y.valueChanged.connect(lambda val: setattr(self.state, 'pol_y', val))
        left_col.addRow("Y:", self.pol_y)

        self.pol_z = QSpinBox()
        self.pol_z.setRange(self.PW_POL_MIN, self.PW_POL_MAX)
        self.pol_z.setValue(self.state.pol_z)
        self.pol_z.valueChanged.connect(lambda val: setattr(self.state, 'pol_z', val))
        left_col.addRow("Z:", self.pol_z)

        # direction
        right_col = QFormLayout()
        self.pw_dir_x = QSpinBox()
        self.pw_dir_x.setRange(self.PW_DIR_MIN, self.PW_DIR_MAX)
        self.pw_dir_x.setValue(self.state.pol_dir_x)
        self.pw_dir_x.valueChanged.connect(lambda val: setattr(self.state, 'pol_dir_x', val))
        right_col.addRow("DIR_X:", self.pw_dir_x)
        
        self.pw_dir_y = QSpinBox()
        self.pw_dir_y.setRange(self.PW_DIR_MIN, self.PW_DIR_MAX)
        self.pw_dir_y.setValue(self.state.pol_dir_y)
        self.pw_dir_y.valueChanged.connect(lambda val: setattr(self.state, 'pol_dir_y', val))
        right_col.addRow("DIR_Y:", self.pw_dir_y)

        self.pw_dir_z = QSpinBox()
        self.pw_dir_z.setRange(self.PW_DIR_MIN, self.PW_DIR_MAX)
        self.pw_dir_z.setValue(self.state.pol_dir_z)
        self.pw_dir_z.valueChanged.connect(lambda val: setattr(self.state, 'pol_dir_z', val))
        right_col.addRow("DIR_Z:", self.pw_dir_z)

        pol_layout.addLayout(left_col)
        pol_layout.addLayout(right_col)
        plane_wave_layout.addRow(self.pol_vectors)

        # Electron Beam ============================================
        self.beam_settings = QWidget()
        beam_layout = QFormLayout(self.beam_settings)

        self.beam_energy = QDoubleSpinBox()
        self.beam_energy.setRange(1.0, self.BEAM_ENERGY_MAX) 
        self.beam_energy.setDecimals(2)
        self.beam_energy.setSuffix(" eV")
        self.beam_energy.setValue(self.state.beam_energy)
        self.beam_energy.valueChanged.connect(lambda val: setattr(self.state, 'beam_energy', val))
        beam_layout.addRow("Beam Energy:", self.beam_energy)

        self.beam_width = QDoubleSpinBox()
        self.beam_width.setRange(0.1, self.BEAM_WIDTH_MAX)
        self.beam_width.setSuffix(" nm")
        self.beam_width.setValue(self.state.beam_width)
        self.beam_width.valueChanged.connect(lambda val: setattr(self.state, 'beam_width', val))
        beam_layout.addRow("Beam Width:", self.beam_width)

        self.impact = QDoubleSpinBox()
        self.impact.setRange(0.5, self.IMPACT_MAX) 
        self.impact.setDecimals(2)
        self.impact.setSuffix(" nm")
        self.impact.setValue(self.state.impact_parameter)
        self.impact.valueChanged.connect(lambda val: setattr(self.state, 'impact_parameter', val))
        beam_layout.addRow("Impact Parameter:", self.impact)

        # Dipole ===================================================
        self.dipole_settings = QWidget()
        dipole_layout = QFormLayout(self.dipole_settings)

        self.dip_vectors = QGroupBox("Position and Dipole Moment")
        vec_layout = QHBoxLayout(self.dip_vectors)

        # moment
        left_col2 = QFormLayout()
        self.moment_x = QSpinBox()
        self.moment_x.setRange(self.PW_POL_MIN, self.PW_POL_MAX) # reusing for now
        self.moment_x.setValue(self.state.dipole_moment_x)
        self.moment_x.valueChanged.connect(lambda val: setattr(self.state, 'dipole_moment_x', val))
        left_col2.addRow("X:", self.moment_x)

        self.moment_y = QSpinBox()
        self.moment_y.setRange(self.PW_POL_MIN, self.PW_POL_MAX)
        self.moment_y.setValue(self.state.dipole_moment_y)
        self.moment_y.valueChanged.connect(lambda val: setattr(self.state, 'dipole_moment_y', val))
        left_col2.addRow("Y:", self.moment_y)

        self.moment_z = QSpinBox()
        self.moment_z.setRange(self.PW_POL_MIN, self.PW_POL_MAX)
        self.moment_z.setValue(self.state.dipole_moment_z)
        self.moment_z.valueChanged.connect(lambda val: setattr(self.state, 'dipole_moment_z', val))
        left_col2.addRow("Z:", self.moment_z)

        # position
        right_col2 = QFormLayout()
        self.dip_pos_x = QSpinBox()
        self.dip_pos_x.setRange(self.PW_DIR_MIN, self.PW_DIR_MAX)
        self.dip_pos_x.setValue(self.state.dipole_pos_x)
        self.dip_pos_x.valueChanged.connect(lambda val: setattr(self.state, 'dipole_pos_x', val))
        right_col2.addRow("POS_X:", self.dip_pos_x)
        
        self.dip_pos_y = QSpinBox()
        self.dip_pos_y.setRange(self.PW_DIR_MIN, self.PW_DIR_MAX)
        self.dip_pos_y.setValue(self.state.dipole_pos_y)
        self.dip_pos_y.valueChanged.connect(lambda val: setattr(self.state, 'dipole_pos_y', val))
        right_col2.addRow("POS_Y:", self.dip_pos_y)

        self.dip_pos_z = QSpinBox()
        self.dip_pos_z.setRange(self.PW_DIR_MIN, self.PW_DIR_MAX)
        self.dip_pos_z.setValue(self.state.dipole_pos_z)
        self.dip_pos_z.valueChanged.connect(lambda val: setattr(self.state, 'dipole_pos_z', val))
        right_col2.addRow("POS_Z:", self.dip_pos_z)

        vec_layout.addLayout(left_col2)
        vec_layout.addLayout(right_col2)
        dipole_layout.addRow(self.dip_vectors)

        
        self.stacked_widget.addWidget(self.plane_wave_settings)
        self.stacked_widget.addWidget(self.beam_settings)
        self.stacked_widget.addWidget(self.dipole_settings)

        # layout
        self.layout = QVBoxLayout(self)
        form_layout = QFormLayout()
        form_layout.addRow(self.source_combo)
        form_layout.addRow(self.settings_group)
        self.layout.addLayout(form_layout)
        self.layout.addStretch()
        
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