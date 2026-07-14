from PySide6.QtWidgets import (QGroupBox, QFormLayout, QComboBox, QStackedWidget,
                               QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QDoubleSpinBox,
                               QSpinBox, QPushButton, QLabel, QAbstractSpinBox)

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
        self._plane_wave_rows: list[dict] = []

        self.source_combo = QComboBox()
        self.source_combo.addItems(["Plane Wave", "Electron Beam", "Dipole"])
        self.source_combo.setCurrentText(self.state.excitation_source)
        
        self.settings_group = QGroupBox("Excitation Settings (Plane Wave)")
        settings_layout = QVBoxLayout(self.settings_group)
        self.stacked_widget = QStackedWidget()
        settings_layout.addWidget(self.stacked_widget)

        # Plane Wave ================================================
        self.plane_wave_settings = QWidget()
        plane_wave_layout = QVBoxLayout(self.plane_wave_settings)

        self.pol_vectors = QGroupBox("Polarizations and Directions")
        pol_layout = QVBoxLayout(self.pol_vectors)

        self._plane_wave_rows_widget = QWidget()
        self._plane_wave_rows_layout = QVBoxLayout(self._plane_wave_rows_widget)
        self._plane_wave_rows_layout.setContentsMargins(0, 0, 0, 0)
        self._plane_wave_rows_layout.setSpacing(4)
        pol_layout.addWidget(self._plane_wave_rows_widget)

        self._add_pol_btn = QPushButton("Add Polarization")
        self._add_pol_btn.clicked.connect(self._on_add_plane_wave_polarization)
        pol_layout.addWidget(self._add_pol_btn)

        plane_wave_layout.addWidget(self.pol_vectors)

        polarizations = self.state.get_plane_wave_polarizations()
        propagation_dirs = self.state.get_plane_wave_propagation_dirs()
        if len(propagation_dirs) < len(polarizations):
            last_dir = propagation_dirs[-1] if len(propagation_dirs) > 0 else [0, 0, 1]
            propagation_dirs = propagation_dirs + [list(last_dir) for _ in range(len(polarizations) - len(propagation_dirs))]

        for pol, direction in zip(polarizations, propagation_dirs):
            self._add_plane_wave_polarization_row(pol, direction)
        self._sync_plane_wave_polarizations()

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

    def _on_add_plane_wave_polarization(self):
        last_pol = self.state.get_plane_wave_polarizations()[-1]
        last_dir = self.state.get_plane_wave_propagation_dirs()[-1]
        self._add_plane_wave_polarization_row(last_pol, last_dir)
        self._sync_plane_wave_polarizations()

    def _add_plane_wave_polarization_row(self, vector, direction):
        row_widget = QWidget()
        row_layout = QGridLayout(row_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setHorizontalSpacing(4)
        row_layout.setVerticalSpacing(2)

        label = QLabel()
        row_layout.addWidget(label, 0, 0)

        row_layout.addWidget(QLabel("PX"), 0, 1)

        x_spin = QSpinBox()
        x_spin.setRange(self.PW_POL_MIN, self.PW_POL_MAX)
        x_spin.setValue(int(vector[0]))
        x_spin.setButtonSymbols(QAbstractSpinBox.NoButtons)
        x_spin.setMaximumWidth(64)
        row_layout.addWidget(x_spin, 0, 2)

        row_layout.addWidget(QLabel("PY"), 0, 3)

        y_spin = QSpinBox()
        y_spin.setRange(self.PW_POL_MIN, self.PW_POL_MAX)
        y_spin.setValue(int(vector[1]))
        y_spin.setButtonSymbols(QAbstractSpinBox.NoButtons)
        y_spin.setMaximumWidth(64)
        row_layout.addWidget(y_spin, 0, 4)

        row_layout.addWidget(QLabel("PZ"), 0, 5)

        z_spin = QSpinBox()
        z_spin.setRange(self.PW_POL_MIN, self.PW_POL_MAX)
        z_spin.setValue(int(vector[2]))
        z_spin.setButtonSymbols(QAbstractSpinBox.NoButtons)
        z_spin.setMaximumWidth(64)
        row_layout.addWidget(z_spin, 0, 6)

        row_layout.addWidget(QLabel("DX"), 1, 1)
        dir_x_spin = QSpinBox()
        dir_x_spin.setRange(self.PW_DIR_MIN, self.PW_DIR_MAX)
        dir_x_spin.setValue(int(direction[0]))
        dir_x_spin.setButtonSymbols(QAbstractSpinBox.NoButtons)
        dir_x_spin.setMaximumWidth(64)
        row_layout.addWidget(dir_x_spin, 1, 2)

        row_layout.addWidget(QLabel("DY"), 1, 3)
        dir_y_spin = QSpinBox()
        dir_y_spin.setRange(self.PW_DIR_MIN, self.PW_DIR_MAX)
        dir_y_spin.setValue(int(direction[1]))
        dir_y_spin.setButtonSymbols(QAbstractSpinBox.NoButtons)
        dir_y_spin.setMaximumWidth(64)
        row_layout.addWidget(dir_y_spin, 1, 4)

        row_layout.addWidget(QLabel("DZ"), 1, 5)
        dir_z_spin = QSpinBox()
        dir_z_spin.setRange(self.PW_DIR_MIN, self.PW_DIR_MAX)
        dir_z_spin.setValue(int(direction[2]))
        dir_z_spin.setButtonSymbols(QAbstractSpinBox.NoButtons)
        dir_z_spin.setMaximumWidth(64)
        row_layout.addWidget(dir_z_spin, 1, 6)

        remove_btn = QPushButton("Remove")
        remove_btn.setMaximumWidth(100)
        row_layout.addWidget(remove_btn, 0, 7, 2, 1)

        row_layout.setColumnStretch(8, 1)

        row = {
            "widget": row_widget,
            "label": label,
            "x": x_spin,
            "y": y_spin,
            "z": z_spin,
            "dir_x": dir_x_spin,
            "dir_y": dir_y_spin,
            "dir_z": dir_z_spin,
            "remove": remove_btn,
        }
        self._plane_wave_rows.append(row)
        self._plane_wave_rows_layout.addWidget(row_widget)

        x_spin.valueChanged.connect(self._sync_plane_wave_polarizations)
        y_spin.valueChanged.connect(self._sync_plane_wave_polarizations)
        z_spin.valueChanged.connect(self._sync_plane_wave_polarizations)
        dir_x_spin.valueChanged.connect(self._sync_plane_wave_polarizations)
        dir_y_spin.valueChanged.connect(self._sync_plane_wave_polarizations)
        dir_z_spin.valueChanged.connect(self._sync_plane_wave_polarizations)
        remove_btn.clicked.connect(
            lambda _checked = False, current_row = row: self._remove_plane_wave_polarization_row(current_row)
        )

        self._refresh_plane_wave_row_state()

    def _remove_plane_wave_polarization_row(self, row: dict):
        if len(self._plane_wave_rows) <= 1:
            return

        self._plane_wave_rows.remove(row)
        row["widget"].setParent(None)
        row["widget"].deleteLater()
        self._sync_plane_wave_polarizations()

    def _refresh_plane_wave_row_state(self):
        can_remove = len(self._plane_wave_rows) > 1
        for idx, row in enumerate(self._plane_wave_rows, start = 1):
            row["label"].setText("Pol {}".format(idx))
            row["remove"].setEnabled(can_remove)

    def _sync_plane_wave_polarizations(self):
        polarizations = []
        propagation_dirs = []
        for row in self._plane_wave_rows:
            polarizations.append([
                int(row["x"].value()),
                int(row["y"].value()),
                int(row["z"].value()),
            ])
            propagation_dirs.append([
                int(row["dir_x"].value()),
                int(row["dir_y"].value()),
                int(row["dir_z"].value()),
            ])

        self.state.set_plane_wave_polarizations(polarizations)
        self.state.set_plane_wave_propagation_dirs(propagation_dirs)
        self._refresh_plane_wave_row_state()