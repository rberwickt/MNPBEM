from PySide6.QtWidgets import (
    QGroupBox, QFormLayout, QSpinBox, QDoubleSpinBox,
    QPushButton, QWidget, QHBoxLayout)
from ..simulation_state import SimulationState


class FieldGridWidget(QGroupBox):

    def __init__(self, state: SimulationState, parent=None):
        super().__init__('Field Grid Settings', parent)
        self.state = state
        self.layout = QFormLayout(self)

        self.x_min_box = QDoubleSpinBox()
        self.x_min_box.setRange(-5000.0, 5000.0)
        self.x_min_box.setValue(self.state.field_x_min)
        self.x_min_box.setSuffix(' nm')
        self.x_min_box.valueChanged.connect(
                lambda val: setattr(self.state, 'field_x_min', float(val)))

        self.x_max_box = QDoubleSpinBox()
        self.x_max_box.setRange(-5000.0, 5000.0)
        self.x_max_box.setValue(self.state.field_x_max)
        self.x_max_box.setSuffix(' nm')
        self.x_max_box.valueChanged.connect(
                lambda val: setattr(self.state, 'field_x_max', float(val)))

        self.y_min_box = QDoubleSpinBox()
        self.y_min_box.setRange(-5000.0, 5000.0)
        self.y_min_box.setValue(self.state.field_y_min)
        self.y_min_box.setSuffix(' nm')
        self.y_min_box.valueChanged.connect(
                lambda val: setattr(self.state, 'field_y_min', float(val)))

        self.y_max_box = QDoubleSpinBox()
        self.y_max_box.setRange(-5000.0, 5000.0)
        self.y_max_box.setValue(self.state.field_y_max)
        self.y_max_box.setSuffix(' nm')
        self.y_max_box.valueChanged.connect(
                lambda val: setattr(self.state, 'field_y_max', float(val)))

        self.z_min_box = QDoubleSpinBox()
        self.z_min_box.setRange(-5000.0, 5000.0)
        self.z_min_box.setValue(self.state.field_z_min)
        self.z_min_box.setSuffix(' nm')
        self.z_min_box.valueChanged.connect(
                lambda val: setattr(self.state, 'field_z_min', float(val)))

        self.z_max_box = QDoubleSpinBox()
        self.z_max_box.setRange(-5000.0, 5000.0)
        self.z_max_box.setValue(self.state.field_z_max)
        self.z_max_box.setSuffix(' nm')
        self.z_max_box.valueChanged.connect(
                lambda val: setattr(self.state, 'field_z_max', float(val)))

        self.nx_box = QSpinBox()
        self.nx_box.setRange(1, 500)
        self.nx_box.setValue(self.state.field_nx)
        self.nx_box.valueChanged.connect(
                lambda val: setattr(self.state, 'field_nx', int(val)))

        self.ny_box = QSpinBox()
        self.ny_box.setRange(1, 500)
        self.ny_box.setValue(self.state.field_ny)
        self.ny_box.valueChanged.connect(
                lambda val: setattr(self.state, 'field_ny', int(val)))

        self.nz_box = QSpinBox()
        self.nz_box.setRange(1, 500)
        self.nz_box.setValue(self.state.field_nz)
        self.nz_box.valueChanged.connect(self._on_nz_changed)

        self.layout.addRow('x min:', self.x_min_box)
        self.layout.addRow('x max:', self.x_max_box)
        self.layout.addRow('y min:', self.y_min_box)
        self.layout.addRow('y max:', self.y_max_box)
        self.layout.addRow('z min:', self.z_min_box)
        self.layout.addRow('z max:', self.z_max_box)
        self.layout.addRow('nx:', self.nx_box)
        self.layout.addRow('ny:', self.ny_box)
        self.layout.addRow('nz:', self.nz_box)

        mode_row = QWidget()
        mode_layout = QHBoxLayout(mode_row)
        mode_layout.setContentsMargins(0, 0, 0, 0)

        self.planar_btn = QPushButton('Set 2D Slice')
        self.planar_btn.clicked.connect(self._set_planar_grid)
        mode_layout.addWidget(self.planar_btn)

        self.vol_btn = QPushButton('Set Volumetric')
        self.vol_btn.clicked.connect(self._set_volumetric_grid)
        mode_layout.addWidget(self.vol_btn)

        self.layout.addRow(mode_row)

        self._on_nz_changed(int(self.state.field_nz))

    def _on_nz_changed(self, val: int):
        self.state.field_nz = int(val)
        is_3d = int(val) > 1
        self.z_min_box.setEnabled(is_3d)
        self.z_max_box.setEnabled(is_3d)

    def _set_planar_grid(self):
        # Keep current z center but collapse z sampling to a single plane.
        z_mid = 0.5 * (float(self.state.field_z_min) + float(self.state.field_z_max))
        self.z_min_box.setValue(z_mid)
        self.z_max_box.setValue(z_mid)
        self.nz_box.setValue(1)

    def _set_volumetric_grid(self):
        # If z is collapsed, expand to a default symmetric slab.
        if abs(float(self.state.field_z_max) - float(self.state.field_z_min)) < 1e-12:
            self.z_min_box.setValue(-50.0)
            self.z_max_box.setValue(50.0)
        if int(self.state.field_nz) <= 1:
            self.nz_box.setValue(21)
