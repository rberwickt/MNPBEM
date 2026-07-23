from PySide6.QtWidgets import (
    QGroupBox, QFormLayout, QSpinBox, QDoubleSpinBox,
    QPushButton, QWidget, QHBoxLayout)
from ..simulation_state import SimulationState


class FieldGridWidget(QGroupBox):

    def __init__(self, state: SimulationState, parent=None):
        super().__init__('Field Grid Settings', parent)
        self.state = state
        self.layout = QFormLayout(self)

        self.x_half_range_box = QDoubleSpinBox()
        self.x_half_range_box.setRange(0.0, 5000.0)
        self.x_half_range_box.setValue(self._symmetric_half_range(self.state.field_x_min, self.state.field_x_max))
        self.x_half_range_box.setSuffix(' nm')
        self.x_half_range_box.valueChanged.connect(
                lambda val: self._set_axis_symmetric_range('x', float(val)))

        self.y_half_range_box = QDoubleSpinBox()
        self.y_half_range_box.setRange(0.0, 5000.0)
        self.y_half_range_box.setValue(self._symmetric_half_range(self.state.field_y_min, self.state.field_y_max))
        self.y_half_range_box.setSuffix(' nm')
        self.y_half_range_box.valueChanged.connect(
                lambda val: self._set_axis_symmetric_range('y', float(val)))

        self.z_half_range_box = QDoubleSpinBox()
        self.z_half_range_box.setRange(0.0, 5000.0)
        self.z_half_range_box.setValue(self._symmetric_half_range(self.state.field_z_min, self.state.field_z_max))
        self.z_half_range_box.setSuffix(' nm')
        self.z_half_range_box.valueChanged.connect(
                lambda val: self._set_axis_symmetric_range('z', float(val)))

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

        self.layout.addRow('x +/-:', self.x_half_range_box)
        self.layout.addRow('y +/-:', self.y_half_range_box)
        self.layout.addRow('z +/-:', self.z_half_range_box)
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
        self.z_half_range_box.setEnabled(is_3d)

    def _set_planar_grid(self):
        # Collapse z sampling to the center plane
        self.z_half_range_box.setValue(0.0)
        self.nz_box.setValue(1)

    def _set_volumetric_grid(self):
        # If z is collapsed, expand to a default symmetric slab
        if abs(float(self.state.field_z_max) - float(self.state.field_z_min)) < 1e-12:
            self.z_half_range_box.setValue(50.0)
        if int(self.state.field_nz) <= 1:
            self.nz_box.setValue(21)

    def _symmetric_half_range(self, min_value: float, max_value: float) -> float:
        min_abs = abs(float(min_value))
        max_abs = abs(float(max_value))
        return max(min_abs, max_abs)

    def _set_axis_symmetric_range(self, axis: str, half_range: float):
        clamped = max(0.0, float(half_range))

        if axis == 'x':
            self.state.field_x_min = -clamped
            self.state.field_x_max = clamped
        elif axis == 'y':
            self.state.field_y_min = -clamped
            self.state.field_y_max = clamped
        elif axis == 'z':
            self.state.field_z_min = -clamped
            self.state.field_z_max = clamped
