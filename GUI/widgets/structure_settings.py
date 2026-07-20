from PySide6.QtWidgets import (QGroupBox, QFormLayout, QComboBox, QTabWidget, 
                               QWidget, QVBoxLayout, QHBoxLayout, QDoubleSpinBox, QSpinBox, QStackedWidget, QPushButton,
                               QScrollArea, QLabel)
from PySide6.QtCore import Qt, Signal
#from PySide6.QtGui import QIntValidator
from ..simulation_state import SimulationState
from .material_dropdown import MaterialComboBox
from .structure_view import StructurePreviewDialog

class StructureSettingsWidget(QGroupBox):
    def __init__(self, state: SimulationState, parent=None):
        super().__init__("Structure Settings", parent)
        self.state = state  # Keep a reference to the data struct

        self.layout = QVBoxLayout(self)
        form_layout = QFormLayout()
        self.geo_combo = QComboBox()
        self.geo_combo.addItems(["Sphere", "Rod", "Cube"])
        self.geo_combo.setCurrentText(self.state.structure)
        self.geo_combo.currentTextChanged.connect(self._on_geo_changed)
        form_layout.addRow("Shape:", self.geo_combo)

        self.geo_mat = MaterialComboBox(self.state)
        if len(self.state.materials) < 1:
            self.state.materials.append(None)
        # hard coding index zero for now (which should be fine?)
        self.geo_mat.currentTextChanged.connect(lambda mat: self.state.materials.__setitem__(0, mat))
        form_layout.addRow("Core Material:", self.geo_mat)

        self.refine = QSpinBox()
        self.refine.setRange(1,100)
        self.refine.setValue(self.state.refine)
        self.refine.valueChanged.connect(lambda val: setattr(self.state, 'refine', val))
        form_layout.addRow("Refine:", self.refine)

        self.layout.addLayout(form_layout)

        self.settings_group = QGroupBox("Shape Settings (Sphere)")
        settings_layout = QVBoxLayout(self.settings_group)
        self.stacked_widget = QStackedWidget()
        settings_layout.addWidget(self.stacked_widget)
        # Sphere Settings =============================================
        self.sphere_settings = QWidget()
        sphere_layout = QFormLayout(self.sphere_settings)
        self.diameter1 = QDoubleSpinBox()
        self.diameter1.setRange(1.0, 2000.0)
        self.diameter1.setSuffix(" nm")
        self.diameter1.setValue(float(self.state.diameter))
        self.diameter1.valueChanged.connect(lambda val: setattr(self.state, 'diameter', val))
        sphere_layout.addRow("Diameter:", self.diameter1)
        self.sphere_n_verts = QSpinBox()
        self.sphere_n_verts.setRange(16, 200000)
        self.sphere_n_verts.setSuffix(" vertices")
        self.sphere_n_verts.setValue(int(self.state.sphere_n_verts))
        self.sphere_n_verts.setToolTip("Sphere mesh resolution as trisphere vertex count.")
        self.sphere_n_verts.valueChanged.connect(lambda val: setattr(self.state, 'sphere_n_verts', int(val)))
        sphere_layout.addRow("Vertices:", self.sphere_n_verts)
        # Rod Settings ================================================
        self.rod_settings = QWidget()
        rod_layout = QFormLayout(self.rod_settings)
        self.diameter2 = QDoubleSpinBox()
        self.diameter2.setRange(1.0, 2000.0)
        self.diameter2.setSuffix(" nm")
        self.diameter2.setValue(float(self.state.diameter))
        self.diameter2.valueChanged.connect(lambda val: setattr(self.state, 'diameter', val))
        rod_layout.addRow("Short Axis:", self.diameter2)
        self.rod_mesh_size = QDoubleSpinBox()
        self.rod_mesh_size.setRange(0.1, 2000.0)
        self.rod_mesh_size.setDecimals(2)
        self.rod_mesh_size.setSingleStep(0.1)
        self.rod_mesh_size.setSuffix(" nm")
        self.rod_mesh_size.setValue(float(self.state.mesh_element_size_nm))
        self.rod_mesh_size.setToolTip("Target element size (nm). Smaller values create finer meshes and more faces.")
        self.rod_mesh_size.valueChanged.connect(self._on_mesh_size_changed)
        rod_layout.addRow("Mesh Element Size:", self.rod_mesh_size)
        # connect the diameter spinners for consistency
        self.diameter1.valueChanged.connect(self.diameter2.setValue)
        self.diameter2.valueChanged.connect(self.diameter1.setValue)

        self.height_box = QDoubleSpinBox()
        self.height_box.setRange(1.0, 2000.0)
        self.height_box.setSuffix(" nm")
        self.height_box.setValue(float(self.state.height))
        self.height_box.valueChanged.connect(lambda val: setattr(self.state, 'height', val))
        rod_layout.addRow("Long Axis:", self.height_box)
        
        # Cube Settings ===============================================
        self.cube_settings = QWidget()
        cube_layout = QFormLayout(self.cube_settings)
        self.size_box = QDoubleSpinBox()
        self.size_box.setRange(1.0, 2000.0)
        self.size_box.setSuffix(" nm")
        self.size_box.setValue(float(self.state.size))
        self.size_box.valueChanged.connect(lambda val: setattr(self.state, 'size', val))
        cube_layout.addRow("Edge Length:", self.size_box)

        self.cube_mesh_size = QDoubleSpinBox()
        self.cube_mesh_size.setRange(0.1, 2000.0)
        self.cube_mesh_size.setDecimals(2)
        self.cube_mesh_size.setSingleStep(0.1)
        self.cube_mesh_size.setSuffix(" nm")
        self.cube_mesh_size.setValue(float(self.state.mesh_element_size_nm))
        self.cube_mesh_size.setToolTip("Target element size (nm). Smaller values create finer meshes and more faces.")
        self.cube_mesh_size.valueChanged.connect(self._on_mesh_size_changed)
        cube_layout.addRow("Mesh Element Size:", self.cube_mesh_size)

        #self.n_per_edge = QSpinBox()
        #self.n_per_edge.setRange(1,100)
        #self.n_per_edge.setValue(int(self.state.n_per_edge))
        #self.n_per_edge.valueChanged.connect(lambda val: setattr(self.state, 'n_per_edge', val))
        #cube_layout.addRow("Divisions Per Edge:", self.n_per_edge)
        # TODO: Allow for shells (using a system like the old material_dropdown widgets)

        self.stacked_widget.addWidget(self.sphere_settings)
        self.stacked_widget.addWidget(self.rod_settings)
        self.stacked_widget.addWidget(self.cube_settings)
        self.layout.addWidget(self.settings_group)

        # Shells Panel (independent, below shape settings)
        self.shells_panel = ShellsPanel(self.state, parent=self)
        self.shells_panel.shells_changed.connect(self._on_shells_changed)
        self.layout.addWidget(self.shells_panel)

        self.preview_btn = QPushButton("Preview Mesh")
        self.preview_btn.clicked.connect(self._on_preview_clicked)
        self.layout.addWidget(self.preview_btn)

        self._on_geo_changed(self.state.structure)

    def _on_geo_changed(self, text: str):
        self.state.structure = text
        if text == "Sphere":
            self.settings_group.setTitle("Shape Settings (Sphere)")
            self.stacked_widget.setCurrentIndex(0)
        elif text == "Rod":
            self.settings_group.setTitle("Shape Settings (Rod)")
            self.stacked_widget.setCurrentIndex(1)
            self.state.horizontal = True
        elif text == "Cube":
            self.settings_group.setTitle("Shape Settings (Cube)")
            self.stacked_widget.setCurrentIndex(2)
            self.state.horizontal = False
        else:
            self.state.horizontal = False

    def _on_mesh_size_changed(self, val: float):
        mesh_val = float(val)
        self.state.mesh_element_size_nm = mesh_val
        if hasattr(self, 'rod_mesh_size') and self.rod_mesh_size.value() != mesh_val:
            self.rod_mesh_size.blockSignals(True)
            self.rod_mesh_size.setValue(mesh_val)
            self.rod_mesh_size.blockSignals(False)
        if hasattr(self, 'cube_mesh_size') and self.cube_mesh_size.value() != mesh_val:
            self.cube_mesh_size.blockSignals(True)
            self.cube_mesh_size.setValue(mesh_val)
            self.cube_mesh_size.blockSignals(False)

    def _on_preview_clicked(self):
        dialog = StructurePreviewDialog(self.state, self)
        dialog.exec()

    def _on_shells_changed(self):
        """Handle shells panel changes (for future sync/validation if needed)."""
        # Currently just ensures state is kept in sync via shells_panel signals
        # Can be extended later for additional UI updates
        pass
        
class ShellsPanel(QGroupBox):
    """Independent shells management panel for core-shell structures.
    
    Uses a custom list layout instead of QTableWidget to avoid widget rendering issues.
    Each shell row: [Thickness] [Material] [Remove]
    Shells automatically inherit core mesh density from core shape.
    """
    shells_changed = Signal()  # Emitted when shells list is modified
    
    def __init__(self, state, parent=None):
        super().__init__("Shells (Optional)", parent)
        self.state = state
        self.shell_rows = []  # List of ShellRowWidget objects
        self.max_visible_shell_rows = 4  # Changed to scroll after 4 entries
        
        layout = QVBoxLayout(self)
        
        # Header row
        header = QHBoxLayout()
        header.setContentsMargins(2, 2, 2, 2)
        header.setSpacing(6)
        
        num_label = QLabel("#")
        num_label.setAlignment(Qt.AlignCenter)
        num_label.setFixedWidth(24)
        
        header.addWidget(num_label, 0)
        header.addWidget(QLabel("Thickness"), 1)
        header.addWidget(QLabel("Material"), 1)
        
        # Add a dummy label to account for the remove button's width, keeping columns aligned
        dummy_label = QLabel("")
        dummy_label.setFixedWidth(36)
        header.addWidget(dummy_label, 0)

        layout.addLayout(header)
        
        # Scrollable container for shell rows
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.shells_container = QWidget()
        self.shells_layout = QVBoxLayout(self.shells_container)
        self.shells_layout.setContentsMargins(0, 0, 0, 0)
        self.shells_layout.setSpacing(1)
        scroll.setWidget(self.shells_container)
        self.shells_scroll = scroll
        layout.addWidget(self.shells_scroll)
        
        # Add Shell button
        btn_layout = QHBoxLayout()
        self.add_shell_btn = QPushButton("+ Add Shell")
        self.add_shell_btn.clicked.connect(self._on_add_shell)
        btn_layout.addWidget(self.add_shell_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)
        
        self._refresh_shells()
    
    def _refresh_shells(self):
        """Rebuild shell rows from state."""
        # Clear existing rows/layout items to avoid accumulating stretch spacers.
        while self.shells_layout.count():
            item = self.shells_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        self.shell_rows = []
        
        # Create new rows
        for i, shell in enumerate(self.state.shells):
            row = ShellRowWidget(i, shell, self.state)
            row.thickness_changed.connect(lambda val, idx=i: self._on_thickness_changed(idx, val))
            row.material_changed.connect(lambda mat, idx=i: self._on_material_changed(idx, mat))
            row.remove_clicked.connect(lambda checked=False, idx=i: self._on_remove_shell(idx))
            self.shells_layout.addWidget(row)
            self.shell_rows.append(row)
        
        self.shells_layout.addStretch()
        self._update_scroll_height()

    def _update_scroll_height(self):
        """Keep scroll area at a constant height for max_visible_shell_rows."""
        row_height = 30
        fixed_rows = self.max_visible_shell_rows  # Always use 4 rows for the calculation
        frame_padding = 4
        layout_spacing = max(0, fixed_rows - 1) * self.shells_layout.spacing()
        
        # Calculate exact height needed for 4 rows + padding + spacing
        target_height = (fixed_rows * row_height) + frame_padding + layout_spacing
        
        # This locks the height permanently so the box never shrinks
        self.shells_scroll.setFixedHeight(target_height)
    
    def _on_thickness_changed(self, idx: int, val: float):
        if idx < len(self.state.shells):
            self.state.shells[idx]['thickness'] = float(val)
            self.shells_changed.emit()
    
    def _on_material_changed(self, idx: int, material: str):
        if idx < len(self.state.shells):
            self.state.shells[idx]['material'] = str(material)
            self.shells_changed.emit()
    
    def _on_add_shell(self):
        """Add a new shell with default values."""
        new_shell = {
            'thickness': 1.0,
            'material': 'silver'
        }
        self.state.shells.append(new_shell)
        self.state.core_shell_enabled = True
        self._refresh_shells()
        self.shells_changed.emit()
    
    def _on_remove_shell(self, idx: int):
        """Remove shell at given index."""
        if 0 <= idx < len(self.state.shells):
            del self.state.shells[idx]
            self.state.core_shell_enabled = len(self.state.shells) > 0
            self._refresh_shells()
            self.shells_changed.emit()
    
    def refresh(self):
        """Refresh rows from state (call after external state changes)."""
        self._refresh_shells()


class ShellRowWidget(QWidget):
    """Single shell row with all controls."""
    thickness_changed = Signal(float)
    material_changed = Signal(str)
    remove_clicked = Signal()
    
    def __init__(self, idx: int, shell: dict, state, parent=None):
        super().__init__(parent)
        self.idx = idx
        self.shell = shell
        self.state = state
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 1, 2, 1)
        layout.setSpacing(6)

        self.index_label = QLabel("{}".format(self.idx + 1))
        self.index_label.setAlignment(Qt.AlignCenter)
        self.index_label.setFixedWidth(24)
        layout.addWidget(self.index_label, 0)
        
        # Thickness spinbox
        self.thick_spin = QDoubleSpinBox()
        self.thick_spin.setRange(0.1, 2000.0)
        self.thick_spin.setSuffix(" nm")
        self.thick_spin.setValue(float(shell.get('thickness', 1.0)))
        self.thick_spin.setMaximumHeight(26)
        self.thick_spin.valueChanged.connect(lambda val: self.thickness_changed.emit(val))
        layout.addWidget(self.thick_spin, 1)
        
        # Material dropdown
        self.mat_combo = MaterialComboBox(self.state)
        self.mat_combo.setMaximumHeight(26)
        self.mat_combo.currentTextChanged.connect(lambda mat: self.material_changed.emit(mat))
        layout.addWidget(self.mat_combo, 1)
        
        # Remove button
        self.remove_btn = QPushButton("X")
        self.remove_btn.setStyleSheet("padding: 0px; margin: 0px;")
        self.remove_btn.setToolTip("Remove shell")
        
        # Increased width to 36 so text isn't clipped by button padding
        self.remove_btn.setMinimumWidth(36)
        self.remove_btn.setMaximumWidth(36)
        self.remove_btn.setMaximumHeight(26)
        
        self.remove_btn.clicked.connect(self.remove_clicked.emit)
        layout.addWidget(self.remove_btn, 0)