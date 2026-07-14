from PySide6.QtWidgets import (QGroupBox, QFormLayout, QComboBox, QTabWidget, 
                               QWidget, QVBoxLayout, QHBoxLayout, QDoubleSpinBox, QSpinBox, QCheckBox, QStackedWidget, QPushButton)
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
        form_layout.addRow("Material:", self.geo_mat)

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
        # Rod Settings ================================================
        self.rod_settings = QWidget()
        rod_layout = QFormLayout(self.rod_settings)
        self.diameter2 = QDoubleSpinBox()
        self.diameter2.setRange(1.0, 2000.0)
        self.diameter2.setSuffix(" nm")
        self.diameter2.setValue(float(self.state.diameter))
        self.diameter2.valueChanged.connect(lambda val: setattr(self.state, 'diameter', val))
        rod_layout.addRow("Diameter:", self.diameter2)
        # connect the diameter spinners for consistency
        self.diameter1.valueChanged.connect(self.diameter2.setValue)
        self.diameter2.valueChanged.connect(self.diameter1.setValue)

        self.height_box = QDoubleSpinBox()
        self.height_box.setRange(1.0, 2000.0)
        self.height_box.setSuffix(" nm")
        self.height_box.setValue(float(self.state.height))
        self.height_box.valueChanged.connect(lambda val: setattr(self.state, 'height', val))
        rod_layout.addRow("Height:", self.height_box)
        
        # Cube Settings ===============================================
        self.cube_settings = QWidget()
        cube_layout = QFormLayout(self.cube_settings)
        self.size_box = QDoubleSpinBox()
        self.size_box.setRange(1.0, 2000.0)
        self.size_box.setSuffix(" nm")
        self.size_box.setValue(float(self.state.size))
        self.size_box.valueChanged.connect(lambda val: setattr(self.state, 'size', val))
        cube_layout.addRow("Size:", self.size_box)

        self.n_per_edge = QSpinBox()
        self.n_per_edge.setRange(1,100)
        self.n_per_edge.setValue(int(self.state.n_per_edge))
        self.n_per_edge.valueChanged.connect(lambda val: setattr(self.state, 'n_per_edge', val))
        cube_layout.addRow("Divisions Per Edge:", self.n_per_edge)
        # TODO: Allow for shells (using a system like the old material_dropdown widgets)

        self.stacked_widget.addWidget(self.sphere_settings)
        self.stacked_widget.addWidget(self.rod_settings)
        self.stacked_widget.addWidget(self.cube_settings)
        self.layout.addWidget(self.settings_group)

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

    def _on_preview_clicked(self):
        dialog = StructurePreviewDialog(self.state, self)
        dialog.exec()
        
