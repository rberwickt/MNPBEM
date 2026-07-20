from pathlib import Path

import numpy as np

from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QLabel, QVBoxLayout,
        QWidget, QPushButton, QCheckBox, QHBoxLayout)


class StructurePreviewDialog(QDialog):
    def __init__(self, state, parent = None):
        super().__init__(parent)
        self.state = state
        self.plotter = None
        self._initial_camera_position = None
        self.show_all_layers = True  # Default: show all layers

        self.setWindowTitle('Structure Mesh Preview')
        self.resize(960, 720)
        self.setModal(True)

        self.layout = QVBoxLayout(self)
        self.status_label = QLabel('Initializing preview...')
        self.layout.addWidget(self.status_label)

        # Layer visibility control (only show if core-shell enabled)
        if self.state.core_shell_enabled and len(self.state.shells) > 0:
            control_layout = QHBoxLayout()
            self.show_layers_check = QCheckBox("Show all shell layers")
            self.show_layers_check.setChecked(True)
            self.show_layers_check.stateChanged.connect(self._on_layer_visibility_changed)
            control_layout.addWidget(self.show_layers_check)
            control_layout.addStretch()
            self.layout.addLayout(control_layout)

        self.viewer_container = QWidget(self)
        self.viewer_layout = QVBoxLayout(self.viewer_container)
        self.viewer_layout.setContentsMargins(0, 0, 0, 0)
        self.layout.addWidget(self.viewer_container, 1)

        self.button_box = QDialogButtonBox(QDialogButtonBox.Close)
        self.button_box.rejected.connect(self.reject)

        self.reset_button = QPushButton("Reset View")
        self.reset_button.clicked.connect(self.reset_view)
        self.button_box.addButton(self.reset_button, QDialogButtonBox.ActionRole)

        self.layout.addWidget(self.button_box)

        self._build_preview_scene()

    def reset_view(self):
        """Re-centers the camera view to the initial position."""
        if self.plotter is not None:
            if self._initial_camera_position is not None:
                self.plotter.camera_position = self._initial_camera_position
            else:
                self.plotter.reset_camera()

            reset_clip = getattr(self.plotter, 'reset_camera_clipping_range', None)
            if callable(reset_clip):
                reset_clip()

            self.plotter.render()  # Forces the interactor to redraw the scene immediately

    def _on_layer_visibility_changed(self, state: int):
        """Handle layer visibility toggle (informational for now).
        
        Currently the preview always shows combined geometry (core + all shells)
        via ComParticle. This toggle is for future enhancement to show individual
        shell layers with different colors.
        """
        self.show_all_layers = bool(state)

    def _build_preview_scene(self):
        try:
            import pyvista as pv
            from pyvistaqt import QtInteractor
        except Exception as exc:
            self.status_label.setText(
                    'PyVista preview unavailable: {}'.format(exc))
            return

        try:
            from pymnpbem_simulation.config import apply_defaults
            from pymnpbem_simulation.structures import build_structure

            cfg = self.state.to_dict(
                    output_dir = str(Path('.') / 'tmp'),
                    output_name = 'mesh_preview')
            cfg = apply_defaults(cfg)

            p, _, _ = build_structure(
                    cfg['structure'],
                    cfg.get('materials', dict()))

            mesh = self._particle_to_polydata(p, pv)
            self.plotter = QtInteractor(self.viewer_container)

            target_widget = getattr(self.plotter, 'interactor', self.plotter)
            self.viewer_layout.addWidget(target_widget)

            self.plotter.add_mesh(
                    mesh,
                    color = '#D4C7A9',
                    show_edges = True,
                    edge_color = '#262626',
                    line_width = 1.0,
                    smooth_shading = False)

            self._add_substrate_proxy(self.plotter, p, mesh, pv)

            self.plotter.add_axes()
            self.plotter.show_grid()
            self.plotter.reset_camera()
            camera_position = self.plotter.camera_position
            if camera_position is not None:
                self._initial_camera_position = tuple(tuple(v) for v in camera_position)
            self.status_label.setText(
                    'Preview ready: {} vertices, {} faces'.format(
                            mesh.n_points,
                            mesh.n_cells))

        except Exception as exc:
            self.status_label.setText(
                    'Failed to build mesh preview: {}'.format(exc))

    def _particle_to_polydata(self, particle, pv):
        verts = np.asarray(particle.verts, dtype = float)
        faces = np.asarray(particle.faces, dtype = float)

        vtk_faces = []
        for row in faces:
            idx = [int(v) for v in row if not np.isnan(v)]
            if len(idx) >= 3:
                vtk_faces.extend([len(idx)] + idx)

        if len(vtk_faces) == 0:
            raise ValueError('No valid faces found for preview geometry')

        return pv.PolyData(verts, np.asarray(vtk_faces, dtype = np.int64))

    def _add_substrate_proxy(self, plotter, particle, mesh, pv):
        layer = getattr(particle, '_mnpbem_layer', None)
        if layer is None and hasattr(particle, 'pfull'):
            layer = getattr(particle.pfull, '_mnpbem_layer', None)

        if layer is None or not hasattr(layer, 'z') or len(layer.z) == 0:
            return

        substrate_z = float(layer.z[0])
        x_min, x_max, y_min, y_max, _, _ = mesh.bounds
        dx = max(1.0, x_max - x_min)
        dy = max(1.0, y_max - y_min)
        margin = 0.3 * max(dx, dy)

        plane = pv.Plane(
                center = ((x_min + x_max) * 0.5,
                          (y_min + y_max) * 0.5,
                          substrate_z),
                direction = (0.0, 0.0, 1.0),
                i_size = dx + 2.0 * margin,
                j_size = dy + 2.0 * margin)

        plotter.add_mesh(
                plane,
                color = '#4E79A7',
                opacity = 0.25,
                show_edges = False)

    def closeEvent(self, event):
        try:
            if self.plotter is not None:
                close_method = getattr(self.plotter, 'close', None)
                if callable(close_method):
                    close_method()
        finally:
            self.plotter = None
            super().closeEvent(event)