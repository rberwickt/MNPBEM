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
        self._layer_actors = dict()
        self._layer_checkboxes = dict()
        self._controls_layout = None
        self.show_layers_check = None

        self.setWindowTitle('Structure Mesh Preview')
        self.resize(960, 720)
        self.setModal(True)

        self.layout = QVBoxLayout(self)
        self.status_label = QLabel('Initializing preview...')
        self.layout.addWidget(self.status_label)

        # Layer visibility controls are created after geometry build so they
        # match the actual number of rendered layers.
        self._controls_layout = QVBoxLayout()
        self.layout.addLayout(self._controls_layout)

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
        """Toggle visibility of all rendered layers."""
        self.show_all_layers = bool(state)
        for name, checkbox in self._layer_checkboxes.items():
            checkbox.blockSignals(True)
            checkbox.setChecked(self.show_all_layers)
            checkbox.blockSignals(False)
            self._set_layer_visible(name, self.show_all_layers)

        if self.plotter is not None:
            self.plotter.render()

    def _on_single_layer_visibility_changed(self, name: str, state: int):
        """Toggle visibility of a single layer actor."""
        visible = bool(state)
        self._set_layer_visible(name, visible)

        all_visible = all(cb.isChecked() for cb in self._layer_checkboxes.values())
        if self.show_layers_check is not None:
            self.show_layers_check.blockSignals(True)
            self.show_layers_check.setChecked(all_visible)
            self.show_layers_check.blockSignals(False)
            self.show_all_layers = all_visible

        if self.plotter is not None:
            self.plotter.render()

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

            layer_entries = self._collect_layer_entries(p, pv)
            if len(layer_entries) == 0:
                raise ValueError('No previewable layer geometry found')

            self.plotter = QtInteractor(self.viewer_container)

            target_widget = getattr(self.plotter, 'interactor', self.plotter)
            self.viewer_layout.addWidget(target_widget)

            self._layer_actors.clear()
            total_verts = 0
            total_faces = 0
            for entry in layer_entries:
                mesh = entry['mesh']
                actor = self.plotter.add_mesh(
                        mesh,
                        color = entry['color'],
                        show_edges = True,
                        edge_color = '#262626',
                        line_width = 1.0,
                        smooth_shading = False)
                self._layer_actors[entry['name']] = actor
                total_verts += int(mesh.n_points)
                total_faces += int(mesh.n_cells)

            combined_mesh = layer_entries[-1]['mesh']
            self._setup_layer_controls(layer_entries)

            self._add_substrate_proxy(self.plotter, p, combined_mesh, pv)

            self.plotter.add_axes()
            self.plotter.show_grid()
            self.plotter.reset_camera()
            camera_position = self.plotter.camera_position
            if camera_position is not None:
                self._initial_camera_position = tuple(tuple(v) for v in camera_position)
            self.status_label.setText(
                    'Preview ready: {} layers, {} vertices, {} faces'.format(
                        len(layer_entries),
                        total_verts,
                        total_faces))

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

    def _collect_layer_entries(self, particle, pv):
        """Build per-layer meshes for core/shell visualization."""
        source = getattr(particle, 'pfull', particle)
        parts = list(getattr(source, 'p', []))

        if len(parts) == 0:
            return [{
                'name': 'Geometry',
                'label': 'Geometry',
                'color': '#D4C7A9',
                'mesh': self._particle_to_polydata(source, pv)}]

        layer_entries = []
        for idx, part in enumerate(parts):
            if idx == 0:
                name = 'core'
                label = 'Core'
            else:
                name = 'shell_{}'.format(idx)
                label = self._shell_label(idx)

            layer_entries.append({
                'name': name,
                'label': label,
                'color': self._layer_color(idx),
                'mesh': self._particle_to_polydata(part, pv)})

        return layer_entries

    def _shell_label(self, shell_index):
        """Human-readable shell label (1-based shell index)."""
        mat_name = None
        if shell_index - 1 < len(self.state.shells):
            mat_name = self.state.shells[shell_index - 1].get('material')

        if mat_name:
            return 'Shell {} ({})'.format(shell_index, mat_name)
        return 'Shell {}'.format(shell_index)

    def _layer_color(self, index):
        """Deterministic palette for core + shell layers."""
        if index == 0:
            return '#D4C7A9'

        shell_palette = [
            '#4E79A7',
            '#F28E2B',
            '#59A14F',
            '#E15759',
            '#76B7B2',
            '#EDC948',
            '#B07AA1',
            '#FF9DA7',
            '#9C755F',
            '#BAB0AC']
        return shell_palette[(index - 1) % len(shell_palette)]

    def _setup_layer_controls(self, layer_entries):
        """Create checkbox controls for per-layer visibility."""
        self._clear_layer_controls()

        if len(layer_entries) <= 1:
            return

        header_row = QHBoxLayout()
        self.show_layers_check = QCheckBox('Show all layers')
        self.show_layers_check.setChecked(True)
        self.show_layers_check.stateChanged.connect(self._on_layer_visibility_changed)
        header_row.addWidget(self.show_layers_check)
        header_row.addStretch()
        self._controls_layout.addLayout(header_row)

        items_row = QHBoxLayout()
        for entry in layer_entries:
            checkbox = QCheckBox(entry['label'])
            checkbox.setChecked(True)
            checkbox.stateChanged.connect(
                    lambda state, layer_name = entry['name']:
                    self._on_single_layer_visibility_changed(layer_name, state))
            self._layer_checkboxes[entry['name']] = checkbox
            items_row.addWidget(checkbox)

        items_row.addStretch()
        self._controls_layout.addLayout(items_row)

    def _clear_layer_controls(self):
        """Remove old dynamic controls before rebuilding."""
        self._layer_checkboxes.clear()
        self.show_layers_check = None

        while self._controls_layout.count() > 0:
            item = self._controls_layout.takeAt(0)
            child_layout = item.layout()
            if child_layout is not None:
                while child_layout.count() > 0:
                    child = child_layout.takeAt(0)
                    widget = child.widget()
                    if widget is not None:
                        widget.deleteLater()
                continue

            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _set_layer_visible(self, layer_name, visible):
        actor = self._layer_actors.get(layer_name)
        if actor is None:
            return

        set_visibility = getattr(actor, 'SetVisibility', None)
        if callable(set_visibility):
            set_visibility(bool(visible))
            return

        if hasattr(actor, 'visibility'):
            actor.visibility = bool(visible)

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
            self._layer_actors.clear()
            super().closeEvent(event)