import json

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QGridLayout,
    QLabel,
    QMessageBox,
    QFileDialog,
    QComboBox,
    QDoubleSpinBox,
    QSpinBox,
    QCheckBox,
    QPushButton,
    QTabWidget,
    QFrame,
)
 # TODO: Divide by E_0 for the field intensity normalization, since right now we don't
 # generally this is fine since the amplitude is 1, but if the user changes the amplitude, 
 # # the field intensity will be off by a factor of E_0^2.
from ..simulation_state import SimulationState
from ..widgets.calculation_figure import CalculationFigure

from matplotlib.figure import Figure
from matplotlib.colors import LogNorm

import numpy as np

from pymnpbem_simulation.postprocess import analyze_spectrum, field_statistics, hotspot_summary
from pymnpbem_simulation.io import save_spectrum, save_field


class ProcessingPage(QWidget):

    def __init__(self, state: SimulationState):
        super().__init__()
        self.state = state
        self.processed_outputs = dict()
        self._field_controls = dict()

        self._spectrum_summary_label = None
        self._field_summary_label = None
        self._spectrum_figure_layout = None
        self._field_figure_layout = None
        self._spectrum_figure_widget = None
        self._field_figure_widget = None

        self._btn_spec_run = None
        self._btn_spec_save_processed = None
        self._btn_spec_save_raw = None
        self._spec_pol_selector = None
        self._btn_field_run = None
        self._btn_field_save_processed = None
        self._btn_field_save_raw = None

        self._build_ui()
        self.setup_ui_from_state()

    def _build_ui(self):
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(8, 8, 8, 8)
        root_layout.setSpacing(8)

        self.tabs = QTabWidget(self)
        root_layout.addWidget(self.tabs)

        self._build_spectrum_tab()
        self._build_field_tab()

    def _build_spectrum_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        intro = QLabel(
            "Spectrum post-processing from raw simulation output. "
            "Computes peaks/FWHM and renders a live figure."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        action_row = QHBoxLayout()
        self._btn_spec_run = QPushButton("Run Analysis")
        self._btn_spec_save_processed = QPushButton("Save Processed")
        self._btn_spec_save_raw = QPushButton("Save Raw")

        self._spec_pol_selector = QComboBox()
        self._spec_pol_selector.addItem("All Polarizations", userData = -1)
        self._spec_pol_selector.setEnabled(False)
        self._spec_pol_selector.currentIndexChanged.connect(self._on_spectrum_selector_changed)

        self._btn_spec_run.clicked.connect(self._run_spectrum_analysis)
        self._btn_spec_save_processed.clicked.connect(
            lambda _checked = False: self._save_processed_output("spectrum"))
        self._btn_spec_save_raw.clicked.connect(
            lambda _checked = False: self._save_raw_output("spectrum"))

        action_row.addWidget(self._btn_spec_run)
        action_row.addWidget(self._btn_spec_save_processed)
        action_row.addWidget(self._btn_spec_save_raw)
        action_row.addWidget(QLabel("View:"))
        action_row.addWidget(self._spec_pol_selector)
        action_row.addStretch(1)
        layout.addLayout(action_row)

        self._spectrum_summary_label = QLabel("No spectrum analysis run yet.")
        self._spectrum_summary_label.setWordWrap(True)
        layout.addWidget(self._spectrum_summary_label)

        figure_frame = QFrame()
        self._spectrum_figure_layout = QVBoxLayout(figure_frame)
        self._spectrum_figure_layout.setContentsMargins(0, 0, 0, 0)
        self._spectrum_figure_layout.setSpacing(0)
        layout.addWidget(figure_frame, 1)

        self.tabs.addTab(tab, "Spectra")

    def _build_field_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        intro = QLabel(
            "Field post-processing from raw simulation output. "
            "Requires field arrays (pos/e). Use controls below to choose 2D/3D style and plotting options."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        action_row = QHBoxLayout()
        self._btn_field_run = QPushButton("Run Analysis")
        self._btn_field_save_processed = QPushButton("Save Processed")
        self._btn_field_save_raw = QPushButton("Save Raw")

        self._btn_field_run.clicked.connect(self._run_field_analysis)
        self._btn_field_save_processed.clicked.connect(
            lambda _checked = False: self._save_processed_output("field"))
        self._btn_field_save_raw.clicked.connect(
            lambda _checked = False: self._save_raw_output("field"))

        action_row.addWidget(self._btn_field_run)
        action_row.addWidget(self._btn_field_save_processed)
        action_row.addWidget(self._btn_field_save_raw)
        action_row.addStretch(1)
        layout.addLayout(action_row)

        opts = QWidget()
        opts_layout = QVBoxLayout(opts)
        opts_layout.setContentsMargins(0, 0, 0, 0)
        opts_layout.setSpacing(6)

        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(4)

        view_mode = QComboBox()
        view_mode.addItems(["Auto", "2D", "3D", "Scatter"])

        slice_axis = QComboBox()
        slice_axis.addItems(["x", "y", "z"])
        slice_axis.setCurrentText("z")

        slice_value = QDoubleSpinBox()
        slice_value.setRange(-1e6, 1e6)
        slice_value.setDecimals(3)
        slice_value.setSingleStep(1.0)
        slice_value.setValue(0.0)

        wavelength_idx = QSpinBox()
        wavelength_idx.setRange(0, 1000000)
        wavelength_idx.setValue(0)

        pol_idx = QSpinBox()
        pol_idx.setRange(-1, 1000000)
        pol_idx.setSpecialValueText("Avg (All)")
        pol_idx.setToolTip("Set to -1 to average all polarizations (unpolarized view).")
        pol_idx.setValue(0)

        log_scale = QCheckBox("Enable log color scale")
        log_scale.setChecked(True)

        clip_percentile = QCheckBox("Use percentile clipping")
        clip_percentile.setChecked(True)

        p_low = QDoubleSpinBox()
        p_low.setRange(0.0, 99.0)
        p_low.setDecimals(1)
        p_low.setSingleStep(0.5)
        p_low.setValue(2.0)

        p_high = QDoubleSpinBox()
        p_high.setRange(1.0, 100.0)
        p_high.setDecimals(1)
        p_high.setSingleStep(0.5)
        p_high.setValue(99.5)

        cmap = QComboBox()
        cmap.addItems(["inferno", "viridis", "magma", "plasma", "cividis"])
        cmap.setCurrentText("inferno")

        vector_overlay = QCheckBox("Overlay 2D field vectors")
        vector_overlay.setChecked(False)

        vector_density = QSpinBox()
        vector_density.setRange(1, 32)
        vector_density.setValue(4)

        manual_log_bounds = QCheckBox("Use manual log bounds")
        manual_log_bounds.setChecked(False)

        log_vmin = QDoubleSpinBox()
        log_vmin.setRange(1e-20, 1e20)
        log_vmin.setDecimals(8)
        log_vmin.setSingleStep(0.1)
        log_vmin.setValue(1e-6)
        log_vmin.setEnabled(False)

        log_vmax = QDoubleSpinBox()
        log_vmax.setRange(1e-20, 1e20)
        log_vmax.setDecimals(8)
        log_vmax.setSingleStep(0.1)
        log_vmax.setValue(1.0)
        log_vmax.setEnabled(False)

        manual_log_bounds.toggled.connect(
            lambda checked: self._set_log_bounds_controls_enabled(bool(checked), log_vmin, log_vmax)
        )

        hotspot_quantile = QDoubleSpinBox()
        hotspot_quantile.setRange(0.50, 0.9999)
        hotspot_quantile.setDecimals(4)
        hotspot_quantile.setSingleStep(0.005)
        hotspot_quantile.setValue(0.99)

        row0 = 0
        grid.addWidget(QLabel("View"), row0, 0)
        grid.addWidget(view_mode, row0, 1)
        grid.addWidget(QLabel("Wavelength Idx"), row0, 2)
        grid.addWidget(wavelength_idx, row0, 3)
        grid.addWidget(QLabel("Pol Idx"), row0, 4)
        grid.addWidget(pol_idx, row0, 5)
        grid.addWidget(QLabel("Colormap"), row0, 6)
        grid.addWidget(cmap, row0, 7)

        row1 = 1
        grid.addWidget(QLabel("Slice Axis"), row1, 0)
        grid.addWidget(slice_axis, row1, 1)
        grid.addWidget(QLabel("Slice Value (nm)"), row1, 2)
        grid.addWidget(slice_value, row1, 3)
        grid.addWidget(QLabel("P Low"), row1, 4)
        grid.addWidget(p_low, row1, 5)
        grid.addWidget(QLabel("P High"), row1, 6)
        grid.addWidget(p_high, row1, 7)

        row2 = 2
        grid.addWidget(QLabel("Vector Density"), row2, 0)
        grid.addWidget(vector_density, row2, 1)
        grid.addWidget(QLabel("Hotspot q"), row2, 2)
        grid.addWidget(hotspot_quantile, row2, 3)
        grid.addWidget(log_scale, row2, 4, 1, 2)
        grid.addWidget(clip_percentile, row2, 6, 1, 2)

        row3 = 3
        grid.addWidget(vector_overlay, row3, 0, 1, 3)
        grid.addWidget(manual_log_bounds, row3, 3, 1, 2)
        grid.addWidget(QLabel("Log vmin"), row3, 5)
        grid.addWidget(log_vmin, row3, 6)
        grid.addWidget(QLabel("Log vmax"), row3, 7)
        grid.addWidget(log_vmax, row3, 8)

        for col in range(9):
            grid.setColumnStretch(col, 1 if col % 2 == 1 else 0)

        opts_layout.addLayout(grid)

        self._field_controls = {
            "view_mode": view_mode,
            "slice_axis": slice_axis,
            "slice_value": slice_value,
            "wavelength_idx": wavelength_idx,
            "pol_idx": pol_idx,
            "log_scale": log_scale,
            "clip_percentile": clip_percentile,
            "p_low": p_low,
            "p_high": p_high,
            "cmap": cmap,
            "vector_overlay": vector_overlay,
            "vector_density": vector_density,
            "hotspot_quantile": hotspot_quantile,
            "manual_log_bounds": manual_log_bounds,
            "log_vmin": log_vmin,
            "log_vmax": log_vmax,
        }

        layout.addWidget(opts)

        self._field_summary_label = QLabel("No field analysis run yet.")
        self._field_summary_label.setWordWrap(True)
        layout.addWidget(self._field_summary_label)

        figure_frame = QFrame()
        self._field_figure_layout = QVBoxLayout(figure_frame)
        self._field_figure_layout.setContentsMargins(0, 0, 0, 0)
        self._field_figure_layout.setSpacing(0)
        layout.addWidget(figure_frame, 1)

        self.tabs.addTab(tab, "Fields")

    def setup_ui_from_state(self):
        """Called by main window right before switching to this page."""
        self.processed_outputs = dict()
        self._set_result_figure("spectrum", None)
        self._set_result_figure("field", None)

        raw = self.state.raw_results
        self._refresh_spectrum_selector(raw)
        if raw is None:
            msg = (
                "No simulation result loaded. Run a simulation first. "
                "Post-processing uses raw simulation output generated after the run."
            )
            self._spectrum_summary_label.setText(msg)
            self._field_summary_label.setText(msg)
            self._set_buttons_enabled(False)
            return

        self._set_buttons_enabled(True)

        if not self._has_spectrum_data(raw):
            self._spectrum_summary_label.setText(
                "Spectrum arrays are missing from current raw result (need wavelength/ext/sca/abs)."
            )
        else:
            self._spectrum_summary_label.setText("Ready. Click Run Analysis to compute spectrum summary.")

        if not self._has_field_data(raw):
            self._field_summary_label.setText(
                "Field arrays are missing from current raw result (need wavelength/pos/e). "
                "Rerun simulation with field calculation enabled."
            )
        else:
            self._field_summary_label.setText("Ready. Click Run Analysis to compute field statistics and maps.")

    def _set_buttons_enabled(self, enabled: bool):
        for btn in (
            self._btn_spec_run,
            self._btn_spec_save_processed,
            self._btn_spec_save_raw,
            self._btn_field_run,
            self._btn_field_save_processed,
            self._btn_field_save_raw,
        ):
            if btn is not None:
                btn.setEnabled(enabled)

    def _set_result_figure(self, kind: str, figure: Figure | None):
        if kind == "spectrum":
            layout = self._spectrum_figure_layout
            old_widget = self._spectrum_figure_widget
        else:
            layout = self._field_figure_layout
            old_widget = self._field_figure_widget

        if old_widget is not None:
            old_widget.setParent(None)
            old_widget.deleteLater()

        if figure is None:
            if kind == "spectrum":
                self._spectrum_figure_widget = None
            else:
                self._field_figure_widget = None
            return

        fig_widget = CalculationFigure(figure)
        layout.addWidget(fig_widget)

        if kind == "spectrum":
            self._spectrum_figure_widget = fig_widget
        else:
            self._field_figure_widget = fig_widget

    def _run_spectrum_analysis(self):
        raw = self.state.raw_results
        if raw is None or not self._has_spectrum_data(raw):
            QMessageBox.warning(
                self,
                "Spectrum Analysis",
                "Spectrum analysis requires raw simulation arrays: wavelength/ext/sca/abs.",
            )
            return

        try:
            summary = analyze_spectrum(raw)
            self.processed_outputs["spectrum"] = summary

            fig = self._build_spectrum_figure(raw, pol_idx = self._selected_spectrum_pol())
            self._spectrum_summary_label.setText(self._format_spectrum_summary(summary))
            self._set_result_figure("spectrum", fig)
        except Exception as exc:
            QMessageBox.critical(self, "Spectrum Analysis Failed", str(exc))

    def _on_spectrum_selector_changed(self, _idx: int):
        raw = self.state.raw_results
        if raw is None or not self._has_spectrum_data(raw):
            return
        try:
            fig = self._build_spectrum_figure(raw, pol_idx = self._selected_spectrum_pol())
            self._set_result_figure("spectrum", fig)
        except Exception:
            # Keep UI responsive even if selection redraw fails.
            pass

    def _run_field_analysis(self):
        raw = self.state.raw_results
        if raw is None or not self._has_field_data(raw):
            QMessageBox.information(
                self,
                "Field Data Needed",
                "Field post-processing is available after simulation, but it still needs raw field arrays (pos/e).\n\n"
                "Current run appears spectrum-only. Rerun the simulation with field calculation enabled, then analyze here.",
            )
            return

        try:
            plot_opts = self._get_field_plot_options()
            
            # Get grid bounds info for diagnostics
            pos = np.asarray(raw.get("pos", []))
            if pos.size > 0:
                x_min, x_max = float(np.min(pos[:, 0])), float(np.max(pos[:, 0]))
                y_min, y_max = float(np.min(pos[:, 1])), float(np.max(pos[:, 1]))
                z_min, z_max = float(np.min(pos[:, 2])), float(np.max(pos[:, 2]))
                
                slice_axis = str(plot_opts.get("slice_axis", "z")).lower()
                slice_value = float(plot_opts.get("slice_value", 0.0))
                
                # Check if slice value is at center of grid
                axis_map = {"x": 0, "y": 1, "z": 2}
                ax_idx = axis_map.get(slice_axis, 2)
                axis_bounds = [(x_min, x_max), (y_min, y_max), (z_min, z_max)]
                axis_min, axis_max = axis_bounds[ax_idx]
                axis_center = (axis_min + axis_max) / 2.0
                
                # If slice is far from center, warn user
                if abs(slice_value - axis_center) > 0.1 * (axis_max - axis_min):
                    info_msg = (
                        "Field Grid Info:\n"
                        "  X: [{:.2f}, {:.2f}] nm (center: {:.2f})\n"
                        "  Y: [{:.2f}, {:.2f}] nm (center: {:.2f})\n"
                        "  Z: [{:.2f}, {:.2f}] nm (center: {:.2f})\n\n"
                        "Current slice ({} = {:.2f} nm) is far from axis center ({:.2f} nm).\n"
                        "Consider setting slice to axis center for best results."
                    ).format(
                        x_min, x_max, (x_min + x_max) / 2.0,
                        y_min, y_max, (y_min + y_max) / 2.0,
                        z_min, z_max, (z_min + z_max) / 2.0,
                        slice_axis, slice_value, axis_center
                    )
                    QMessageBox.information(self, "Slice Position Info", info_msg)
            
            field_slice, meta = self._extract_field_slice(
                raw,
                wl_idx = int(plot_opts["wavelength_idx"]),
                pol_idx = int(plot_opts["polarization_idx"]),
            )
            stats = field_statistics(field_slice)
            hotspots = hotspot_summary(
                field_slice,
                threshold_quantile = float(plot_opts["hotspot_quantile"]),
            )

            # percentiles of graph 
            pos = np.asarray(field_slice["pos"])
            e = np.asarray(field_slice["e"])
            intensity = self._compute_intensity(
                e,
                intensity_override = field_slice.get("intensity", None),
            )
            
            slice_axis = str(plot_opts.get("slice_axis", "z")).lower()
            slice_value = float(plot_opts.get("slice_value", 0.0))
            _, _, int_u, _ = self._slice_points(
                pos, e, intensity, axis = slice_axis, value = slice_value
            )
            
            valid = int_u[np.isfinite(int_u)]
            if valid.size == 0:
                valid = np.asarray([0.0], dtype=float)
                
            p_low_pct = float(plot_opts.get("p_low", 2.0))
            p_high_pct = float(plot_opts.get("p_high", 99.5))
            
            actual_low_val = float(np.percentile(valid, p_low_pct))
            actual_high_val = float(np.percentile(valid, p_high_pct))

            payload = {
                "metadata": meta,
                "statistics": stats,
                "hotspots": hotspots,
                "plot_options": plot_opts,
                "actual_low_val": actual_low_val,
                "actual_high_val": actual_high_val,
            }
            self.processed_outputs["field"] = payload

            fig = self._build_field_figure(field_slice, meta, plot_opts)
            self._field_summary_label.setText(self._format_field_summary(payload))
            self._set_result_figure("field", fig)
        except Exception as exc:
            QMessageBox.critical(self, "Field Analysis Failed", str(exc))

    

    def _build_spectrum_figure(self, raw: dict, pol_idx: int = -1) -> Figure:
        wl = np.asarray(raw["wavelength"])
        ext = np.asarray(raw["ext"])
        sca = np.asarray(raw["sca"])
        abs_ = np.asarray(raw["abs"])

        fig = Figure()
        ax = fig.add_subplot(111)

        n_pol = ext.shape[1]
        show_all = (pol_idx < 0) or (pol_idx >= n_pol)

        # When showing all polarizations with many pols, only show averages to avoid legend explosion
        max_individual_pols_to_plot = 3
        show_individual = show_all and (n_pol <= max_individual_pols_to_plot)

        if show_individual:
            # Show individual polarizations for small n_pol
            for i in range(n_pol):
                ax.plot(wl, ext[:, i], label = "ext pol {}".format(i))
                ax.plot(wl, sca[:, i], linestyle = "--", label = "sca pol {}".format(i))
                ax.plot(wl, abs_[:, i], linestyle = ":", label = "abs pol {}".format(i))
        elif show_all and n_pol > max_individual_pols_to_plot:
            # For many polarizations, show only averages with a note
            ax.plot(
                wl,
                np.mean(ext, axis = 1),
                color = "black",
                linewidth = 2.2,
                label = "ext avg ({} pols)".format(n_pol),
            )
            ax.plot(
                wl,
                np.mean(sca, axis = 1),
                color = "black",
                linestyle = "--",
                linewidth = 2.0,
                label = "sca avg ({} pols)".format(n_pol),
            )
            ax.plot(
                wl,
                np.mean(abs_, axis = 1),
                color = "black",
                linestyle = ":",
                linewidth = 2.0,
                label = "abs avg ({} pols)".format(n_pol),
            )
        else:
            # Show single polarization
            i = int(pol_idx)
            ax.plot(wl, ext[:, i], label = "ext pol {}".format(i))
            ax.plot(wl, sca[:, i], linestyle = "--", label = "sca pol {}".format(i))
            ax.plot(wl, abs_[:, i], linestyle = ":", label = "abs pol {}".format(i))

        if show_all and n_pol > 1 and show_individual:
            ax.plot(
                wl,
                np.mean(ext, axis = 1),
                color = "black",
                linewidth = 2.2,
                label = "ext avg",
            )
            ax.plot(
                wl,
                np.mean(sca, axis = 1),
                color = "black",
                linestyle = "--",
                linewidth = 2.0,
                label = "sca avg",
            )
            ax.plot(
                wl,
                np.mean(abs_, axis = 1),
                color = "black",
                linestyle = ":",
                linewidth = 2.0,
                label = "abs avg",
            )

        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Cross Section")
        if show_all:
            ax.set_title("Spectrum Overview ({} polarizations)".format(n_pol) if not show_individual else "Spectrum Overview")
        else:
            ax.set_title("Spectrum Overview (Polarization {})".format(int(pol_idx)))
        ax.grid(True, alpha = 0.3)
        ax.legend(fontsize = 8, loc = 'best')
        fig.tight_layout()
        return fig

    def _selected_spectrum_pol(self) -> int:
        if self._spec_pol_selector is None:
            return -1
        val = self._spec_pol_selector.currentData()
        try:
            return int(val)
        except Exception:
            return -1

    def _refresh_spectrum_selector(self, raw: dict | None):
        if self._spec_pol_selector is None:
            return

        self._spec_pol_selector.blockSignals(True)
        self._spec_pol_selector.clear()
        self._spec_pol_selector.addItem("All Polarizations", userData = -1)

        n_pol = 0
        if raw is not None and self._has_spectrum_data(raw):
            ext = np.asarray(raw["ext"])
            if ext.ndim >= 2:
                n_pol = int(ext.shape[1])

        for i in range(n_pol):
            self._spec_pol_selector.addItem("Polarization {}".format(i), userData = i)

        self._spec_pol_selector.setEnabled(n_pol > 0)
        self._spec_pol_selector.setCurrentIndex(0)
        self._spec_pol_selector.blockSignals(False)

    def _build_field_figure(self, field_slice: dict, meta: dict, plot_opts: dict) -> Figure:
        pos = np.asarray(field_slice["pos"])
        e = np.asarray(field_slice["e"])
        intensity = self._compute_intensity(
            e,
            intensity_override = field_slice.get("intensity", None),
        )

        mode = str(plot_opts.get("view_mode", "auto")).lower()

        if mode == "3d":
            return self._build_field_figure_3d(pos, intensity, meta, plot_opts)

        if mode == "scatter":
            return self._build_field_figure_scatter(pos, e, intensity, meta, plot_opts)

        if mode == "2d":
            return self._build_field_figure_2d(pos, e, intensity, meta, plot_opts)

        if self._is_grid_3d(pos):
            return self._build_field_figure_3d(pos, intensity, meta, plot_opts)

        return self._build_field_figure_2d(pos, e, intensity, meta, plot_opts)

    def _build_field_figure_scatter(self,
            pos: np.ndarray,
            e: np.ndarray,
            intensity: np.ndarray,
            meta: dict,
            plot_opts: dict) -> Figure:
        """
        Raw scatter plot of field data (no interpolation).
        Useful for diagnosing raw data issues vs interpolation problems.
        """
        slice_axis = str(plot_opts.get("slice_axis", "z")).lower()
        slice_value = float(plot_opts.get("slice_value", 0.0))
        pos_u, e_u, int_u, ax_idx = self._slice_points(
            pos, e, intensity, axis = slice_axis, value = slice_value)

        other = [i for i in range(3) if i != ax_idx]
        names = ["x", "y", "z"]
        x = pos_u[:, other[0]]
        y = pos_u[:, other[1]]

        # Use all valid data (no masking)
        finite_mask = np.isfinite(int_u)
        if np.sum(finite_mask) < 1:
            finite_mask = np.ones_like(int_u, dtype = bool)

        x_plot = x[finite_mask]
        y_plot = y[finite_mask]
        int_plot = int_u[finite_mask]

        # Determine color scale
        cmap = str(plot_opts.get("cmap", "inferno"))
        clip_percentile = bool(plot_opts.get("clip_percentile", True))
        p_low = float(plot_opts.get("p_low", 2.0))
        p_high = float(plot_opts.get("p_high", 99.5))

        valid = int_plot[np.isfinite(int_plot)]
        if valid.size == 0:
            valid = np.asarray([0.0], dtype = float)

        if clip_percentile:
            vmin = float(np.percentile(valid, p_low))
            vmax = float(np.percentile(valid, p_high))
        else:
            vmin = float(np.min(valid))
            vmax = float(np.max(valid))

        if vmax <= vmin:
            vmax = vmin + 1e-12

        fig = Figure(figsize = (12, 6))
        ax_lin = fig.add_subplot(121)
        ax_log = fig.add_subplot(122)

        # Linear scatter
        sc_lin = ax_lin.scatter(
            x_plot, y_plot, c = int_plot,
            cmap = cmap, s = 20, vmin = vmin, vmax = vmax,
            alpha = 0.7, edgecolors = 'none'
        )
        ax_lin.set_xlabel("{} (nm)".format(names[other[0]]))
        ax_lin.set_ylabel("{} (nm)".format(names[other[1]]))
        ax_lin.set_title("Linear |E|^2 (Raw Points)")
        ax_lin.grid(True, alpha = 0.3)
        ax_lin.set_aspect('equal')
        cb_lin = fig.colorbar(sc_lin, ax = ax_lin)
        cb_lin.set_label("|E|^2")

        # Log scatter
        pos_valid = int_plot[int_plot > 0]
        if pos_valid.size > 0:
            use_manual_log_bounds = bool(plot_opts.get("manual_log_bounds", False))
            if use_manual_log_bounds:
                lvmin = max(float(plot_opts.get("log_vmin", 1e-12)), 1e-12)
                lvmax = float(plot_opts.get("log_vmax", lvmin * 10.0))
            else:
                if clip_percentile:
                    lvmin = max(float(np.percentile(pos_valid, p_low)), 1e-12)
                    lvmax = float(np.percentile(pos_valid, p_high))
                else:
                    lvmin = max(float(np.min(pos_valid)), 1e-12)
                    lvmax = float(np.max(pos_valid))
            if lvmax <= lvmin:
                lvmax = lvmin * 10.0

            # Mask very small values for log plot
            log_mask = int_plot > 1e-15
            sc_log = ax_log.scatter(
                x_plot[log_mask], y_plot[log_mask], c = int_plot[log_mask],
                cmap = cmap, s = 20, norm = LogNorm(vmin = lvmin, vmax = lvmax),
                alpha = 0.7, edgecolors = 'none'
            )
            cb_label = "|E|^2 (log)"
        else:
            sc_log = ax_log.scatter(
                x_plot, y_plot, c = int_plot,
                cmap = cmap, s = 20, vmin = vmin, vmax = vmax,
                alpha = 0.7, edgecolors = 'none'
            )
            cb_label = "|E|^2"

        ax_log.set_xlabel("{} (nm)".format(names[other[0]]))
        ax_log.set_ylabel("{} (nm)".format(names[other[1]]))
        ax_log.set_title("Log-aware |E|^2 (Raw Points)")
        ax_log.grid(True, alpha = 0.3)
        ax_log.set_aspect('equal')
        cb_log = fig.colorbar(sc_log, ax = ax_log)
        cb_log.set_label(cb_label)

        pol_label = self._format_pol_label(meta)
        fig.suptitle(
            "Raw Field Scatter (no interpolation) - wl={:.2f} nm, pol={}, {} points".format(
                float(meta.get("wavelength_nm", 0.0)),
                pol_label,
                len(x_plot)
            ),
            fontsize = 11,
        )

        fig.tight_layout()
        return fig

    def _set_log_bounds_controls_enabled(self,
            enabled: bool,
            log_vmin_ctrl: QDoubleSpinBox,
            log_vmax_ctrl: QDoubleSpinBox):
        log_vmin_ctrl.setEnabled(enabled)
        log_vmax_ctrl.setEnabled(enabled)

    def _compute_intensity(self,
            e: np.ndarray,
            intensity_override: np.ndarray | None = None) -> np.ndarray:
        if intensity_override is not None:
            return np.asarray(intensity_override, dtype = float)

        e2 = np.sum(np.abs(e) ** 2, axis = -1)
        while e2.ndim > 1:
            e2 = np.mean(e2, axis = -1)
        return np.asarray(e2, dtype = float)

    def _normalize_field_components(self,
            e_block: np.ndarray,
            pos_count: int,
            requested_pol_idx: int) -> tuple[np.ndarray, int, np.ndarray | None]:
        arr = np.asarray(e_block)
        average_all_pol = requested_pol_idx < 0

        if arr.ndim == 2:
            if arr.shape == (pos_count, 3):
                return arr, 0, None
            if arr.shape == (3, pos_count):
                return arr.T, 0, None
            if arr.shape[-1] == 3:
                return arr.reshape(-1, 3), 0, None
            if arr.shape[0] == 3:
                return arr.T.reshape(-1, 3), 0, None
            return arr.reshape(pos_count, 3), 0, None

        if arr.ndim != 3:
            return arr.reshape(pos_count, 3), 0, None

        pos_axes = [ax for ax, size in enumerate(arr.shape) if size == pos_count]
        comp_axes = [ax for ax, size in enumerate(arr.shape) if size == 3]

        pos_axis = pos_axes[0] if pos_axes else 0
        comp_axis = next((ax for ax in comp_axes if ax != pos_axis), None)
        if comp_axis is None:
            comp_axis = 1 if arr.shape[1] == 3 else arr.ndim - 1

        pol_axis = next((ax for ax in range(arr.ndim) if ax not in (pos_axis, comp_axis)), None)
        if pol_axis is None:
            normalized = np.moveaxis(arr, (pos_axis, comp_axis), (0, 1)).reshape(pos_count, 3)
            return normalized, 0, None

        normalized = np.moveaxis(arr, (pos_axis, comp_axis, pol_axis), (0, 1, 2))

        if normalized.shape[1] != 3 and normalized.shape[2] == 3:
            normalized = np.moveaxis(normalized, 2, 1)

        if normalized.shape[0] != pos_count or normalized.shape[1] != 3:
            normalized = normalized.reshape(pos_count, 3, -1)

        if average_all_pol:
            # Physical unpolarized average: <|E|^2> over polarization,
            # not |<E>|^2 which can cancel by phase/symmetry.
            intensity_per_pol = np.sum(np.abs(normalized) ** 2, axis = 1)
            intensity_avg = np.mean(intensity_per_pol, axis = 1)
            # Keep representative complex field for vector overlay.
            e_rep = np.mean(normalized, axis = 2)
            return e_rep, -1, np.asarray(intensity_avg, dtype = float)

        used_pol_idx = min(max(0, requested_pol_idx), normalized.shape[2] - 1)
        return normalized[:, :, used_pol_idx], used_pol_idx, None

    def _structured_slice_grid(self,
            x: np.ndarray,
            y: np.ndarray,
            values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        x = np.asarray(x, dtype = float)
        y = np.asarray(y, dtype = float)
        values = np.asarray(values, dtype = float)

        if x.size == 0 or y.size == 0 or values.size == 0:
            return None

        x_key = np.round(x, decimals = 9)
        y_key = np.round(y, decimals = 9)
        x_unique, x_inv = np.unique(x_key, return_inverse = True)
        y_unique, y_inv = np.unique(y_key, return_inverse = True)

        if x_unique.size * y_unique.size != values.size:
            return None

        grid = np.full((y_unique.size, x_unique.size), np.nan, dtype = float)
        counts = np.zeros_like(grid, dtype = int)

        for idx, val in enumerate(values):
            yi = y_inv[idx]
            xi = x_inv[idx]
            if np.isnan(grid[yi, xi]):
                grid[yi, xi] = val
            else:
                grid[yi, xi] += val
            counts[yi, xi] += 1

        if np.any(counts == 0):
            return None

        multi_mask = counts > 1
        if np.any(multi_mask):
            grid[multi_mask] /= counts[multi_mask]

        x_coords = np.array([np.mean(x[x_inv == i]) for i in range(x_unique.size)], dtype = float)
        y_coords = np.array([np.mean(y[y_inv == i]) for i in range(y_unique.size)], dtype = float)
        return x_coords, y_coords, grid

    def _slice_points(self,
            pos: np.ndarray,
            e: np.ndarray,
            intensity: np.ndarray,
            axis: str,
            value: float):
        """
        Extract a 2D slice of field data along a specified axis.
        Uses a fixed tolerance approach (more robust than dynamic calculations).
        """
        axis_map = {"x": 0, "y": 1, "z": 2}
        ax_idx = axis_map.get(axis, 2)

        coords = np.asarray(pos[:, ax_idx], dtype = float)
        if coords.size == 0:
            return pos, e, intensity, ax_idx

        coord_min = float(np.min(coords))
        coord_max = float(np.max(coords))
        coord_range = coord_max - coord_min

        # If the slice value is outside the grid bounds, use the center instead
        if value < coord_min or value > coord_max:
            value = coord_min + 0.5 * coord_range

        # Use fixed tolerance approach (from Gemini/BEM best practices)
        # This is more robust than dynamic grid spacing calculations
        diffs = np.abs(coords - value)
        
        # Start with a conservative tolerance (1e-3 nm is reasonable for most grids)
        base_tol = 1e-3
        
        # If grid spacing appears very small, adapt slightly
        n_unique = len(np.unique(np.round(coords, 9)))
        if n_unique > 1:
            grid_spacing = coord_range / (n_unique - 1)
            # Use max of base_tol and 1/10 of grid spacing
            tol = max(base_tol, 0.1 * grid_spacing)
        else:
            tol = base_tol
        
        mask = diffs <= tol
        
        # If we get too few points, gradually increase tolerance
        for tol_scale in [1.0, 2.0, 5.0, 10.0, 100.0]:
            if np.sum(mask) >= 4:
                break
            mask = diffs <= tol_scale * tol
        
        # Final fallback: get the nearest plane
        if np.sum(mask) == 0:
            nearest_idx = np.argmin(diffs)
            mask = coords == coords[nearest_idx]
        
        return pos[mask], e[mask], intensity[mask], ax_idx

    def _is_grid_3d(self, pos: np.ndarray, tol: float = 1e-9) -> bool:
        if pos.ndim != 2 or pos.shape[0] == 0 or pos.shape[1] < 3:
            return False

        variable_axes = 0
        for ax in range(3):
            if float(np.ptp(pos[:, ax])) > tol:
                variable_axes += 1

        return variable_axes >= 3

    def _build_field_figure_2d(self,
            pos: np.ndarray,
            e: np.ndarray,
            intensity: np.ndarray,
            meta: dict,
            plot_opts: dict) -> Figure:
        from scipy.interpolate import griddata

        slice_axis = str(plot_opts.get("slice_axis", "z")).lower()
        slice_value = float(plot_opts.get("slice_value", 0.0))
        pos_u, e_u, int_u, ax_idx = self._slice_points(
            pos, e, intensity, axis = slice_axis, value = slice_value)

        other = [i for i in range(3) if i != ax_idx]
        names = ["x", "y", "z"]
        x = pos_u[:, other[0]]
        y = pos_u[:, other[1]]

        # --- Data cleaning: only remove NaN/inf, keep all valid data ---
        finite_mask = np.isfinite(int_u)
        
        # If all data is NaN, use unfiltered (to avoid empty grid)
        if np.sum(finite_mask) < 4:
            valid_mask = np.ones_like(int_u, dtype = bool)
        else:
            valid_mask = finite_mask

        x_clean = x[valid_mask]
        y_clean = y[valid_mask]
        int_clean = int_u[valid_mask]

    # ------------------------------------------------------------------------------------------------
        clip_percentile = bool(plot_opts.get("clip_percentile", True))
        p_low = float(plot_opts.get("p_low", 2.0))
        p_high = float(plot_opts.get("p_high", 99.5))
        if p_high <= p_low:
            p_low, p_high = 2.0, 99.5

        cmap = str(plot_opts.get("cmap", "inferno"))
        log_scale = bool(plot_opts.get("log_scale", True))
        vector_overlay = bool(plot_opts.get("vector_overlay", False))
        vec_density = int(plot_opts.get("vector_density", 4))

        valid = int_clean[np.isfinite(int_clean)]
        if valid.size == 0:
            valid = np.asarray([0.0], dtype = float)

        if clip_percentile:
            vmin = float(np.percentile(valid, p_low))
            vmax = float(np.percentile(valid, p_high))
            if vmax <= vmin:
                vmax = vmin + 1e-12
        else:
            vmin = float(np.min(valid))
            vmax = float(np.max(valid))
            if vmax <= vmin:
                vmax = vmin + 1e-12

        fig = Figure(figsize = (13, 5.8))
        ax_lin = fig.add_subplot(121)
        ax_log = fig.add_subplot(122)

        mesh = self._structured_slice_grid(x_clean, y_clean, int_clean)
        image_grid = None
        use_scatter_fallback = x_clean.size < 3

        if mesh is not None:
            x_grid, y_grid, intensity_grid = mesh
            image_grid = {
                "grid": intensity_grid,
                "extent": [x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()],
            }
        elif not use_scatter_fallback:
            x_span = float(np.ptp(x_clean))
            y_span = float(np.ptp(y_clean))
            nx_unique = max(2, np.unique(np.round(x_clean, 9)).size)
            ny_unique = max(2, np.unique(np.round(y_clean, 9)).size)

            grid_res_x = min(320, max(120, nx_unique * 4))
            grid_res_y = min(320, max(120, ny_unique * 4))

            if x_span <= 0.0:
                grid_res_x = 2
            if y_span <= 0.0:
                grid_res_y = 2

            xi = np.linspace(float(np.min(x_clean)), float(np.max(x_clean)), grid_res_x)
            yi = np.linspace(float(np.min(y_clean)), float(np.max(y_clean)), grid_res_y)
            XI, YI = np.meshgrid(xi, yi)

            try:
                interp_grid = griddata(
                    (x_clean, y_clean),
                    int_clean,
                    (XI, YI),
                    method = 'cubic',
                )
                if np.any(np.isnan(interp_grid)):
                    fill_grid = griddata(
                        (x_clean, y_clean),
                        int_clean,
                        (XI, YI),
                        method = 'nearest',
                    )
                    interp_grid = np.where(np.isnan(interp_grid), fill_grid, interp_grid)

                if np.any(np.isfinite(interp_grid)):
                    image_grid = {
                        "grid": interp_grid,
                        "extent": [xi.min(), xi.max(), yi.min(), yi.max()],
                    }
                else:
                    use_scatter_fallback = True
            except Exception:
                use_scatter_fallback = True

        def _draw_intensity(ax, *, use_log_norm: bool):
            if use_scatter_fallback or image_grid is None:
                scatter_kwargs = {
                    "c": int_clean,
                    "cmap": cmap,
                    "s": 16,
                    "edgecolors": "none",
                }
                if use_log_norm:
                    scatter_kwargs["norm"] = norm
                else:
                    scatter_kwargs["vmin"] = vmin
                    scatter_kwargs["vmax"] = vmax
                return ax.scatter(x_clean, y_clean, **scatter_kwargs)

            image_kwargs = {
                "cmap": cmap,
                "origin": "lower",
                "extent": image_grid["extent"],
                "aspect": "equal",
                "interpolation": "bilinear",
            }
            if use_log_norm:
                image_kwargs["norm"] = norm
            else:
                image_kwargs["vmin"] = vmin
                image_kwargs["vmax"] = vmax
            return ax.imshow(image_grid["grid"], **image_kwargs)

        im_lin = _draw_intensity(ax_lin, use_log_norm = False)
        ax_lin.set_xlabel("{} (nm)".format(names[other[0]]))
        ax_lin.set_ylabel("{} (nm)".format(names[other[1]]))
        ax_lin.set_title("Linear |E|^2")
        ax_lin.grid(True, alpha = 0.3)
        ax_lin.set_aspect('equal')
        cb_lin = fig.colorbar(im_lin, ax = ax_lin)
        cb_lin.set_label("|E|^2")

        pos_valid = int_clean[int_clean > 0]
        if log_scale and pos_valid.size > 0:
            use_manual_log_bounds = bool(plot_opts.get("manual_log_bounds", False))
            if use_manual_log_bounds:
                lvmin = max(float(plot_opts.get("log_vmin", 1e-12)), 1e-12)
                lvmax = float(plot_opts.get("log_vmax", lvmin * 10.0))
            else:
                if clip_percentile:
                    lvmin = max(float(np.percentile(pos_valid, p_low)), 1e-12)
                    lvmax = float(np.percentile(pos_valid, p_high))
                else:
                    lvmin = max(float(np.min(pos_valid)), 1e-12)
                    lvmax = float(np.max(pos_valid))
            if lvmax <= lvmin:
                lvmax = lvmin * 10.0
            norm = LogNorm(vmin = lvmin, vmax = lvmax)

            im_log = _draw_intensity(ax_log, use_log_norm = True)
            cb_label = "|E|^2 (log)"
        else:
            im_log = _draw_intensity(ax_log, use_log_norm = False)
            cb_label = "|E|^2"

        ax_log.set_xlabel("{} (nm)".format(names[other[0]]))
        ax_log.set_ylabel("{} (nm)".format(names[other[1]]))
        ax_log.set_title("Log-aware |E|^2")
        ax_log.grid(True, alpha = 0.3)
        ax_log.set_aspect('equal')
        cb_log = fig.colorbar(im_log, ax = ax_log)
        cb_log.set_label(cb_label)

        # Vector Overlay (remains anchored to original sparse data coordinate arrays x and y)
        # Skip vector overlay for very large datasets to avoid slowness
        max_vectors_for_overlay = 2000
        if vector_overlay and e_u.ndim >= 2 and e_u.shape[1] >= 3 and x.shape[0] <= max_vectors_for_overlay:
            uu = np.real(e_u[:, other[0]])
            vv = np.real(e_u[:, other[1]])
            step = max(1, vec_density)
            idx = np.arange(0, x.shape[0], step, dtype = int)
            try:
                ax_lin.quiver(x[idx], y[idx], uu[idx], vv[idx],
                        color = 'cyan', alpha = 0.75, width = 0.003)
            except Exception:
                # Skip vector overlay silently if it fails
                pass
        elif vector_overlay and x.shape[0] > max_vectors_for_overlay:
            # Notify user that vector overlay is skipped for large datasets
            ax_lin.text(0.5, 0.95, "Vector overlay skipped (too many points)", 
                       transform=ax_lin.transAxes, ha='center', va='top',
                       fontsize=8, color='gray', style='italic')

        pol_label = self._format_pol_label(meta)
        fig.suptitle(
            "Field Intensity Map (slice {}={:.2f} nm, wl={:.2f} nm, pol={})".format(
                slice_axis,
                float(slice_value),
                float(meta.get("wavelength_nm", 0.0)),
                pol_label,
            ),
            fontsize = 11,
        )

        fig.tight_layout()
        return fig

    def _build_field_figure_3d(self,
            pos: np.ndarray,
            intensity: np.ndarray,
            meta: dict,
            plot_opts: dict) -> Figure:
        q = float(plot_opts.get("hotspot_quantile", 0.99))
        q = min(0.9999, max(0.5, q))
        threshold = float(np.quantile(intensity, q))
        mask = intensity >= threshold
        cmap = str(plot_opts.get("cmap", "inferno"))

        fig = Figure(figsize = (13, 6))
        ax = fig.add_subplot(121, projection = '3d')
        ax_hist = fig.add_subplot(122)

        # For large datasets, downsample background scatter for performance
        n_points = pos.shape[0]
        max_bg_points = 10000
        if n_points > max_bg_points:
            bg_idx = np.random.choice(n_points, size = max_bg_points, replace = False)
            pos_bg = pos[bg_idx]
        else:
            pos_bg = pos

        ax.scatter(
            pos_bg[:, 0], pos_bg[:, 1], pos_bg[:, 2],
            c = '#7f7f7f', s = 6, alpha = 0.08)

        if np.any(mask):
            scatter = ax.scatter(
                pos[mask, 0], pos[mask, 1], pos[mask, 2],
                c = intensity[mask], cmap = cmap, s = 24, alpha = 0.95)
            cbar = fig.colorbar(scatter, ax = ax, shrink = 0.75)
            cbar.set_label('|E|^2 (top {:.2f}%)'.format(100.0 * (1.0 - q)))

        ax.set_xlabel('x (nm)')
        ax.set_ylabel('y (nm)')
        ax.set_zlabel('z (nm)')
        ax.set_title(
            'Field Hotspots 3D |E|^2 (wl={:.2f} nm, pol={})'.format(
                float(meta.get('wavelength_nm', 0.0)), self._format_pol_label(meta)))

        finite = intensity[np.isfinite(intensity)]
        if finite.size > 0:
            ax_hist.hist(finite, bins = 60, color = '#ff7f0e', alpha = 0.85)
        ax_hist.set_title('Intensity Distribution')
        ax_hist.set_xlabel('|E|^2')
        ax_hist.set_ylabel('Count')
        ax_hist.grid(True, alpha = 0.3)

        fig.tight_layout()
        return fig

    # TODO: make this like the field summary rework
    def _format_spectrum_summary(self, summary: dict) -> str:
        lines = [
            "n_wavelengths: {}".format(summary.get("n_wavelengths", "?")),
            "n_pol: {}".format(summary.get("n_pol", "?")),
        ]

        per_pol = summary.get("per_pol", dict())
        for key in sorted(per_pol.keys(), key = lambda x: int(x)):
            pol_data = per_pol[key]
            lines.append(
                "pol {} peak: wl={:.3f} nm, ext={:.6g}, fwhm={}".format(
                    key,
                    float(pol_data.get("peak_wl_nm", float("nan"))),
                    float(pol_data.get("peak_ext", float("nan"))),
                    pol_data.get("fwhm_nm", "nan"),
                )
            )

        return "\n".join(lines)

    def _format_field_summary(self, payload: dict) -> str:
        stats = payload.get("statistics", dict())
        hot = payload.get("hotspots", dict())
        meta = payload.get("metadata", dict())
        opts = payload.get("plot_options", dict())
        
        # Safely pull the calculated actual values
        v_low_val = payload.get("actual_low_val", 0.0)
        v_high_val = payload.get("actual_high_val", 0.0)

        parts = [
            "<b>Wavelength:</b> {:.2f} nm".format(float(meta.get("wavelength_nm", 0.0))),
            "<b>Pol:</b> {}".format(self._format_pol_label(meta)),
            "<b>Slice:</b> {} = {:.2f} nm".format(opts.get("slice_axis", "z"), float(opts.get("slice_value", 0.0))),
            "<b>Clip Range:</b> [{:.4g} to {:.4g}]".format(v_low_val, v_high_val), # Shows the real values
            "<b>Max |E|²:</b> {:.4g}".format(float(stats.get("max", 0.0))),
            "<b>Mean |E|²:</b> {:.4g}".format(float(stats.get("mean", 0.0))),
            "<b>P99 |E|²:</b> {:.4g}".format(float(stats.get("percentile_99", 0.0))),
            "<b>Hotspots (q={:.4f}):</b> {}".format(float(opts.get("hotspot_quantile", 0.99)), int(hot.get("n_hotspots", 0))),
        ]
        return " &nbsp;&nbsp;|&nbsp;&nbsp; ".join(parts)

    def _format_pol_label(self, meta: dict) -> str:
        pol_idx = int(meta.get("polarization_idx", 0))
        if pol_idx < 0:
            return "avg(all)"
        return str(pol_idx)

    def _extract_field_slice(self, raw: dict, wl_idx: int = 0, pol_idx: int = 0):
        e = np.asarray(raw["e"])
        pos = np.asarray(raw["pos"])
        wl = np.asarray(raw.get("wavelength", []))

        wl_idx = max(0, int(wl_idx))
        requested_pol_idx = int(pol_idx)
        average_all_pol = requested_pol_idx < 0
        used_pol_idx = requested_pol_idx

        if e.ndim == 4:
            wl_idx = min(wl_idx, e.shape[0] - 1)
            e_wl = e[wl_idx]
        elif e.ndim == 3 and wl.size > 0 and e.shape[0] == wl.size:
            wl_idx = min(wl_idx, e.shape[0] - 1)
            e_wl = e[wl_idx]
        else:
            e_wl = e

        e_slice, used_pol_idx, intensity_override = self._normalize_field_components(
            e_wl,
            pos_count = pos.shape[0],
            requested_pol_idx = requested_pol_idx,
        )

        field_slice = {
            "e": e_slice,
            "pos": pos,
        }
        if intensity_override is not None:
            field_slice["intensity"] = intensity_override

        meta = {
            "wavelength_nm": float(wl[wl_idx]) if wl.size > wl_idx else 0.0,
            "wavelength_idx": wl_idx,
            "polarization_idx": used_pol_idx,
        }
        return field_slice, meta

    def _get_field_plot_options(self) -> dict:
        c = self._field_controls
        if not c:
            return {
                "view_mode": "auto",
                "slice_axis": "z",
                "slice_value": 0.0,
                "wavelength_idx": 0,
                "polarization_idx": 0,
                "log_scale": True,
                "clip_percentile": True,
                "p_low": 2.0,
                "p_high": 99.5,
                "cmap": "inferno",
                "vector_overlay": False,
                "vector_density": 4,
                "hotspot_quantile": 0.99,
                "manual_log_bounds": False,
                "log_vmin": 1e-6,
                "log_vmax": 1.0,
            }

        p_low = float(c["p_low"].value())
        p_high = float(c["p_high"].value())
        if p_high <= p_low:
            p_low, p_high = 2.0, 99.5

        return {
            "view_mode": str(c["view_mode"].currentText()).strip().lower(),
            "slice_axis": str(c["slice_axis"].currentText()).strip().lower(),
            "slice_value": float(c["slice_value"].value()),
            "wavelength_idx": int(c["wavelength_idx"].value()),
            "polarization_idx": int(c["pol_idx"].value()),
            "log_scale": bool(c["log_scale"].isChecked()),
            "clip_percentile": bool(c["clip_percentile"].isChecked()),
            "p_low": p_low,
            "p_high": p_high,
            "cmap": str(c["cmap"].currentText()),
            "vector_overlay": bool(c["vector_overlay"].isChecked()),
            "vector_density": int(c["vector_density"].value()),
            "hotspot_quantile": float(c["hotspot_quantile"].value()),
            "manual_log_bounds": bool(c["manual_log_bounds"].isChecked()),
            "log_vmin": float(c["log_vmin"].value()),
            "log_vmax": float(c["log_vmax"].value()),
        }

    def _save_processed_output(self, kind: str):
        if kind not in self.processed_outputs:
            QMessageBox.information(
                self,
                "No Processed Output",
                "Run {} analysis first, then save processed output.".format(kind),
            )
            return

        default_name = "{}_postprocess.json".format(kind)
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Processed Output",
            default_name,
            "JSON Files (*.json)",
        )

        if not path:
            return

        payload = {
            "kind": kind,
            "processed": self._to_jsonable(self.processed_outputs[kind]),
        }

        try:
            with open(path, "w", encoding = "utf-8") as f:
                json.dump(payload, f, indent = 2)
            QMessageBox.information(self, "Saved", "Processed output saved to:\n{}".format(path))
        except Exception as exc:
            QMessageBox.critical(self, "Save Failed", str(exc))

    def _save_raw_output(self, kind: str):
        raw = self.state.raw_results
        if raw is None:
            QMessageBox.warning(self, "No Raw Output", "No raw simulation output is available.")
            return

        out_dir = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if not out_dir:
            return

        try:
            if kind == "spectrum":
                if not self._has_spectrum_data(raw):
                    QMessageBox.warning(self, "Save Raw", "Current raw output has no spectrum arrays.")
                    return
                saved = save_spectrum(out_dir, raw)
            elif kind == "field":
                if not self._has_field_data(raw):
                    QMessageBox.warning(self, "Save Raw", "Current raw output has no field arrays.")
                    return
                saved = save_field(out_dir, raw)
            else:
                QMessageBox.warning(self, "Save Raw", "Unknown output kind <{}>.".format(kind))
                return

            msg = "Saved raw {} output:\n{}".format(kind, "\n".join(saved.values()))
            QMessageBox.information(self, "Saved", msg)
        except Exception as exc:
            QMessageBox.critical(self, "Save Raw Failed", str(exc))

    def _has_spectrum_data(self, raw: dict) -> bool:
        required = ["wavelength", "ext", "sca", "abs"]
        return all(k in raw for k in required)

    def _has_field_data(self, raw: dict) -> bool:
        required = ["wavelength", "pos", "e"]
        return all(k in raw for k in required)

    def _to_jsonable(self, obj):
        if isinstance(obj, dict):
            return {str(k): self._to_jsonable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._to_jsonable(v) for v in obj]
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return obj
