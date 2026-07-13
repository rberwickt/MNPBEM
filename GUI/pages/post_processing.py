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

        self._btn_spec_run.clicked.connect(self._run_spectrum_analysis)
        self._btn_spec_save_processed.clicked.connect(
            lambda _checked = False: self._save_processed_output("spectrum"))
        self._btn_spec_save_raw.clicked.connect(
            lambda _checked = False: self._save_raw_output("spectrum"))

        action_row.addWidget(self._btn_spec_run)
        action_row.addWidget(self._btn_spec_save_processed)
        action_row.addWidget(self._btn_spec_save_raw)
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
        view_mode.addItems(["Auto", "2D", "3D"])

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
        pol_idx.setRange(0, 1000000)
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

        for col in range(8):
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

            fig = self._build_spectrum_figure(raw)
            self._spectrum_summary_label.setText(self._format_spectrum_summary(summary))
            self._set_result_figure("spectrum", fig)
        except Exception as exc:
            QMessageBox.critical(self, "Spectrum Analysis Failed", str(exc))

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

            payload = {
                "metadata": meta,
                "statistics": stats,
                "hotspots": hotspots,
                "plot_options": plot_opts,
            }
            self.processed_outputs["field"] = payload

            fig = self._build_field_figure(field_slice, meta, plot_opts)
            self._field_summary_label.setText(self._format_field_summary(payload))
            self._set_result_figure("field", fig)
        except Exception as exc:
            QMessageBox.critical(self, "Field Analysis Failed", str(exc))

    def _build_spectrum_figure(self, raw: dict) -> Figure:
        wl = np.asarray(raw["wavelength"])
        ext = np.asarray(raw["ext"])
        sca = np.asarray(raw["sca"])
        abs_ = np.asarray(raw["abs"])

        fig = Figure()
        ax = fig.add_subplot(111)

        n_pol = ext.shape[1]
        for i in range(n_pol):
            ax.plot(wl, ext[:, i], label = "ext pol {}".format(i))
            ax.plot(wl, sca[:, i], linestyle = "--", label = "sca pol {}".format(i))
            ax.plot(wl, abs_[:, i], linestyle = ":", label = "abs pol {}".format(i))

        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Cross Section")
        ax.set_title("Spectrum Overview")
        ax.grid(True, alpha = 0.3)
        ax.legend(fontsize = 8)
        fig.tight_layout()
        return fig

    def _build_field_figure(self, field_slice: dict, meta: dict, plot_opts: dict) -> Figure:
        pos = np.asarray(field_slice["pos"])
        e = np.asarray(field_slice["e"])
        intensity = self._compute_intensity(e)

        mode = str(plot_opts.get("view_mode", "auto")).lower()

        if mode == "3d":
            return self._build_field_figure_3d(pos, intensity, meta, plot_opts)

        if mode == "2d":
            return self._build_field_figure_2d(pos, e, intensity, meta, plot_opts)

        if self._is_grid_3d(pos):
            return self._build_field_figure_3d(pos, intensity, meta, plot_opts)

        return self._build_field_figure_2d(pos, e, intensity, meta, plot_opts)

    def _compute_intensity(self, e: np.ndarray) -> np.ndarray:
        e2 = np.sum(np.abs(e) ** 2, axis = -1)
        while e2.ndim > 1:
            e2 = np.mean(e2, axis = -1)
        return np.asarray(e2, dtype = float)

    def _slice_points(self,
            pos: np.ndarray,
            e: np.ndarray,
            intensity: np.ndarray,
            axis: str,
            value: float):
        axis_map = {"x": 0, "y": 1, "z": 2}
        ax_idx = axis_map.get(axis, 2)

        coords = np.asarray(pos[:, ax_idx], dtype = float)
        if coords.size == 0:
            return pos, e, intensity, ax_idx

        if np.allclose(coords, value):
            mask = np.ones(coords.shape[0], dtype = bool)
        else:
            diffs = np.abs(coords - value)
            nearest = float(np.min(diffs))
            denom = max(1, np.unique(np.round(coords, 9)).size)
            tol = max(1e-9, 0.5 * float(coords.max() - coords.min()) / denom)
            mask = diffs <= max(tol, nearest + 1e-12)

        if np.sum(mask) == 0:
            mask[np.argmin(np.abs(coords - value))] = True

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

        slice_axis = str(plot_opts.get("slice_axis", "z")).lower()
        slice_value = float(plot_opts.get("slice_value", 0.0))
        pos_u, e_u, int_u, ax_idx = self._slice_points(
            pos, e, intensity, axis = slice_axis, value = slice_value)

        other = [i for i in range(3) if i != ax_idx]
        names = ["x", "y", "z"]
        x = pos_u[:, other[0]]
        y = pos_u[:, other[1]]

        clip_percentile = bool(plot_opts.get("clip_percentile", True))
        p_low = float(plot_opts.get("p_low", 2.0))
        p_high = float(plot_opts.get("p_high", 99.5))
        if p_high <= p_low:
            p_low, p_high = 2.0, 99.5

        cmap = str(plot_opts.get("cmap", "inferno"))
        log_scale = bool(plot_opts.get("log_scale", True))
        vector_overlay = bool(plot_opts.get("vector_overlay", False))
        vec_density = int(plot_opts.get("vector_density", 4))

        valid = int_u[np.isfinite(int_u)]
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

        sc_lin = ax_lin.scatter(x, y, c = int_u, cmap = cmap, s = 20,
                vmin = vmin, vmax = vmax)
        ax_lin.set_xlabel("{} (nm)".format(names[other[0]]))
        ax_lin.set_ylabel("{} (nm)".format(names[other[1]]))
        ax_lin.set_title("Linear |E|^2")
        ax_lin.grid(True, alpha = 0.3)
        cb_lin = fig.colorbar(sc_lin, ax = ax_lin)
        cb_lin.set_label("|E|^2")

        pos_valid = int_u[int_u > 0]
        if log_scale and pos_valid.size > 0:
            if clip_percentile:
                lvmin = max(float(np.percentile(pos_valid, p_low)), 1e-12)
                lvmax = float(np.percentile(pos_valid, p_high))
            else:
                lvmin = max(float(np.min(pos_valid)), 1e-12)
                lvmax = float(np.max(pos_valid))
            if lvmax <= lvmin:
                lvmax = lvmin * 10.0
            norm = LogNorm(vmin = lvmin, vmax = lvmax)
            sc_log = ax_log.scatter(x, y, c = int_u, cmap = cmap, s = 20,
                    norm = norm)
            cb_label = "|E|^2 (log)"
        else:
            sc_log = ax_log.scatter(x, y, c = int_u, cmap = cmap, s = 20,
                    vmin = vmin, vmax = vmax)
            cb_label = "|E|^2"

        ax_log.set_xlabel("{} (nm)".format(names[other[0]]))
        ax_log.set_ylabel("{} (nm)".format(names[other[1]]))
        ax_log.set_title("Log-aware |E|^2")
        ax_log.grid(True, alpha = 0.3)
        cb_log = fig.colorbar(sc_log, ax = ax_log)
        cb_log.set_label(cb_label)

        if vector_overlay and e_u.ndim >= 2 and e_u.shape[1] >= 3:
            uu = np.real(e_u[:, other[0]])
            vv = np.real(e_u[:, other[1]])
            step = max(1, vec_density)
            idx = np.arange(0, x.shape[0], step, dtype = int)
            ax_lin.quiver(x[idx], y[idx], uu[idx], vv[idx],
                    color = 'cyan', alpha = 0.75, width = 0.003)

        fig.suptitle(
            "Field Intensity Map (slice {}={:.2f} nm, wl={:.2f} nm, pol={})".format(
                slice_axis,
                float(slice_value),
                float(meta.get("wavelength_nm", 0.0)),
                int(meta.get("polarization_idx", 0)),
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

        ax.scatter(
            pos[:, 0], pos[:, 1], pos[:, 2],
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
                float(meta.get('wavelength_nm', 0.0)), int(meta.get('polarization_idx', 0))))

        finite = intensity[np.isfinite(intensity)]
        if finite.size > 0:
            ax_hist.hist(finite, bins = 60, color = '#ff7f0e', alpha = 0.85)
        ax_hist.set_title('Intensity Distribution')
        ax_hist.set_xlabel('|E|^2')
        ax_hist.set_ylabel('Count')
        ax_hist.grid(True, alpha = 0.3)

        fig.tight_layout()
        return fig

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

        lines = [
            "wavelength_nm: {:.3f}".format(float(meta.get("wavelength_nm", 0.0))),
            "polarization_idx: {}".format(int(meta.get("polarization_idx", 0))),
            "view_mode: {}".format(opts.get("view_mode", "auto")),
            "slice: {} = {:.3f} nm".format(
                opts.get("slice_axis", "z"), float(opts.get("slice_value", 0.0))),
            "max |E|^2: {:.6g}".format(float(stats.get("max", 0.0))),
            "mean |E|^2: {:.6g}".format(float(stats.get("mean", 0.0))),
            "p99 |E|^2: {:.6g}".format(float(stats.get("percentile_99", 0.0))),
            "hotspots (q={:.4f}): {}".format(
                float(opts.get("hotspot_quantile", 0.99)), int(hot.get("n_hotspots", 0))),
            "hotspot max intensity: {:.6g}".format(float(hot.get("max_intensity", 0.0))),
        ]
        return "\n".join(lines)

    def _extract_field_slice(self, raw: dict, wl_idx: int = 0, pol_idx: int = 0):
        e = np.asarray(raw["e"])
        pos = np.asarray(raw["pos"])
        wl = np.asarray(raw.get("wavelength", []))

        wl_idx = max(0, int(wl_idx))
        pol_idx = max(0, int(pol_idx))

        if e.ndim == 4:
            wl_idx = min(wl_idx, e.shape[0] - 1)
            pol_idx = min(pol_idx, e.shape[3] - 1)
            e_slice = e[wl_idx, :, :, pol_idx]
        elif e.ndim == 3:
            if wl.size > 0 and e.shape[0] == wl.size:
                wl_idx = min(wl_idx, e.shape[0] - 1)
                e_wl = e[wl_idx]
            else:
                e_wl = e

            if e_wl.ndim == 3 and e_wl.shape[1] == 3:
                pol_idx = min(pol_idx, e_wl.shape[2] - 1)
                e_slice = e_wl[:, :, min(pol_idx, e_wl.shape[2] - 1)]
            elif e_wl.ndim == 3 and e_wl.shape[2] == 3:
                pol_idx = min(pol_idx, e_wl.shape[1] - 1)
                e_slice = e_wl[:, min(pol_idx, e_wl.shape[1] - 1), :]
            elif e_wl.ndim == 2:
                e_slice = e_wl
            else:
                e_slice = e_wl.reshape(-1, 3)
        elif e.ndim == 2:
            e_slice = e
        else:
            e_slice = e.reshape(-1, 3)

        if e_slice.shape[1] != 3:
            e_slice = e_slice.reshape(pos.shape[0], 3)

        field_slice = {
            "e": e_slice,
            "pos": pos,
        }

        meta = {
            "wavelength_nm": float(wl[wl_idx]) if wl.size > wl_idx else 0.0,
            "wavelength_idx": wl_idx,
            "polarization_idx": pol_idx,
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
