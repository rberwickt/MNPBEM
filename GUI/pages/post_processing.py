import json

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QScrollArea,
    QGridLayout,
    QLabel,
    QMessageBox,
    QFileDialog,
)
from PySide6.QtCore import Qt
from ..simulation_state import SimulationState
from ..widgets.calculation_item import CalculationItemWidget
from ..widgets.calculation_figure import CalculationFigure
from matplotlib.figure import Figure
import numpy as np
from pymnpbem_simulation.postprocess import analyze_spectrum, field_statistics, hotspot_summary
from pymnpbem_simulation.io import save_spectrum, save_field


class ProcessingPage(QWidget):
    CARD_MIN_WIDTH = 560

    def __init__(self, state: SimulationState):
        super().__init__()
        self.state = state
        self.items = []
        self.processed_outputs = dict()

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)

        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        self.content_widget = QWidget()
        self.grid_layout = QGridLayout(self.content_widget)
        self.grid_layout.setContentsMargins(12, 12, 12, 12)
        self.grid_layout.setHorizontalSpacing(12)
        self.grid_layout.setVerticalSpacing(12)
        self.grid_layout.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.scroll_area.setWidget(self.content_widget)
        root_layout.addWidget(self.scroll_area)

        self.setup_ui_from_state()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._reflow_grid()

    def _reflow_grid(self):
        viewport_width = max(1, self.scroll_area.viewport().width())

        left, _, right, _ = self.grid_layout.getContentsMargins()
        spacing = max(0, self.grid_layout.horizontalSpacing())
        usable_width = max(1, viewport_width - left - right)

        hint_widths = []
        for item in self.items:
            hint_widths.append(item.minimumSizeHint().width())
            hint_widths.append(item.sizeHint().width())
        effective_card_width = max([self.CARD_MIN_WIDTH] + hint_widths) + 8

        # Account for spacing in the fit calculation so we wrap before the
        # last card is partially clipped at the right edge.
        cols = max(1, (usable_width + spacing) // (effective_card_width + spacing))

        while self.grid_layout.count():
            self.grid_layout.takeAt(0)

        for idx, item in enumerate(self.items):
            row = idx // cols
            col = idx % cols
            self.grid_layout.addWidget(item, row, col)

        for col in range(cols):
            self.grid_layout.setColumnStretch(col, 1)
        last_row = max(0, (len(self.items) - 1) // cols)
        self.grid_layout.setRowStretch(last_row + 1, 1)
        self.content_widget.adjustSize()

    def setup_ui_from_state(self):
        """Called by main window right before switching to this page"""
        self._clear_cards()
        self.processed_outputs = dict()

        raw = self.state.raw_results

        if raw is None:
            self._add_info_card(
                "No Simulation Result",
                "Run a simulation first. Post-processing uses the raw simulation output generated after the run."
            )
            self._reflow_grid()
            return

        self._add_spectrum_card()
        self._add_field_card()

        self._reflow_grid()

    def _clear_cards(self):
        while self.grid_layout.count():
            item = self.grid_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)

        for card in self.items:
            card.deleteLater()
        self.items = []

    def _add_info_card(self, title: str, message: str):
        card = CalculationItemWidget(title)
        card.set_actions([])

        content = QWidget()
        layout = QVBoxLayout(content)
        label = QLabel(message)
        label.setWordWrap(True)
        layout.addWidget(label)

        card.set_content_widget(content)
        card.mark_as_processed()
        self.items.append(card)

    def _add_spectrum_card(self):
        card = CalculationItemWidget("Spectrum Post-Processing")
        card.set_actions([
            ("Run Analysis", lambda _checked = False, c = card: self._run_spectrum_analysis(c)),
            ("Save Processed", lambda _checked = False: self._save_processed_output("spectrum")),
            ("Save Raw", lambda _checked = False: self._save_raw_output("spectrum")),
        ])

        intro = QWidget()
        intro_layout = QVBoxLayout(intro)
        label = QLabel(
            "Runs from raw simulation output (not pre-generated post-process files). "
            "Computes peaks/FWHM and renders a live figure."
        )
        label.setWordWrap(True)
        intro_layout.addWidget(label)

        card.set_content_widget(intro)
        card.mark_as_processed()
        self.items.append(card)

    def _add_field_card(self):
        card = CalculationItemWidget("Field Post-Processing")
        card.set_actions([
            ("Run Analysis", lambda _checked = False, c = card: self._run_field_analysis(c)),
            ("Save Processed", lambda _checked = False: self._save_processed_output("field")),
            ("Save Raw", lambda _checked = False: self._save_raw_output("field")),
        ])

        intro = QWidget()
        intro_layout = QVBoxLayout(intro)
        label = QLabel(
            "Runs from raw simulation output. Field analysis needs field arrays (pos/e). "
            "If your run was spectrum-only, rerun with field calculation enabled to analyze fields."
        )
        label.setWordWrap(True)
        intro_layout.addWidget(label)

        card.set_content_widget(intro)
        card.mark_as_processed()
        self.items.append(card)

    def _run_spectrum_analysis(self, card: CalculationItemWidget):
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
            content = self._build_content_widget(self._format_spectrum_summary(summary), fig)
            card.set_content_widget(content)
            card.mark_as_processed()
            self._reflow_grid()
        except Exception as exc:
            QMessageBox.critical(self, "Spectrum Analysis Failed", str(exc))

    def _run_field_analysis(self, card: CalculationItemWidget):
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
            field_slice, meta = self._extract_field_slice(raw)
            stats = field_statistics(field_slice)
            hotspots = hotspot_summary(field_slice, threshold_quantile = 0.99)

            payload = {
                "metadata": meta,
                "statistics": stats,
                "hotspots": hotspots,
            }
            self.processed_outputs["field"] = payload

            fig = self._build_field_figure(field_slice, meta)
            content = self._build_content_widget(self._format_field_summary(payload), fig)
            card.set_content_widget(content)
            card.mark_as_processed()
            self._reflow_grid()
        except Exception as exc:
            QMessageBox.critical(self, "Field Analysis Failed", str(exc))

    def _build_content_widget(self, summary_text: str, figure: Figure):
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        label = QLabel(summary_text)
        label.setWordWrap(True)

        fig_widget = CalculationFigure(figure)
        layout.addWidget(label)
        layout.addWidget(fig_widget, 1)
        return content

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

    def _build_field_figure(self, field_slice: dict, meta: dict) -> Figure:
        pos = np.asarray(field_slice["pos"])
        e = np.asarray(field_slice["e"])
        intensity = np.sum(np.abs(e) ** 2, axis = 1)

        if self._is_grid_3d(pos):
            return self._build_field_figure_3d(pos, intensity, meta)

        return self._build_field_figure_2d(pos, intensity, meta)

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
            intensity: np.ndarray,
            meta: dict) -> Figure:

        spans = np.ptp(pos[:, :3], axis = 0)
        axes = np.argsort(spans)[::-1][:2]
        ax0 = int(axes[0])
        ax1 = int(axes[1])
        names = ["x", "y", "z"]

        fig = Figure()
        ax = fig.add_subplot(111)
        scatter = ax.scatter(pos[:, ax0], pos[:, ax1], c = intensity, cmap = "inferno", s = 20)

        ax.set_xlabel("{} (nm)".format(names[ax0]))
        ax.set_ylabel("{} (nm)".format(names[ax1]))
        ax.set_title(
            "Field Intensity |E|^2 (wl={:.2f} nm, pol={})".format(
                float(meta.get("wavelength_nm", 0.0)), int(meta.get("polarization_idx", 0))
            )
        )
        ax.grid(True, alpha = 0.3)

        cbar = fig.colorbar(scatter, ax = ax)
        cbar.set_label("|E|^2")

        fig.tight_layout()
        return fig

    def _build_field_figure_3d(self,
            pos: np.ndarray,
            intensity: np.ndarray,
            meta: dict) -> Figure:
        # Match the existing postprocess hotspot plot behavior: show top 1%
        # of |E|^2 points as the main 3D signal, with a faint full cloud for
        # geometric context.
        threshold = float(np.quantile(intensity, 0.99))
        mask = intensity >= threshold

        fig = Figure()
        ax = fig.add_subplot(111, projection = '3d')

        ax.scatter(
            pos[:, 0], pos[:, 1], pos[:, 2],
            c = '#7f7f7f', s = 6, alpha = 0.08)

        if np.any(mask):
            scatter = ax.scatter(
                pos[mask, 0], pos[mask, 1], pos[mask, 2],
                c = intensity[mask], cmap = 'inferno', s = 24, alpha = 0.95)
            cbar = fig.colorbar(scatter, ax = ax, shrink = 0.75)
            cbar.set_label('|E|^2 (top 1%)')

        ax.set_xlabel('x (nm)')
        ax.set_ylabel('y (nm)')
        ax.set_zlabel('z (nm)')
        ax.set_title(
            'Field Hotspots 3D |E|^2 (wl={:.2f} nm, pol={})'.format(
                float(meta.get('wavelength_nm', 0.0)), int(meta.get('polarization_idx', 0))))

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

        lines = [
            "wavelength_nm: {:.3f}".format(float(meta.get("wavelength_nm", 0.0))),
            "polarization_idx: {}".format(int(meta.get("polarization_idx", 0))),
            "max |E|^2: {:.6g}".format(float(stats.get("max", 0.0))),
            "mean |E|^2: {:.6g}".format(float(stats.get("mean", 0.0))),
            "p99 |E|^2: {:.6g}".format(float(stats.get("percentile_99", 0.0))),
            "hotspots (q=0.99): {}".format(int(hot.get("n_hotspots", 0))),
            "hotspot max intensity: {:.6g}".format(float(hot.get("max_intensity", 0.0))),
        ]
        return "\n".join(lines)

    def _extract_field_slice(self, raw: dict):
        e = np.asarray(raw["e"])
        pos = np.asarray(raw["pos"])
        wl = np.asarray(raw.get("wavelength", []))

        wl_idx = 0
        pol_idx = 0

        if e.ndim == 4:
            e_slice = e[wl_idx, :, :, pol_idx]
        elif e.ndim == 3:
            if wl.size > 0 and e.shape[0] == wl.size:
                e_wl = e[wl_idx]
            else:
                e_wl = e

            if e_wl.ndim == 3 and e_wl.shape[1] == 3:
                e_slice = e_wl[:, :, min(pol_idx, e_wl.shape[2] - 1)]
            elif e_wl.ndim == 3 and e_wl.shape[2] == 3:
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