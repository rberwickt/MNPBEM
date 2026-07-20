from PySide6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QMessageBox, QScrollArea, QBoxLayout, QSizePolicy
from PySide6.QtCore import Signal, Qt, QTimer
from ..simulation_state import SimulationState
from ..widgets.solver_options import SolverOptionsWidget
from ..widgets.excitation_settings import ExcitationSettingsWidget
from ..widgets.material_settings import MaterialOptionsWidget
from ..widgets.structure_settings import StructureSettingsWidget
from ..widgets.energy_range import EnergyRangeWidget
from ..widgets.field_grid import FieldGridWidget
from ..widgets.simulation_dialog import SimulationProgressDialog
from ..widgets.refractive_display import RefractiveIndexWidget
from pathlib import Path


class SimulationPage(QWidget):
    sim_completed = Signal()  # Alert main.py when simulation finishes
    RESPONSIVE_STACK_WIDTH = 1500

    def __init__(self, state: SimulationState):
        super().__init__()
        self.state = state  # Access to the shared state

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        self.content_widget = QWidget(self.scroll_area)
        self.content_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.columns = QBoxLayout(QBoxLayout.LeftToRight)
        self.columns.setContentsMargins(8, 8, 8, 8)
        self.columns.setSpacing(12)
        self.content_widget.setLayout(self.columns)
        self.scroll_area.setWidget(self.content_widget)
        root_layout.addWidget(self.scroll_area)
        
        self.col_1 = QVBoxLayout()
        
        self.solver_options = SolverOptionsWidget(state)
        self.col_1.addWidget(self.solver_options)
        

        self.energy_range = EnergyRangeWidget(state)
        self.col_1.addWidget(self.energy_range)

        self.field_grid = FieldGridWidget(state)
        self.col_1.addWidget(self.field_grid)
        
        self.col_1.addStretch()

        self.col_2 = QVBoxLayout()
        
        self.material_settings = MaterialOptionsWidget(state)
        self.col_2.addWidget(self.material_settings)

        self.structure_settings = StructureSettingsWidget(state)
        self.col_2.addWidget(self.structure_settings)

        self.col_2.addStretch()

        self.col_3 = QVBoxLayout()

        self.excitation_settings = ExcitationSettingsWidget(state)
        self.col_3.addWidget(self.excitation_settings)

        self.refractive_index = RefractiveIndexWidget(state)
        self.col_3.addWidget(self.refractive_index)

        self.run_btn = QPushButton("Run Simulation", self)
        self.run_btn.clicked.connect(self.on_run_simulation_clicked)
        self.col_3.addWidget(self.run_btn)

        self.col_3.addStretch()

        self.columns.addLayout(self.col_1)
        self.columns.addLayout(self.col_2)
        self.columns.addLayout(self.col_3)

        self._set_column_widget_policies()
        self._update_responsive_layout()
        QTimer.singleShot(0, self._update_responsive_layout)

    def _set_column_widget_policies(self):
        """Allow child sections to shrink and grow with the available width."""
        self.setMinimumWidth(0)
        self.content_widget.setMinimumWidth(0)

        self.solver_options.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.solver_options.setMinimumWidth(0)
        self.energy_range.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.energy_range.setMinimumWidth(0)
        self.field_grid.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.field_grid.setMinimumWidth(0)

        self.material_settings.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.material_settings.setMinimumWidth(0)
        self.structure_settings.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.structure_settings.setMinimumWidth(0)
        self.refractive_index.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.refractive_index.setMinimumWidth(0)

        self.excitation_settings.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.excitation_settings.setMinimumWidth(0)
        self.run_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def _update_responsive_layout(self):
        available_width = self.scroll_area.viewport().width()
        if available_width < self.RESPONSIVE_STACK_WIDTH:
            self.columns.setDirection(QBoxLayout.TopToBottom)
        else:
            self.columns.setDirection(QBoxLayout.LeftToRight)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_responsive_layout()

    def showEvent(self, event):
        super().showEvent(event)
        # Ensure the first layout mode uses real viewport geometry.
        QTimer.singleShot(0, self._update_responsive_layout)

    def setup_ui_from_state(self):
        """Called by main window right before switching to this page"""
        pass

    def on_run_simulation_clicked(self):
        """Handle Run Simulation button click with validation"""
        is_valid, error_msg = self.state.validate_state()
        
        if not is_valid:
            QMessageBox.warning(
                self,
                "Simulation Configuration Error",
                f"Cannot run simulation:\n\n{error_msg}",
                QMessageBox.Ok
            )
            return
        
        output_dir = str(Path("..") / "tmp")
        
        progress_dialog = SimulationProgressDialog(
            state=self.state,
            output_dir=output_dir,
            parent=self
        )
        
        progress_dialog.simulation_success.connect(self.on_simulation_success)
        progress_dialog.simulation_error.connect(self.on_simulation_error)
        
        # Run with the explicit Start-page environment choice.
        n_threads = max(1, int(getattr(self.state, "env_n_threads", 1)))

        # run the simulation (non-blocking via threading)
        progress_dialog.run(n_threads=n_threads, save_outputs=False)
        
        # show the progress dialog (this part is blocking)
        progress_dialog.exec()

    def on_simulation_success(self, result: dict):
        """Handle successful simulation completion"""
        # store result in state for post-processing page access
        self.state.raw_results = result
        
        # success message
        QMessageBox.information(
            self,
            "Simulation Complete",
            "Simulation completed successfully!\n"
            "Results are ready for post-processing.",
            QMessageBox.Ok
        )
        
        self.sim_completed.emit()

    def on_simulation_error(self, exception: Exception):
        """Handle simulation error"""
        error_details = str(exception)
        
        QMessageBox.critical(
            self,
            "Simulation Failed",
            f"Simulation encountered an error:\n\n{error_details}",
            QMessageBox.Ok
        )
        
        # also log to console for debugging
        print(f"Simulation error: {exception}")
