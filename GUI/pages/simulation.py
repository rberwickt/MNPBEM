from PySide6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel, QHBoxLayout, QMessageBox
from PySide6.QtCore import Signal
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

    def __init__(self, state: SimulationState):
        super().__init__()
        self.state = state  # Access to the shared state
        
        self.columns = QHBoxLayout(self)
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

        self.refractive_index = RefractiveIndexWidget(state)
        self.col_2.addWidget(self.refractive_index)

        self.col_2.addStretch()

        self.col_3 = QVBoxLayout()

        self.excitation_settings = ExcitationSettingsWidget(state)
        self.col_3.addWidget(self.excitation_settings)

        self.run_btn = QPushButton("Run Simulation", self)
        self.run_btn.clicked.connect(self.on_run_simulation_clicked)
        self.col_3.addWidget(self.run_btn)

        self.col_3.addStretch()

        self.columns.addLayout(self.col_1)
        self.columns.addLayout(self.col_2)
        self.columns.addLayout(self.col_3)

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
