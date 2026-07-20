from PySide6.QtWidgets import QDialog, QVBoxLayout, QLabel, QProgressBar, QTextEdit, QPushButton, QHBoxLayout
from PySide6.QtCore import Qt, Signal, QObject, Slot
from PySide6.QtGui import QFont
from ..simulation_state import SimulationState
from pathlib import Path
import shutil


class SimulationWorkerSignals(QObject):
    progress = Signal(str)  # Progress message
    success = Signal(dict)  # Result dict
    error = Signal(Exception)  # Exception


class SimulationProgressDialog(QDialog):
    """Modal dialog that runs a simulation and displays real-time progress.
    
    Signals:
        simulation_success: Emitted when simulation completes successfully. Passes result dict.
        simulation_error: Emitted when simulation fails. Passes Exception.
    """
    
    simulation_success = Signal(dict)  # result dict
    simulation_error = Signal(Exception)  # exception

    def __init__(self, state: SimulationState, output_dir: str = None, parent=None):
        super().__init__(parent)
        self.state = state
        self.output_dir = output_dir or str(Path(".") / "tmp" / "simulation_run")
        self.output_name = "simulation"
        self.simulation_thread = None
        self.is_running = False
        self.cancel_requested = False
        
        # Create signals for thread-safe communication
        self.signals = SimulationWorkerSignals()
        self.signals.progress.connect(self.append_output)
        self.signals.success.connect(self.on_simulation_success)
        self.signals.error.connect(self.on_simulation_error)
        
        self.setWindowTitle("Running Simulation")
        self.setModal(True)
        self.setMinimumWidth(600)
        self.setMinimumHeight(400)
        # Prevent closing while simulation is running
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowCloseButtonHint)
        
        self.init_ui()

    def init_ui(self):
        """Initialize UI components."""
        layout = QVBoxLayout(self)
        
        title = QLabel("Simulation Progress")
        title_font = QFont()
        title_font.setPointSize(12)
        title_font.setBold(True)
        title.setFont(title_font)
        layout.addWidget(title)
        
        self.status_label = QLabel("Initializing simulation...")
        layout.addWidget(self.status_label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(0)  # Indeterminate (bouncing) mode
        layout.addWidget(self.progress_bar)
        
        # Output text area
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        self.output_text.setFont(QFont("Courier", 9))
        layout.addWidget(self.output_text)
        
        # Buttons layout
        button_layout = QHBoxLayout()
        
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.on_cancel)
        button_layout.addWidget(self.cancel_btn)
        
        button_layout.addStretch()
        
        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.accept)
        self.close_btn.setEnabled(False)
        button_layout.addWidget(self.close_btn)
        
        layout.addLayout(button_layout)

    def run(self, n_threads: int = 1, save_outputs: bool = False):
        """Start the simulation.
        
        Args:
            n_threads: Number of threads for the simulation.
            save_outputs: Whether to save output files.
        """
        if self.is_running:
            return
        
        self.is_running = True
        self.cancel_requested = False
        self.cancel_btn.setEnabled(True)
        self.close_btn.setEnabled(False)
        self.output_text.clear()
        
        # Add initial message
        self.append_output(f"Output directory: {self.output_dir}")
        self.append_output(f"Using {n_threads} thread(s)\n")
        
        # Start simulation in background thread using the callbacks
        self.simulation_thread = self.state.run_simulation_threaded(
            on_success=self.signals.success.emit,
            on_error=self.signals.error.emit,
            on_progress=self.signals.progress.emit,
            output_dir=self.output_dir,
            output_name=self.output_name,
            save_outputs=save_outputs,
            n_threads=n_threads
        )

    def _run_output_path(self) -> Path:
        return Path(self.output_dir) / self.output_name

    def _clear_run_artifacts(self) -> None:
        """Best-effort cleanup for canceled/aborted runs in GUI tmp output."""
        output_path = self._run_output_path()

        # Clear simulation result payload stored in shared GUI state.
        self.state.raw_results = None

        # Remove sigma cache folder first (if available), then clean output dir.
        try:
            from pymnpbem_simulation import sigma_cache as _sc
            sigma_root = Path(_sc.sigma_dir(str(output_path)))
            if sigma_root.exists():
                shutil.rmtree(sigma_root, ignore_errors=True)
        except Exception:
            pass

        try:
            if output_path.exists():
                shutil.rmtree(output_path, ignore_errors=True)
        except Exception:
            pass

    @Slot(str)
    def append_output(self, message: str):
        """Append message to output text area (thread-safe)."""
        self.output_text.append(message)
        # Auto-scroll to bottom
        self.output_text.verticalScrollBar().setValue(
            self.output_text.verticalScrollBar().maximum()
        )
        self.status_label.setText(message)

    @Slot(dict)
    def on_simulation_success(self, result: dict):
        """Handle successful simulation completion."""
        if self.cancel_requested:
            # User canceled while worker was still running; drop late result.
            self._clear_run_artifacts()
            return

        self.is_running = False
        self.simulation_thread = None
        self.cancel_btn.setEnabled(False)
        self.close_btn.setEnabled(True)
        
        # Change progress bar to show completion
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(100)
        
        self.append_output("\nSimulation completed successfully!")
        
        # Emit success signal
        self.simulation_success.emit(result)

    @Slot(Exception)
    def on_simulation_error(self, exception: Exception):
        """Handle simulation error."""
        if self.cancel_requested:
            # User canceled while worker was still running; ignore late error.
            self._clear_run_artifacts()
            return

        self.is_running = False
        self.simulation_thread = None
        self.cancel_btn.setEnabled(False)
        self.close_btn.setEnabled(True)
        
        # Change progress bar to show error
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setStyleSheet("QProgressBar::chunk { background-color: red; }")
        
        error_msg = f"\nSimulation failed:\n{str(exception)}"
        self.append_output(error_msg)
        
        # Emit error signal
        self.simulation_error.emit(exception)

    def on_cancel(self):
        """Handle cancel button click."""
        if self.is_running:
            self.cancel_requested = True
            self.is_running = False
            killed = self.state.cancel_simulation_thread(self.simulation_thread)
            if killed:
                self.append_output("\nCancel requested. Worker thread stop signal sent.")
            else:
                self.append_output("\nCancel requested. Could not confirm worker thread stop signal.")

            self.append_output("Clearing current run results/cache and returning to simulation setup.")
            self._clear_run_artifacts()
            self.simulation_thread = None
            self.cancel_btn.setEnabled(False)
            self.close_btn.setEnabled(True)
            self.reject()
        else:
            self.reject()

    def closeEvent(self, event):
        """Handle dialog close event."""
        if self.is_running and (not self.cancel_requested):
            # Don't allow closing while simulation is running
            event.ignore()
        else:
            event.accept()
