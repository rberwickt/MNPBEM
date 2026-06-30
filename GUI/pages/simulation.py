from PySide6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel
from PySide6.QtCore import Signal
from ..simulation_state import SimulationState
from ..widgets.excitation_settings import ExcitationSettingsWidget
class SimulationPage(QWidget):
    sim_completed = Signal()  # Alert main.py when simulation finishes

    def __init__(self, state: SimulationState):
        super().__init__()
        self.state = state  # Access to the shared state
        
        self.layout = QVBoxLayout(self)
        self.label = QLabel("Waiting for data from Page 1...", self)
        self.layout.addWidget(self.label)
        
        self.excitation_settings = ExcitationSettingsWidget(state)
        self.layout.addWidget(self.excitation_settings)

        self.run_btn = QPushButton("Run Simulation", self)
        self.run_btn.clicked.connect(self.run_simulation)
        self.layout.addWidget(self.run_btn)

    def setup_ui_from_state(self):
        """Called by main window right before switching to this page"""
        table_names = list(self.state.dat_tables.keys())
        self.label.setText(f"Loaded tables available for simulation: {', '.join(table_names)}")

    def run_simulation(self):
        # 1. Pull the tables loaded by Page 1
        tables = self.state.dat_tables
        
        # 2. (Run your simulation logic here using tables)
        sim_results = "Simulation Matrix Output" 
        
        # 3. Save results back into the state for Page 3 to use
        self.state.raw_results = sim_results
        
        # 4. Tell the main window we are done
        self.sim_completed.emit()