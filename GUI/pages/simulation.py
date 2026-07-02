from PySide6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel, QHBoxLayout
from PySide6.QtCore import Signal
from ..simulation_state import SimulationState
from ..widgets.solver_options import SolverOptionsWidget
from ..widgets.excitation_settings import ExcitationSettingsWidget
from ..widgets.material_settings import MaterialOptionsWidget
class SimulationPage(QWidget):
    sim_completed = Signal()  # Alert main.py when simulation finishes

    def __init__(self, state: SimulationState):
        super().__init__()
        self.state = state  # Access to the shared state
        self.columns = QHBoxLayout(self)
        self.col_1 = QVBoxLayout()
        
        self.solver_options = SolverOptionsWidget(state)
        self.col_1.addWidget(self.solver_options)
        
        self.excitation_settings = ExcitationSettingsWidget(state)
        self.col_1.addWidget(self.excitation_settings)

        self.run_btn = QPushButton("Run Simulation", self)
        self.run_btn.clicked.connect(self.run_simulation)
        self.col_1.addWidget(self.run_btn)

        self.col_2 = QVBoxLayout()
        
        self.material_settings = MaterialOptionsWidget(state)
        self.col_2.addWidget(self.material_settings)

        self.columns.addLayout(self.col_1)
        self.columns.addLayout(self.col_2)

    def setup_ui_from_state(self):
        """Called by main window right before switching to this page"""
        pass

    def run_simulation(self):
        # 1. Pull the tables loaded by Page 1
        tables = self.state.dat_tables
        
        # 2. (Run your simulation logic here using tables)
        sim_results = "Simulation Matrix Output" 
        
        # 3. Save results back into the state for Page 3 to use
        self.state.raw_results = sim_results
        
        # 4. Tell the main window we are done
        self.sim_completed.emit()