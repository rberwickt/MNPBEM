from PySide6.QtWidgets import (QDialog, QFormLayout, QTextBrowser,
                               QLabel)
#from PySide6.QtCore import Qt
#from PySide6.QtGui import 
from ..simulation_state import SimulationState
class StateDebugDialog(QDialog):
    def __init__(self, state: SimulationState, parent=None):
        super().__init__(parent)
        self.state = state  # Keep a reference to the data struct
        self.setWindowTitle("DEBUG: State View")
        self.resize(900, 600)

        dlg_layout = QFormLayout(self)
        
        dlg_layout.addRow(QLabel("Close Debug Window to Continue Using Main Window (not multi-threaded)"))
        self.state_data = QTextBrowser()
        # cut off the SimulationState( at the start, and the ) at the end
        formatted_state = repr(self.state)[16:-1].replace(",", "<br>")
        html_text = f"""
            <div style="column-count: 3; column-gap: 20px;">
                {formatted_state}
            </div>
            """
        self.state_data.setHtml(html_text)
        dlg_layout.addRow("Data:", self.state_data)

