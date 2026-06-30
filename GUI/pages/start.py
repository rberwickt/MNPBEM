# page_two.py
from PySide6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel
from PySide6.QtCore import Signal

class StartPage(QWidget):
    settings_completed = Signal()  # Alert main.py when simulation finishes

    def __init__(self, state_reference):
        super().__init__()
        self.state = state_reference  # Access to the shared state
        
        self.layout = QVBoxLayout(self)
        self.label = QLabel("Load data", self)
        self.layout.addWidget(self.label)
        
        self.run_btn = QPushButton("Continue to Simulation", self)
        self.run_btn.clicked.connect(self.finish_loading)
        self.layout.addWidget(self.run_btn)

    def finish_loading(self):
        self.settings_completed.emit()


