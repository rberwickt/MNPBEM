# page_two.py
from PySide6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel
from PySide6.QtCore import Signal

class ProcessingPage(QWidget):
    

    def __init__(self, state_reference):
        super().__init__()
        self.state = state_reference  # Access to the shared state
        
        self.layout = QVBoxLayout(self)
        
        self.run_btn = QPushButton("Process", self)
        self.layout.addWidget(self.run_btn)

    def setup_ui_from_state(self):
        """Called by main window right before switching to this page"""
        pass