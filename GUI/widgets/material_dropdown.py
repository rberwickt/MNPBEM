from PySide6.QtCore import Qt
from PySide6.QtWidgets import QComboBox, QWidget, QFormLayout
from ..simulation_state import SimulationState
class MaterialComboBox(QComboBox):

    def __init__(self, state: SimulationState, parent=None): 
        # the attr is the variable inside the state that the material is tied to
        # index is used to choose an index inside that variable if needed
        super().__init__(parent)
        self.state = state
        # a labeled combo box
        self.addItems(list(self.state.loaded_dielectrics.keys()))
        self.setPlaceholderText("")

        # Set to -1 so the placeholder actually shows up
        self.setCurrentIndex(-1)
