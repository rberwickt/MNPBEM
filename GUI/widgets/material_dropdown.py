from PySide6.QtCore import Qt
from PySide6.QtWidgets import QComboBox, QWidget, QFormLayout
from ..simulation_state import SimulationState
# a selection box for choosing the material of one of the structures in the simulation. Meant to be modular
# should tie to a specific place in the state to choose material?
class MaterialComboBox(QWidget):

    def __init__(self, state: SimulationState, index: int, label: str, parent=None):
        super().__init__(parent)
        self.state = state
        self.index = index
        self.label = label
        self.prev_text = None
        # a labeled combo box
        self.layout = QFormLayout(self)
        self.combo = QComboBox()
        self.combo.addItems(list(self.state.loaded_dielectrics.keys()))
        self.combo.currentTextChanged.connect(self._material_changed)

        self.layout.addRow(self.label, self.combo)
    
    def _material_changed(self, text: str):
        if text == self.prev_text:
            return
        self.prev_text = text

        if len(self.state.materials) <= self.index:
            print(f"MaterialComboBox {self.label} with index {self.index}: OUT OF BOUNDS")
        else:
            try:
                self.state.materials[self.index] = self.state.loaded_dielectrics[text]
            except Exception as e:
                print(f"ERROR WITH {text} AT INDEX {self.index}: {e}")

        
    def setCurrentText(self, text: str):
        self.combo.setCurrentText(text) # should trigger signal?
