# main.py
import sys, os
from PySide6.QtWidgets import QApplication, QMainWindow, QStackedWidget
from PySide6.QtGui import QIcon
from .simulation_state import SimulationState
from .pages.start import StartPage
from .pages.simulation import SimulationPage
from .pages.post_processing import ProcessingPage
# Import all the pages here

class MainController(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyMNPBEM GUI")
        
        # icon path
        base_dir = os.path.dirname(os.path.abspath(__file__))
        icon_path = os.path.join(base_dir, "images", "Landes_group_logo_cropped.png")
        self.setWindowIcon(QIcon(icon_path))
        
        self.resize(900, 600)

        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)

        self.state = SimulationState() # this is where all the data gets stored (may become RAM heavy when it stores the sim data)

        self.page1 = StartPage(self.state)
        self.page2 = SimulationPage(self.state)
        self.page3 = ProcessingPage(self.state)

        self.stacked_widget.addWidget(self.page1)
        self.stacked_widget.addWidget(self.page2)
        self.stacked_widget.addWidget(self.page3)

        self.page1.settings_completed.connect(self.go_to_sim)

    def go_to_sim(self):
        # Sim page needs to refresh its UI based on what was loaded in the initial page
        self.page2.setup_ui_from_state() 
        self.stacked_widget.setCurrentWidget(self.page2)


if __name__ == "__main__":
    app = QApplication(sys.argv) # can probably just leave as [] instead of sys.argv, could have it be the target file directory later?
    # maybe a config file input/address?
    window = MainController()
    window.show()
    sys.exit(app.exec())