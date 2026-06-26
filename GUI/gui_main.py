# main.py
import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QStackedWidget
from PySide6.QtGui import QIcon

# Import all the pages here

class MainController(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyMNPBEM GUI")
        self.setWindowIcon(QIcon("./images/Landes_group_logo_cropped.png")) 

        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)


if __name__ == "__main__":
    app = QApplication(sys.argv) # can probably just leave as [] instead of sys.argv, could have it be the target file directory later?
    # maybe a config file input/address?
    window = MainController()
    window.show()
    sys.exit(app.exec())