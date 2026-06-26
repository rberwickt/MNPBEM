from PySide6.QtGui import QIcon, QKeySequence
from PySide6.QtWidgets import QMainWindow
import sys
from PySide6.QtWidgets import QApplication

# example ripped from the qt for python docs

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Eartquakes information")

        # Menu
        self.menu = self.menuBar()
        file_menu = self.menu.addMenu("File")

        # Exit QAction
        file_menu.addAction(QIcon.fromTheme(QIcon.ThemeIcon.ApplicationExit),
                            "Exit", QKeySequence.StandardKey.Quit, self.close)

        # Status Bar
        self.status = self.statusBar()
        self.status.showMessage("Data loaded and plotted")

        # Window dimensions
        geometry = self.screen().availableGeometry()
        self.setFixedSize(geometry.width() * 0.8, geometry.height() * 0.7)

if __name__ == "__main__":
    app = QApplication(sys.argv) # can probably just leave as [] instead of sys.argv, could have it be the target file directory later?
    # maybe a config file input/address?
    window = MainWindow()
    window.show()
    sys.exit(app.exec())