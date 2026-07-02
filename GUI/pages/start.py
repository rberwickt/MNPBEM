# page_two.py
from PySide6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel
from PySide6.QtCore import Signal, QFileSystemWatcher, QUrl
from PySide6.QtGui import QDesktopServices
from ..simulation_state import SimulationState
from pathlib import Path
class StartPage(QWidget):
    settings_completed = Signal()  # Alert main.py when simulation finishes

    def __init__(self, state: SimulationState):
        super().__init__()
        self.state = state 
        
        self.layout = QVBoxLayout(self)
        
        self.import_btn = QPushButton("Import Functions and .dat Tables", self)
        self.import_btn.clicked.connect(self._open_import)
        self.layout.addWidget(self.import_btn)

        self.calc_watcher = QFileSystemWatcher(self)
        self.calc_watcher.directoryChanged.connect(self.on_folder_updated)

        self.mat_watcher = QFileSystemWatcher(self)
        self.mat_watcher.directoryChanged.connect(self.on_folder_updated)

        self.user_path = Path("../user-defined").resolve()
        self.user_path.mkdir(parents=True, exist_ok=True)

        self.calc_path = Path("../user-defined/calculations").resolve()
        self.calc_path.mkdir(parents=True, exist_ok=True)

        self.mat_path = Path("../user-defined/materials").resolve()
        self.mat_path.mkdir(parents=True, exist_ok=True)
    
        self.user_url = QUrl.fromLocalFile(str(self.user_path))
        self.calc_url = QUrl.fromLocalFile(str(self.calc_path))
        self.mat_url = QUrl.fromLocalFile(str(self.mat_path))

        self.calc_watcher.addPath(str(self.calc_path)) # watch for added files so that we can update dynamically
        self.mat_watcher.addPath(str(self.mat_path)) # watch for added files so that we can update dynamically

        self.load_calculations()
        self.load_materials()

        self.open_folder_btn = QPushButton("Open User-Defined Content Folder")
        self.open_folder_btn.clicked.connect(self.open_file_directory)
        self.layout.addWidget(self.open_folder_btn)
        self.run_btn = QPushButton("Continue to Simulation", self)
        self.run_btn.clicked.connect(self.finish_loading)
        self.layout.addWidget(self.run_btn)
        
    def open_file_directory(self):
        QDesktopServices.openUrl(self.user_url)

    def on_calc_updated(self, path):
        print(f"The user added, removed, or changed something in: {path}")
        # Trigger your UI update or processing logic here!
    
    def load_calculations(self):
        pass
    def load_materials(self):
        pass

    def finish_loading(self):
        # load all of the modules into the state and progress onto simulation
        self.settings_completed.emit()
    


