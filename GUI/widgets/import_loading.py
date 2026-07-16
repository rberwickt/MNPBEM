import sys
import time
from PySide6.QtCore import QRunnable, QThreadPool, Signal, QObject, Slot, Qt
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QLabel, QProgressBar, QSpacerItem, QSizePolicy, QHBoxLayout)
from PySide6.QtGui import QFont

# 1. Create a QObject to hold the signals. 
# (QRunnable doesn't inherit from QObject, so it needs a helper to emit signals)
class LoaderSignals(QObject):
    finished = Signal()
    progress = Signal(str)

# 2. Define the background task using QRunnable
class ImportLoader(QRunnable):
    def __init__(self):
        super().__init__()
        self.signals = LoaderSignals()

    def run(self):
        
        self.signals.progress.emit("Loading Simulation Settings...")
        from ..pages.simulation import SimulationPage
        
        self.signals.progress.emit("Loading Post-Processing Calculations...")
        from ..pages.post_processing import ProcessingPage

        self.signals.finished.emit()

class ImportProgressDialog(QDialog):
    """Modal dialog that imports the GUI pages (and therefore mnpbem), which are quite heavy and slow to import."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Importing MNPBEM tools...")
        self.setModal(True)
        self.setMinimumWidth(650)
        self.setMinimumHeight(450)
        self.setWindowFlags(Qt.Window | Qt.WindowTitleHint | Qt.CustomizeWindowHint)
        self.allow_close = False
        
        self.init_ui()
        self.run_import()

    def init_ui(self):
        """Initialize UI components."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(40, 40, 40, 40)
        
        main_layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))
        
        content_layout = QVBoxLayout()
        content_layout.setSpacing(15)
        
        # --- Title ---
        title = QLabel("Importing MNPBEM Tools")
        title.setAlignment(Qt.AlignCenter)
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title.setFont(title_font)
        content_layout.addWidget(title)
        
        self.status_label = QLabel("Waiting to start import...")
        self.status_label.setAlignment(Qt.AlignCenter)
        status_font = QFont()
        status_font.setPointSize(11)
        self.status_label.setFont(status_font)
        # Give the text a subtle muted color
        self.status_label.setStyleSheet("color: #555555;") 
        content_layout.addWidget(self.status_label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(0)  # Indeterminate (bouncing) mode
        self.progress_bar.setFixedHeight(18)  
        self.progress_bar.setTextVisible(False)

        pb_layout = QHBoxLayout()
        pb_layout.addStretch()
        pb_layout.addWidget(self.progress_bar, stretch=4)  # Takes up 80% of the centered block
        pb_layout.addStretch()
        content_layout.addLayout(pb_layout)
        
        main_layout.addLayout(content_layout)
        
        main_layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

    def run_import(self):
        """Run the import in a separate thread."""
        self.threadpool = QThreadPool()
        self.loader = ImportLoader()
        self.loader.signals.progress.connect(self.update_status)
        self.loader.signals.finished.connect(self.on_import_finished)
        self.threadpool.start(self.loader)
    
    @Slot(str)
    def update_status(self, message: str):
        self.status_label.setText(message)

    @Slot()
    def on_import_finished(self):
        self.status_label.setText("Import finished.")
        self.progress_bar.setMaximum(1)
        self.progress_bar.setValue(1)
        self.allow_close = True
        self.accept()  # This safely breaks the .exec() loop and returns QDialog.Accepted

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            event.ignore()  # Prevent Escape from closing it
        else:
            super().keyPressEvent(event)
    
    def closeEvent(self, event):
        if self.allow_close:
            event.accept()
        else:
            event.ignore()  
