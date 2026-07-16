# page_two.py
from PySide6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel, QScrollArea, QHBoxLayout, QGroupBox, QFormLayout, QSpinBox, QMessageBox
from PySide6.QtCore import Signal, QFileSystemWatcher, QUrl, Qt
from PySide6.QtGui import QDesktopServices

from ..simulation_state import SimulationState
from pathlib import Path


class StartPage(QWidget):
    settings_completed = Signal()  # Alert main.py when simulation finishes
    USER_DIR = Path(__file__).parent.parent / "user-defined"
    MAT_DIR = USER_DIR / "materials"

    def __init__(self, state: SimulationState):
        super().__init__()
        self.state = state 
        
        self.layout = QVBoxLayout(self)
        self.displays_layout = QHBoxLayout()
        # initialize everything for the user defined content imports

        self.mat_watcher = QFileSystemWatcher(self)
        self.mat_watcher.directoryChanged.connect(self.load_materials)

        self.user_path = self.USER_DIR.resolve()
        self.user_path.mkdir(parents=True, exist_ok=True)


        self.mat_path = self.MAT_DIR.resolve()
        self.mat_path.mkdir(parents=True, exist_ok=True)
    
        self.user_url = QUrl.fromLocalFile(str(self.user_path))
        
        self.mat_url = QUrl.fromLocalFile(str(self.mat_path))

        self.mat_watcher.addPath(str(self.mat_path)) # watch for added files so that we can update dynamically

        # actual GUI of state page
        self.mat_display = QScrollArea()
        self.mat_display.setWidgetResizable(True)  # allows inner widgets to scale
        #self.mat_display.setFixedHeight(300)       # fixes the height so it won't grow instead of the inner widgets
        self.mat_display.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.mat_content = QWidget()
        self.mat_layout = QVBoxLayout(self.mat_content)
        self.mat_layout.setAlignment(Qt.AlignTop)

        self.mat_display.setWidget(self.mat_content)
        self.mat_group = QGroupBox("Loaded Materials:")
        mat_group_layout = QVBoxLayout(self.mat_group)
        mat_group_layout.addWidget(self.mat_display)
        self.displays_layout.addWidget(self.mat_group)


        #self.layout.addWidget(QLabel("Non-local pairs not yet supported (returns more than one callable)"))
        self.layout.addWidget(QLabel("See user-defined/materials/vacuum.py for example function header!"))
        # could try an elif and name the function in the materials file something different if it returns more than one callable? 
        #   question for later: will it only ever be 1 or 2 callables returned?

        self.env_group = QGroupBox("Environment Setup")
        env_layout = QFormLayout(self.env_group)

        self.n_workers_input = QSpinBox()
        self.n_workers_input.setRange(1, 256)
        self.n_workers_input.setValue(int(getattr(self.state, "env_n_workers", 1)))

        self.n_threads_input = QSpinBox()
        self.n_threads_input.setRange(1, 512)
        self.n_threads_input.setValue(int(getattr(self.state, "env_n_threads", 6)))

        self.n_gpus_input = QSpinBox()
        self.n_gpus_input.setRange(0, 16)
        self.n_gpus_input.setValue(int(getattr(self.state, "env_n_gpus_per_worker", 0)))

        env_layout.addRow("Workers", self.n_workers_input)
        env_layout.addRow("Threads", self.n_threads_input)
        env_layout.addRow("GPUs per worker", self.n_gpus_input)

        self.layout.addWidget(self.env_group)
        self.layout.addLayout(self.displays_layout)

        self.load_materials()

        self.open_folder_btn = QPushButton("Open User-Defined Content Folder")
        self.open_folder_btn.clicked.connect(self.open_file_directory)
        self.layout.addWidget(self.open_folder_btn)
        self.run_btn = QPushButton("Continue to Simulation", self)
        self.run_btn.clicked.connect(self.finish_loading)
        self.layout.addWidget(self.run_btn)
        
    def open_file_directory(self):
        QDesktopServices.openUrl(self.user_url)

    
    def load_materials(self):
        self.state.loaded_dielectrics.clear()
        self.state.material_descriptors.clear()

        for file_path in self.MAT_DIR.glob('*.*'):
            if file_path.name.startswith('__'):
                continue

            module_name = file_path.stem

            try:
                if file_path.suffix == '.dat':

                    self.state.loaded_dielectrics.append(module_name)
                    self.state.material_descriptors[module_name] = {
                        "type": "table",
                        "file": str(file_path.resolve())
                    }

                elif file_path.suffix == '.py':
                    # Register module path only; do not import here.
                    # Some user modules import mnpbem, and importing them on Start
                    # would violate setup_env-before-mnpbem ordering.
                    self.state.loaded_dielectrics.append(module_name)
                    self.state.material_descriptors[module_name] = {
                        "type": "python_module",
                        "module_path": str(file_path.resolve()),
                        "factory": "generate_eps_func"
                    }

            except Exception as e:
                print(f"Error importing {file_path.name}: {e}")

        self.state.loaded_dielectrics.sort()
        self.update_material_list()

    def update_material_list(self):
        self._clear_layout(self.mat_layout)
        for material in self.state.loaded_dielectrics:
            self.mat_layout.addWidget(QLabel(material))

    def finish_loading(self):
        # Configure runtime environment before any mnpbem-dependent pages are created.
        n_workers = int(self.n_workers_input.value())
        n_threads = int(self.n_threads_input.value())
        n_gpus_per_worker = int(self.n_gpus_input.value())

        try:
            from pymnpbem_simulation.env_setup import assert_pre_import, setup_env

            assert_pre_import()
            setup_env(n_threads = n_threads, n_gpus_per_worker = n_gpus_per_worker)
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Environment Setup Failed",
                "Failed to configure environment before simulation imports:\n\n{}".format(exc),
                QMessageBox.Ok
            )
            return

        self.state.env_n_workers = n_workers
        self.state.env_n_threads = n_threads
        self.state.env_n_gpus_per_worker = n_gpus_per_worker

        # load all of the modules into the state and progress onto simulation
        self.settings_completed.emit()
    
    def _clear_layout(self, to_clear):
        if to_clear is not None:
            while to_clear.count():
                item = to_clear.takeAt(0)
                widget = item.widget()
                
                if widget is not None:
                    widget.deleteLater()
                else:
                    sub_layout = item.layout()
                    if sub_layout is not None:
                        # recursively clear the sub layouts (overkill but you never know)
                        self._clear_layout(sub_layout)


