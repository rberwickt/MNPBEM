# page_two.py
from PySide6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel, QScrollArea, QHBoxLayout, QGroupBox
from PySide6.QtCore import Signal, QFileSystemWatcher, QUrl, Qt
from PySide6.QtGui import QDesktopServices
from ..simulation_state import SimulationState
from ...mnpbem.materials.eps_table import EpsTable
from pathlib import Path
import importlib, sys
class StartPage(QWidget):
    settings_completed = Signal()  # Alert main.py when simulation finishes
    USER_DIR = Path(__file__).parent.parent / "user-defined"
    MAT_DIR = USER_DIR / "materials"
    CALC_DIR = USER_DIR / "calculations"

    def __init__(self, state: SimulationState):
        super().__init__()
        self.state = state 
        
        self.layout = QVBoxLayout(self)
        self.displays_layout = QHBoxLayout()
        # initialize everything for the user defined content imports
        self.calc_watcher = QFileSystemWatcher(self)
        self.calc_watcher.directoryChanged.connect(self.load_calculations)

        self.mat_watcher = QFileSystemWatcher(self)
        self.mat_watcher.directoryChanged.connect(self.load_materials)

        self.user_path = self.USER_DIR.resolve()
        self.user_path.mkdir(parents=True, exist_ok=True)

        self.calc_path = self.CALC_DIR.resolve()
        self.calc_path.mkdir(parents=True, exist_ok=True)

        self.mat_path = self.MAT_DIR.resolve()
        self.mat_path.mkdir(parents=True, exist_ok=True)
    
        self.user_url = QUrl.fromLocalFile(str(self.user_path))
        self.calc_url = QUrl.fromLocalFile(str(self.calc_path))
        self.mat_url = QUrl.fromLocalFile(str(self.mat_path))

        self.calc_watcher.addPath(str(self.calc_path)) # watch for added files so that we can update dynamically
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

        self.calc_display = QScrollArea()
        self.calc_display.setWidgetResizable(True)  # allows inner widgets to scale
        #self.calc_display.setFixedHeight(300)       # fixes the height so it won't grow instead of the inner widgets
        self.calc_display.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.calc_content = QWidget()
        self.calc_layout = QVBoxLayout(self.calc_content)
        self.calc_layout.setAlignment(Qt.AlignTop)

        self.calc_display.setWidget(self.calc_content)
        self.calc_group = QGroupBox("Loaded Calculations:")
        calc_group_layout = QVBoxLayout(self.calc_group)
        calc_group_layout.addWidget(self.calc_display)
        self.displays_layout.addWidget(self.calc_group)

        self.layout.addWidget(QLabel("Non-local pairs not yet supported! (returns more than one callable)"))
        self.layout.addWidget(QLabel("See user-defined/materials/vacuum.py for example function header"))
        # could try an elif and name the function in the materials file something different if it returns more than one callable? 
        #   question for later: will it only ever be 1 or 2 callables returned?
        self.layout.addLayout(self.displays_layout)

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

    
    def load_calculations(self):
        pass
    def load_materials(self):
        for file_path in self.MAT_DIR.glob('*.*'):
            if file_path.name.startswith('__'):
                continue

            module_name = file_path.stem

            try:
                # .dat case
                if file_path.suffix == '.dat':
                    plugin_obj = EpsTable(str(file_path))
                    self.state.loaded_dielectrics[module_name] = plugin_obj
                    print(f"Successfully loaded tabulated data: {module_name}")

                # .py case
                elif file_path.suffix == '.py':
                    project_root = str(self.USER_DIR.parent)
                
                    if project_root not in sys.path:
                        sys.path.insert(0, project_root)
                    sys.modules.pop(module_name, None)
                    spec = importlib.util.spec_from_file_location(module_name, str(file_path))

                    if spec is None or spec.loader is None:
                        continue
                
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[module_name] = module
                    spec.loader.exec_module(module)

                    if hasattr(module, 'generate_eps_func'):
                        plugin_obj = module.generate_eps_func()
                        if callable(plugin_obj):
                            self.state.loaded_dielectrics[module_name] = plugin_obj
                            print(f"Successfully loaded script: {module_name}")
                        else:
                            print(f"{file_path.name} 'generate_eps_func' did not return a callable object.")
                    else:
                        print(f"{file_path.name} is missing 'generate_eps_func' function.")

            except Exception as e:
                print(f"Error importing {file_path.name}: {e}")
        print(f"LOADED {len(self.state.loaded_dielectrics)} MATERIALS")
        self.update_material_list()

    def update_material_list(self):
        self._clear_layout(self.mat_layout)
        for material in self.state.loaded_dielectrics.keys():
            self.mat_layout.addWidget(QLabel(material))
    def update_calculation_list(self):
        self._clear_layout(self.calc_layout)

    def finish_loading(self):
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


