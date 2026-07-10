from PySide6.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

class CalculationFigure(QWidget):

    def __init__(self, figure: Figure, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.canvas = FigureCanvas(figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self._bind_toolbar_to_canvas()
        self.layout.addWidget(self.toolbar)
        self.layout.addWidget(self.canvas)
        
        # redraw so that it actually displays
        self._bind_toolbar_to_canvas()
        self.canvas.draw()

    def _bind_toolbar_to_canvas(self):
        # Matplotlib figure options dialog calls figure.canvas.toolbar.push_current().
        self.canvas.toolbar = self.toolbar
        self.canvas.figure.set_canvas(self.canvas)