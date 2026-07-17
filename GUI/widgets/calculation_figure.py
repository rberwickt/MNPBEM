from PySide6.QtWidgets import QWidget, QVBoxLayout, QSizePolicy
from PySide6.QtCore import QTimer
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.ticker import ScalarFormatter

class CalculationFigure(QWidget):

    def __init__(self, figure: Figure, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(4)

        figure.tight_layout()
        self._disable_axis_offset_text(figure)

        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(0, 260)

        self.canvas = FigureCanvas(figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.setMinimumSize(0, 0)
        self.canvas.updateGeometry()

        self.toolbar = NavigationToolbar(self.canvas, self)
        self.toolbar.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._bind_toolbar_to_canvas()
        self.layout.addWidget(self.toolbar)
        self.layout.addWidget(self.canvas, 1)
        
        # redraw so that it actually displays
        self._bind_toolbar_to_canvas()
        self.canvas.draw_idle()
        QTimer.singleShot(0, self.canvas.draw_idle)

    def _disable_axis_offset_text(self, figure: Figure):
        # Prevent top-left offset labels like "1e-12 + 9.49999" on linear axes.
        for ax in figure.get_axes():
            if ax.get_xscale() == "linear":
                xfmt = ScalarFormatter(useOffset = False)
                ax.xaxis.set_major_formatter(xfmt)
            if ax.get_yscale() == "linear":
                yfmt = ScalarFormatter(useOffset = False)
                ax.yaxis.set_major_formatter(yfmt)

    def _bind_toolbar_to_canvas(self):
        # Matplotlib figure options dialog calls figure.canvas.toolbar.push_current().
        self.canvas.toolbar = self.toolbar
        self.canvas.figure.set_canvas(self.canvas)