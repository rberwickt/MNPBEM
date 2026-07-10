from PySide6.QtWidgets import QWidget, QVBoxLayout, QScrollArea, QGridLayout
from PySide6.QtCore import Qt
from ..simulation_state import SimulationState
from ..widgets.calculation_item import CalculationItemWidget
from ..widgets.calculation_figure import CalculationFigure
from matplotlib.figure import Figure
import numpy as np


class ProcessingPage(QWidget):
    CARD_MIN_WIDTH = 560 # tentative on hardcoding this
    

    def __init__(self, state: SimulationState):
        super().__init__()
        self.state = state
        self.items = []

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)

        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        self.content_widget = QWidget()
        self.grid_layout = QGridLayout(self.content_widget)
        self.grid_layout.setContentsMargins(12, 12, 12, 12)
        self.grid_layout.setHorizontalSpacing(12)
        self.grid_layout.setVerticalSpacing(12)
        self.scroll_area.setWidget(self.content_widget)
        root_layout.addWidget(self.scroll_area)
        
        for i in range(5):
            item = CalculationItemWidget(f"Calculation {i+1}")
            self.items.append(item)
        
        # test plotting stuff
        t = np.arange(0.0, 2.0, 0.01)
        s = 1 + np.sin(2 * np.pi * t)

        for item in self.items:
            fig = Figure()
            ax = fig.add_subplot(111)
            ax.plot(t, s)
            ax.set( # mainly taken from the matplotlib example (so ignore the data)
                xlabel = 'time (s)',
                ylabel = 'voltage (mV)',
                title = 'About as simple as it gets, folks'
            )
            ax.grid()

            figure = CalculationFigure(fig)
            item.container_layout.addWidget(figure)
            item.content_container.setVisible(True)
            item.update_arrow_ui()
            item.updateGeometry()

        self._reflow_grid()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._reflow_grid()

    def _reflow_grid(self):
        viewport_width = max(1, self.scroll_area.viewport().width())

        left, _, right, _ = self.grid_layout.getContentsMargins()
        spacing = max(0, self.grid_layout.horizontalSpacing())
        usable_width = max(1, viewport_width - left - right)

        hint_widths = []
        for item in self.items:
            hint_widths.append(item.minimumSizeHint().width())
            hint_widths.append(item.sizeHint().width())
        effective_card_width = max([self.CARD_MIN_WIDTH] + hint_widths) + 8

        # Account for spacing in the fit calculation so we wrap before the
        # last card is partially clipped at the right edge.
        cols = max(1, (usable_width + spacing) // (effective_card_width + spacing))

        while self.grid_layout.count():
            self.grid_layout.takeAt(0)

        for idx, item in enumerate(self.items):
            row = idx // cols
            col = idx % cols
            self.grid_layout.addWidget(item, row, col)

        for col in range(cols):
            self.grid_layout.setColumnStretch(col, 1)
        self.content_widget.adjustSize()

    def setup_ui_from_state(self):
        """Called by main window right before switching to this page"""
        pass