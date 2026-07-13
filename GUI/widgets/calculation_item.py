from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout,
                               QLabel, QCheckBox, QPushButton, QFrame, QSizePolicy)
from PySide6.QtCore import Qt, QSize
from typing import Callable

class CalculationItemWidget(QWidget):
    
    def __init__(self, title_text, parent=None):
        super().__init__(parent)

        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(6, 6, 6, 6)
        self.main_layout.setSpacing(6)
        self.main_layout.setAlignment(Qt.AlignTop)

        self.setObjectName("calc_item")

        self.action_buttons = []

        self.title_bar = QFrame()
        self.title_bar.setObjectName("calc_title_bar")
        self.title_bar.setFrameShape(QFrame.StyledPanel)
        self.title_bar.setFixedHeight(40)

        self.title_bar.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.title_layout = QHBoxLayout(self.title_bar)
        self.title_layout.setContentsMargins(10, 5, 10, 5)

        self.checkbox = QCheckBox()
        self.checkbox.setCursor(Qt.PointingHandCursor)

        self.toggle_button = QPushButton("▼")
        self.toggle_button.setObjectName("calc_toggle_button")
        self.toggle_button.setFixedWidth(30)
        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(True) 
        self.toggle_button.setCursor(Qt.PointingHandCursor)
        self.toggle_button.setVisible(False) 
        self.toggle_button.clicked.connect(self.toggle_content)

        self.title_label = QLabel(title_text)
        self.title_label.setObjectName("calc_title_label")

        self.title_layout.addWidget(self.checkbox)
        self.title_layout.addWidget(self.toggle_button)
        self.title_layout.addWidget(self.title_label)
        self.title_layout.addStretch(1)

        self.action_layout = QHBoxLayout()
        self.action_layout.setContentsMargins(0, 0, 0, 0)
        self.action_layout.setSpacing(6)
        self.title_layout.addLayout(self.action_layout)

        self.content_container = QWidget()
        self.container_layout = QVBoxLayout(self.content_container)
        self.container_layout.setContentsMargins(8, 0, 8, 8)
        self.container_layout.setSpacing(6)
        self.content_container.setVisible(False) 
        
        self.content_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.main_layout.addWidget(self.title_bar)
        self.main_layout.addWidget(self.content_container)

    # size hint stuff 

    def sizeHint(self) -> QSize:
        title_hint = self.title_bar.sizeHint()
        base_width = max(300, title_hint.width()) 
        
        if self.content_container.isVisible() and self.container_layout.count() > 0:
            content_hint = self.content_container.sizeHint()
            return QSize(max(base_width, content_hint.width()), 
                         title_hint.height() + content_hint.height())
        else:
            return QSize(base_width, title_hint.height())

    def minimumSizeHint(self) -> QSize:
        title_hint = self.title_bar.minimumSizeHint()
        base_width = max(300, title_hint.width())
        
        if self.content_container.isVisible() and self.container_layout.count() > 0:
            content_min = self.content_container.minimumSizeHint()
            return QSize(max(base_width, content_min.width()), 
                         title_hint.height() + content_min.height())
        return QSize(base_width, title_hint.height())

    def set_actions(self, actions: list[tuple[str, Callable]]):
        while self.action_layout.count():
            item = self.action_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

        self.action_buttons = []

        for label, callback in actions:
            button = QPushButton(label)
            button.setObjectName("calc_header_action_button")
            button.setCursor(Qt.PointingHandCursor)
            button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            button.clicked.connect(callback)
            self.action_layout.addWidget(button)
            self.action_buttons.append(button)

        self.title_layout.invalidate()
        self.updateGeometry()


    def set_content_widget(self, custom_widget: QWidget):
        while self.container_layout.count():
            item = self.container_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

        self.container_layout.addWidget(custom_widget)
        # recalculate geometry
        self.updateGeometry()

    def is_selected(self) -> bool:
        return self.checkbox.isVisible() and self.checkbox.isChecked()

    def mark_as_processed(self):
        self.checkbox.setVisible(False)
        self.toggle_button.setVisible(True)
        self.content_container.setVisible(True)
        self.update_arrow_ui()
        self.updateGeometry()

    def toggle_content(self):
        self.content_container.setVisible(self.toggle_button.isChecked())
        self.update_arrow_ui()
        self.updateGeometry()

    def update_arrow_ui(self):
        self.toggle_button.setText("▼" if self.content_container.isVisible() else "▶")