from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout,
                               QLabel, QCheckBox, QPushButton, QFrame, QSizePolicy)
from PySide6.QtCore import Qt, QSize

class CalculationItemWidget(QWidget):
    
    def __init__(self, title_text, parent=None):
        super().__init__(parent)

        self.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Preferred)

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        self.title_bar = QFrame()
        self.title_bar.setFrameShape(QFrame.StyledPanel)
        self.title_bar.setFixedHeight(40)

        self.title_bar.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.title_bar.setStyleSheet("background-color: #f0f0f0; border-radius: 4px;")

        self.title_layout = QHBoxLayout(self.title_bar)
        self.title_layout.setContentsMargins(10, 5, 10, 5)

        self.checkbox = QCheckBox()
        self.checkbox.setCursor(Qt.PointingHandCursor)

        self.toggle_button = QPushButton("▼")
        self.toggle_button.setFixedWidth(30)
        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(True) 
        self.toggle_button.setCursor(Qt.PointingHandCursor)
        self.toggle_button.setStyleSheet("border: none; font-weight: bold;")
        self.toggle_button.setVisible(False) 
        self.toggle_button.clicked.connect(self.toggle_content)

        self.title_label = QLabel(title_text)
        self.title_label.setStyleSheet("font-weight: bold; margin-left: 5px;")

        self.title_layout.addWidget(self.checkbox)
        self.title_layout.addWidget(self.toggle_button)
        self.title_layout.addWidget(self.title_label)
        self.title_layout.addStretch()

        self.content_container = QWidget()
        self.container_layout = QVBoxLayout(self.content_container)
        self.container_layout.setContentsMargins(0, 0, 0, 0)
        self.content_container.setVisible(False) 
        
        self.content_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

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