from PySide6.QtCore import Qt
from PySide6.QtWidgets import QComboBox
# easy way to make a combobox with tool tips on hover (ideally good for user-created content?)
class ToolTipComboBox(QComboBox):

    def addItemsWithTooltips(self, items: list[tuple[str, str]]):
        """Accepts a list of (text, tooltip) tuples and adds them all at once."""
        for text, tooltip in items:
            self.addItem(text)
            self.setItemData(self.count() - 1, tooltip, Qt.ToolTipRole)