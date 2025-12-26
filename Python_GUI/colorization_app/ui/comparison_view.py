"""
Comparison View Widget for Before/After Image Display
Provides side-by-side and overlay comparison modes
"""

import logging
from typing import Optional

import numpy as np
from PyQt5.QtCore import QRect, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QLabel,
    QRadioButton,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

# Set up logging
logger = logging.getLogger(__name__)


class ImageDisplayLabel(QLabel):
    """Custom QLabel for displaying images with proper scaling"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(380, 250)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("""
            QLabel {
                border: 2px solid #555;
                background-color: #2b2b2b;
                border-radius: 8px;
                padding: 4px;
            }
        """)
        self.setAlignment(Qt.AlignCenter)
        self.setScaledContents(False)
        self._pixmap = None

    def setPixmap(self, pixmap):
        """Set pixmap with proper scaling"""
        self._pixmap = pixmap
        if pixmap is not None:
            # Scale pixmap to fit label while maintaining aspect ratio
            scaled_pixmap = pixmap.scaled(
                self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            super().setPixmap(scaled_pixmap)
        else:
            super().setPixmap(pixmap)

    def resizeEvent(self, event):
        """Handle resize events to rescale image"""
        super().resizeEvent(event)
        if self._pixmap is not None:
            scaled_pixmap = self._pixmap.scaled(
                self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            super().setPixmap(scaled_pixmap)


class ComparisonView(QWidget):
    """Main comparison view widget with multiple display modes"""

    # Signals
    mode_changed = pyqtSignal(str)  # Emits current mode

    # Display modes
    SIDE_BY_SIDE = "side_by_side"
    BEFORE_ONLY = "before_only"
    AFTER_ONLY = "after_only"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_mode = self.SIDE_BY_SIDE
        self.before_image = None
        self.after_image = None
        self.last_panel_size = None  # For persistent sizing

        self.setup_ui()
        self.setup_connections()

    def setup_ui(self):
        """Setup the user interface"""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(8)

        # Mode selection
        mode_frame = QFrame()
        mode_frame.setFrameStyle(QFrame.Box)
        mode_frame.setStyleSheet("""
            QFrame {
                border: 1px solid #555;
                border-radius: 6px;
                background-color: #3c3c3c;
                padding: 4px;
            }
        """)
        mode_layout = QHBoxLayout(mode_frame)
        mode_layout.setContentsMargins(8, 4, 8, 4)

        # Mode radio buttons
        self.mode_group = QButtonGroup(self)

        self.side_by_side_radio = QRadioButton("Side by Side")
        self.before_only_radio = QRadioButton("Before Only")
        self.after_only_radio = QRadioButton("After Only")

        self.side_by_side_radio.setChecked(True)

        self.mode_group.addButton(self.side_by_side_radio, 0)
        self.mode_group.addButton(self.before_only_radio, 1)
        self.mode_group.addButton(self.after_only_radio, 2)

        mode_layout.addWidget(self.side_by_side_radio)
        mode_layout.addWidget(self.before_only_radio)
        mode_layout.addWidget(self.after_only_radio)
        mode_layout.addStretch()

        main_layout.addWidget(mode_frame)

        # Display area
        self.display_widget = QWidget()
        self.display_layout = QHBoxLayout(self.display_widget)
        self.display_layout.setContentsMargins(5, 5, 5, 5)
        self.display_layout.setSpacing(8)

        # Side-by-side view
        self.side_by_side_widget = QWidget()
        sbs_layout = QHBoxLayout(self.side_by_side_widget)
        sbs_layout.setContentsMargins(5, 5, 5, 5)
        sbs_layout.setSpacing(8)

        # Before image (left)
        before_frame = QFrame()
        before_frame.setFrameStyle(QFrame.Box)
        before_frame.setMinimumSize(400, 300)
        before_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        before_layout = QVBoxLayout(before_frame)
        before_layout.setContentsMargins(5, 5, 5, 5)

        before_label = QLabel("BEFORE (Grayscale)")
        before_label.setAlignment(Qt.AlignCenter)
        before_label.setStyleSheet(
            "font-weight: bold; color: #fff; font-size: 11pt; padding: 4px;"
        )
        self.before_display = ImageDisplayLabel()
        self.before_display.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        before_layout.addWidget(before_label)
        before_layout.addWidget(self.before_display)

        # After image (right)
        after_frame = QFrame()
        after_frame.setFrameStyle(QFrame.Box)
        after_frame.setMinimumSize(400, 300)
        after_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        after_layout = QVBoxLayout(after_frame)
        after_layout.setContentsMargins(5, 5, 5, 5)

        after_label = QLabel("AFTER (Colorized)")
        after_label.setAlignment(Qt.AlignCenter)
        after_label.setStyleSheet(
            "font-weight: bold; color: #fff; font-size: 11pt; padding: 4px;"
        )
        self.after_display = ImageDisplayLabel()
        self.after_display.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        after_layout.addWidget(after_label)
        after_layout.addWidget(self.after_display)

        # Add frames with equal stretch to ensure identical sizes
        sbs_layout.addWidget(before_frame, 1)
        sbs_layout.addWidget(after_frame, 1)

        # Single image display
        self.single_display = ImageDisplayLabel()
        self.single_display.setMinimumSize(400, 300)
        self.single_display.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Add widgets to display layout (only one will be visible at a time)
        self.display_layout.addWidget(self.side_by_side_widget)
        self.display_layout.addWidget(self.single_display)

        main_layout.addWidget(self.display_widget)

        # Initially show side-by-side
        self.single_display.hide()

        # Set size policy
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Style the widget
        self.setStyleSheet("""
            QWidget {
                background-color: #2b2b2b;
                color: #ffffff;
            }
            QRadioButton {
                color: #ffffff;
                font-weight: bold;
            }
            QRadioButton::indicator {
                width: 16px;
                height: 16px;
            }
            QRadioButton::indicator::unchecked {
                border: 2px solid #555;
                border-radius: 9px;
                background-color: #2b2b2b;
            }
            QRadioButton::indicator::checked {
                border: 2px solid #0078d4;
                border-radius: 9px;
                background-color: #0078d4;
            }
            QFrame {
                background-color: #3c3c3c;
                border: 1px solid #555;
                border-radius: 5px;
            }
        """)

    def setup_connections(self):
        """Setup signal connections"""
        self.mode_group.buttonClicked.connect(self.on_mode_changed)

    def on_mode_changed(self, button):
        """Handle mode change"""
        mode_map = {
            0: self.SIDE_BY_SIDE,
            1: self.BEFORE_ONLY,
            2: self.AFTER_ONLY,
        }

        mode_id = self.mode_group.id(button)
        self.current_mode = mode_map.get(mode_id, self.SIDE_BY_SIDE)

        # Store current panel size before switching
        self._store_panel_size()

        self.update_display_mode()

        # Restore panel size after switching
        self._restore_panel_size()

        self.mode_changed.emit(self.current_mode)

    def update_display_mode(self):
        """Update the display based on current mode"""
        # Hide all widgets first
        self.side_by_side_widget.hide()
        self.single_display.hide()

        if self.current_mode == self.SIDE_BY_SIDE:
            self.side_by_side_widget.show()
        elif self.current_mode == self.BEFORE_ONLY:
            self.single_display.show()
            if self.before_image:
                self.single_display.setPixmap(self.before_image)
        elif self.current_mode == self.AFTER_ONLY:
            self.single_display.show()
            if self.after_image:
                self.single_display.setPixmap(self.after_image)

    def set_before_image(self, image_array: np.ndarray):
        """
        Set the before (grayscale) image

        Args:
            image_array: Image as numpy array (RGB or grayscale)
        """
        try:
            # Ensure image is in proper format
            if image_array is None:
                return

            # Ensure image is uint8
            if image_array.dtype != np.uint8:
                if image_array.max() <= 1.0:
                    image_array = (image_array * 255).astype(np.uint8)
                else:
                    image_array = np.clip(image_array, 0, 255).astype(np.uint8)

            # Convert numpy array to QPixmap
            if len(image_array.shape) == 2:  # Grayscale
                height, width = image_array.shape
                # Ensure contiguous array
                if not image_array.flags["C_CONTIGUOUS"]:
                    image_array = np.ascontiguousarray(image_array)
                bytes_per_line = width
                q_image = QImage(
                    image_array.data,
                    width,
                    height,
                    bytes_per_line,
                    QImage.Format_Grayscale8,
                )
            else:  # RGB
                height, width, channel = image_array.shape
                # Ensure contiguous array
                if not image_array.flags["C_CONTIGUOUS"]:
                    image_array = np.ascontiguousarray(image_array)
                bytes_per_line = 3 * width
                q_image = QImage(
                    image_array.data,
                    width,
                    height,
                    bytes_per_line,
                    QImage.Format_RGB888,
                )

            self.before_image = QPixmap.fromImage(q_image)
            self.before_display.setPixmap(self.before_image)

            # Update single display if showing before only
            if self.current_mode == self.BEFORE_ONLY:
                self.single_display.setPixmap(self.before_image)

        except Exception as e:
            logger.error(f"Error setting before image: {e}")

    def set_after_image(self, image_array: np.ndarray):
        """
        Set the after (colorized) image

        Args:
            image_array: Image as numpy array (RGB)
        """
        try:
            # If None, clear the after image display
            if image_array is None:
                self.after_image = None
                self.after_display.clear()
                self.after_display.setText("Colorized")
                return

            # Ensure image is uint8
            if image_array.dtype != np.uint8:
                if image_array.max() <= 1.0:
                    image_array = (image_array * 255).astype(np.uint8)
                else:
                    image_array = np.clip(image_array, 0, 255).astype(np.uint8)

            # Convert numpy array to QPixmap
            if len(image_array.shape) == 3:
                height, width, channel = image_array.shape
                # Ensure contiguous array
                if not image_array.flags["C_CONTIGUOUS"]:
                    image_array = np.ascontiguousarray(image_array)
                bytes_per_line = 3 * width
                q_image = QImage(
                    image_array.data,
                    width,
                    height,
                    bytes_per_line,
                    QImage.Format_RGB888,
                )

                self.after_image = QPixmap.fromImage(q_image)
                self.after_display.setPixmap(self.after_image)

                # Update single display if showing after only
                if self.current_mode == self.AFTER_ONLY:
                    self.single_display.setPixmap(self.after_image)

        except Exception as e:
            logger.error(f"Error setting after image: {e}")

    def clear_images(self):
        """Clear both images"""
        self.before_image = None
        self.after_image = None
        self.before_display.clear()
        self.after_display.clear()
        self.single_display.clear()

    def get_current_mode(self) -> str:
        """Get the current display mode"""
        return self.current_mode

    def set_mode(self, mode: str):
        """
        Set the display mode programmatically

        Args:
            mode: Display mode (use class constants)
        """
        mode_map = {
            self.SIDE_BY_SIDE: 0,
            self.BEFORE_ONLY: 1,
            self.AFTER_ONLY: 2,
        }

        if mode in mode_map:
            button_id = mode_map[mode]
            button = self.mode_group.button(button_id)
            if button:
                button.setChecked(True)
                self.current_mode = mode
                self.update_display_mode()

    def _store_panel_size(self):
        """Store current panel size for persistent sizing"""
        try:
            if self.current_mode == self.SIDE_BY_SIDE:
                if hasattr(self, "before_display") and self.before_display.isVisible():
                    self.last_panel_size = self.before_display.size()
            elif self.current_mode in [self.BEFORE_ONLY, self.AFTER_ONLY]:
                if hasattr(self, "single_display") and self.single_display.isVisible():
                    self.last_panel_size = self.single_display.size()
        except:
            pass

    def _restore_panel_size(self):
        """Restore panel size after mode switch"""
        try:
            if hasattr(self, "last_panel_size") and self.last_panel_size:
                if self.current_mode == self.SIDE_BY_SIDE:
                    if hasattr(self, "before_display") and hasattr(
                        self, "after_display"
                    ):
                        # Force both panels to have the same size
                        self.before_display.setMinimumSize(self.last_panel_size)
                        self.after_display.setMinimumSize(self.last_panel_size)
                        # Process events to apply the sizing
                        from PyQt5.QtWidgets import QApplication

                        QApplication.processEvents()
                elif self.current_mode in [self.BEFORE_ONLY, self.AFTER_ONLY]:
                    if hasattr(self, "single_display"):
                        self.single_display.setMinimumSize(self.last_panel_size)
                        QApplication.processEvents()
        except:
            pass


# Import QImage here to avoid circular import issues
from PyQt5.QtGui import QImage
