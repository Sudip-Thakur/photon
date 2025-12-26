"""
Settings Dialog for Colorization App
Provides configuration options for camera, processing, and output settings
"""

import logging
import os
from typing import Any, Dict

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

# Set up logging
logger = logging.getLogger(__name__)


class SettingsDialog(QDialog):
    """Settings dialog for application configuration"""

    settings_changed = pyqtSignal(dict)  # Emits new settings

    def __init__(self, parent=None, current_settings=None):
        """
        Initialize settings dialog

        Args:
            parent: Parent widget
            current_settings: Dictionary of current settings
        """
        super().__init__(parent)
        self.current_settings = current_settings or {}
        self.temp_settings = self.current_settings.copy()

        self.setup_ui()
        self.load_current_settings()
        self.setup_connections()

        # Make dialog modal
        self.setModal(True)
        self.setWindowTitle("Settings")
        self.setMinimumSize(500, 600)

    def setup_ui(self):
        """Setup the user interface"""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)

        # Create tab widget
        self.tab_widget = QTabWidget()

        # Create tabs
        self.create_camera_tab()
        self.create_processing_tab()
        self.create_output_tab()
        self.create_interface_tab()

        layout.addWidget(self.tab_widget)

        # Dialog buttons
        self.button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel | QDialogButtonBox.Apply
        )
        layout.addWidget(self.button_box)

        # Apply dark theme
        self.apply_dark_theme()

    def create_camera_tab(self):
        """Create camera settings tab"""
        camera_tab = QWidget()
        layout = QVBoxLayout(camera_tab)
        layout.setSpacing(15)

        # Camera Selection Group
        camera_group = QGroupBox("Camera Selection")
        camera_form = QFormLayout(camera_group)

        self.camera_combo = QComboBox()
        self.refresh_cameras_btn = QPushButton("Refresh")
        self.refresh_cameras_btn.setMaximumWidth(80)

        camera_selection_layout = QHBoxLayout()
        camera_selection_layout.addWidget(self.camera_combo)
        camera_selection_layout.addWidget(self.refresh_cameras_btn)

        camera_form.addRow("Camera Device:", camera_selection_layout)
        layout.addWidget(camera_group)

        # Camera Properties Group
        props_group = QGroupBox("Camera Properties")
        props_form = QFormLayout(props_group)

        self.resolution_combo = QComboBox()
        self.resolution_combo.addItems(
            ["640x480", "800x600", "1024x768", "1280x720", "1280x960", "1920x1080"]
        )

        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 60)
        self.fps_spin.setValue(30)
        self.fps_spin.setSuffix(" fps")

        props_form.addRow("Resolution:", self.resolution_combo)
        props_form.addRow("Frame Rate:", self.fps_spin)
        layout.addWidget(props_group)

        # Real-time Processing Group
        realtime_group = QGroupBox("Real-time Processing")
        realtime_form = QFormLayout(realtime_group)

        self.frame_skip_spin = QSpinBox()
        self.frame_skip_spin.setRange(1, 10)
        self.frame_skip_spin.setValue(1)
        self.frame_skip_spin.setToolTip(
            "Process every Nth frame (higher = faster but less smooth)"
        )

        self.auto_resize_check = QCheckBox("Auto-resize for performance")
        self.auto_resize_check.setChecked(True)
        self.auto_resize_check.setToolTip(
            "Automatically resize large frames for better performance"
        )

        realtime_form.addRow("Frame Skip:", self.frame_skip_spin)
        realtime_form.addRow("", self.auto_resize_check)
        layout.addWidget(realtime_group)

        layout.addStretch()
        self.tab_widget.addTab(camera_tab, "Camera")

    def create_processing_tab(self):
        """Create processing settings tab"""
        processing_tab = QWidget()
        layout = QVBoxLayout(processing_tab)
        layout.setSpacing(15)

        # Model Settings Group
        model_group = QGroupBox("Model Settings")
        model_form = QFormLayout(model_group)

        self.device_combo = QComboBox()
        self.device_combo.addItems(["Auto-detect", "CPU", "GPU (CUDA)"])

        self.model_input_size_combo = QComboBox()
        self.model_input_size_combo.addItems(["256x256", "512x512", "1024x1024"])

        model_form.addRow("Processing Device:", self.device_combo)
        model_form.addRow("Model Input Size:", self.model_input_size_combo)
        layout.addWidget(model_group)

        # Performance Settings Group
        perf_group = QGroupBox("Performance Settings")
        perf_form = QFormLayout(perf_group)

        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 16)
        self.batch_size_spin.setValue(1)
        self.batch_size_spin.setToolTip("Batch size for processing multiple images")

        self.memory_fraction_slider = QSlider(Qt.Horizontal)
        self.memory_fraction_slider.setRange(10, 90)
        self.memory_fraction_slider.setValue(80)
        self.memory_fraction_label = QLabel("80%")

        memory_layout = QHBoxLayout()
        memory_layout.addWidget(self.memory_fraction_slider)
        memory_layout.addWidget(self.memory_fraction_label)

        self.clear_cache_check = QCheckBox("Clear GPU cache after processing")
        self.clear_cache_check.setChecked(True)

        perf_form.addRow("Batch Size:", self.batch_size_spin)
        perf_form.addRow("Max GPU Memory:", memory_layout)
        perf_form.addRow("", self.clear_cache_check)
        layout.addWidget(perf_group)

        # Image Enhancement Group
        enhance_group = QGroupBox("Image Enhancement")
        enhance_form = QFormLayout(enhance_group)

        self.brightness_slider = self.create_adjustment_slider(0.5, 2.0, 1.0)
        self.contrast_slider = self.create_adjustment_slider(0.5, 2.0, 1.0)
        self.saturation_slider = self.create_adjustment_slider(0.0, 2.0, 1.0)

        enhance_form.addRow("Brightness:", self.brightness_slider["widget"])
        enhance_form.addRow("Contrast:", self.contrast_slider["widget"])
        enhance_form.addRow("Saturation:", self.saturation_slider["widget"])
        layout.addWidget(enhance_group)

        layout.addStretch()
        self.tab_widget.addTab(processing_tab, "Processing")

    def create_output_tab(self):
        """Create output settings tab"""
        output_tab = QWidget()
        layout = QVBoxLayout(output_tab)
        layout.setSpacing(15)

        # Output Directory Group
        dir_group = QGroupBox("Output Directory")
        dir_layout = QVBoxLayout(dir_group)

        dir_selection_layout = QHBoxLayout()
        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setPlaceholderText("Select output directory...")
        self.browse_dir_btn = QPushButton("Browse")

        dir_selection_layout.addWidget(self.output_dir_edit)
        dir_selection_layout.addWidget(self.browse_dir_btn)
        dir_layout.addLayout(dir_selection_layout)
        layout.addWidget(dir_group)

        # File Format Group
        format_group = QGroupBox("File Format")
        format_form = QFormLayout(format_group)

        self.image_format_combo = QComboBox()
        self.image_format_combo.addItems(["PNG", "JPEG", "BMP", "TIFF"])

        self.jpeg_quality_spin = QSpinBox()
        self.jpeg_quality_spin.setRange(1, 100)
        self.jpeg_quality_spin.setValue(95)
        self.jpeg_quality_spin.setSuffix("%")
        self.jpeg_quality_spin.setEnabled(False)  # Enable when JPEG is selected

        format_form.addRow("Image Format:", self.image_format_combo)
        format_form.addRow("JPEG Quality:", self.jpeg_quality_spin)
        layout.addWidget(format_group)

        # Naming Convention Group
        naming_group = QGroupBox("File Naming")
        naming_form = QFormLayout(naming_group)

        self.filename_prefix_edit = QLineEdit()
        self.filename_prefix_edit.setText("colorized")
        self.filename_prefix_edit.setPlaceholderText("File prefix...")

        self.add_timestamp_check = QCheckBox("Add timestamp to filename")
        self.add_timestamp_check.setChecked(True)

        self.preserve_original_check = QCheckBox("Preserve original filename")
        self.preserve_original_check.setChecked(False)

        naming_form.addRow("Filename Prefix:", self.filename_prefix_edit)
        naming_form.addRow("", self.add_timestamp_check)
        naming_form.addRow("", self.preserve_original_check)
        layout.addWidget(naming_group)

        # Auto-save Group
        autosave_group = QGroupBox("Auto-save")
        autosave_form = QFormLayout(autosave_group)

        self.autosave_realtime_check = QCheckBox("Auto-save camera snapshots")
        self.autosave_realtime_check.setChecked(False)

        self.autosave_interval_spin = QSpinBox()
        self.autosave_interval_spin.setRange(1, 60)
        self.autosave_interval_spin.setValue(5)
        self.autosave_interval_spin.setSuffix(" seconds")
        self.autosave_interval_spin.setEnabled(False)

        autosave_form.addRow("", self.autosave_realtime_check)
        autosave_form.addRow("Save Interval:", self.autosave_interval_spin)
        layout.addWidget(autosave_group)

        layout.addStretch()
        self.tab_widget.addTab(output_tab, "Output")

    def create_interface_tab(self):
        """Create interface settings tab"""
        interface_tab = QWidget()
        layout = QVBoxLayout(interface_tab)
        layout.setSpacing(15)

        # Display Settings Group
        display_group = QGroupBox("Display Settings")
        display_form = QFormLayout(display_group)

        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Dark", "Light"])

        self.show_fps_check = QCheckBox("Show FPS counter")
        self.show_fps_check.setChecked(True)

        self.show_processing_time_check = QCheckBox("Show processing time")
        self.show_processing_time_check.setChecked(True)

        self.show_memory_usage_check = QCheckBox("Show memory usage")
        self.show_memory_usage_check.setChecked(False)

        display_form.addRow("Theme:", self.theme_combo)
        display_form.addRow("", self.show_fps_check)
        display_form.addRow("", self.show_processing_time_check)
        display_form.addRow("", self.show_memory_usage_check)
        layout.addWidget(display_group)

        # Window Settings Group
        window_group = QGroupBox("Window Settings")
        window_form = QFormLayout(window_group)

        self.remember_size_check = QCheckBox("Remember window size and position")
        self.remember_size_check.setChecked(True)

        self.start_maximized_check = QCheckBox("Start maximized")
        self.start_maximized_check.setChecked(False)

        self.confirm_exit_check = QCheckBox("Confirm before exit")
        self.confirm_exit_check.setChecked(True)

        window_form.addRow("", self.remember_size_check)
        window_form.addRow("", self.start_maximized_check)
        window_form.addRow("", self.confirm_exit_check)
        layout.addWidget(window_group)

        # Keyboard Shortcuts Group
        shortcuts_group = QGroupBox("Keyboard Shortcuts")
        shortcuts_layout = QVBoxLayout(shortcuts_group)

        shortcuts_info = QLabel("""
        <b>Keyboard Shortcuts:</b><br>
        Ctrl+O: Open Image<br>
        Ctrl+S: Save Image<br>
        Ctrl+Shift+S: Save As<br>
        Space: Start/Stop Camera<br>
        F11: Toggle Full Screen<br>
        Ctrl+R: Reset View<br>
        Ctrl+Q: Quit Application
        """)
        shortcuts_info.setWordWrap(True)
        shortcuts_layout.addWidget(shortcuts_info)
        layout.addWidget(shortcuts_group)

        layout.addStretch()
        self.tab_widget.addTab(interface_tab, "Interface")

    def create_adjustment_slider(self, min_val, max_val, default_val):
        """Create adjustment slider with label"""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        slider = QSlider(Qt.Horizontal)
        slider.setRange(int(min_val * 100), int(max_val * 100))
        slider.setValue(int(default_val * 100))

        label = QLabel(f"{default_val:.2f}")
        label.setMinimumWidth(40)

        layout.addWidget(slider)
        layout.addWidget(label)

        # Connect slider to label update
        slider.valueChanged.connect(lambda v: label.setText(f"{v / 100:.2f}"))

        return {"widget": widget, "slider": slider, "label": label}

    def setup_connections(self):
        """Setup signal connections"""
        # Button box
        self.button_box.accepted.connect(self.accept_settings)
        self.button_box.rejected.connect(self.reject)
        self.button_box.button(QDialogButtonBox.Apply).clicked.connect(
            self.apply_settings
        )

        # Camera tab
        self.refresh_cameras_btn.clicked.connect(self.refresh_camera_list)

        # Output tab
        self.browse_dir_btn.clicked.connect(self.browse_output_directory)
        self.image_format_combo.currentTextChanged.connect(self.on_format_changed)
        self.autosave_realtime_check.toggled.connect(
            self.autosave_interval_spin.setEnabled
        )

        # Processing tab
        self.memory_fraction_slider.valueChanged.connect(
            lambda v: self.memory_fraction_label.setText(f"{v}%")
        )

    def refresh_camera_list(self):
        """Refresh the list of available cameras"""
        try:
            import os
            import sys

            # Add paths for imports
            current_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(current_dir)
            sys.path.insert(0, os.path.join(parent_dir, "utils"))

            from camera import CameraCapture

            cameras = CameraCapture.get_available_cameras()

            self.camera_combo.clear()
            for camera_id in cameras:
                self.camera_combo.addItem(f"Camera {camera_id}", camera_id)

            if not cameras:
                self.camera_combo.addItem("No cameras found", -1)

        except Exception as e:
            logger.error(f"Error refreshing camera list: {e}")
            QMessageBox.warning(
                self, "Error", f"Failed to refresh camera list:\n{str(e)}"
            )

    def browse_output_directory(self):
        """Browse for output directory"""
        current_dir = self.output_dir_edit.text() or os.path.expanduser("~")
        selected_dir = QFileDialog.getExistingDirectory(
            self, "Select Output Directory", current_dir
        )

        if selected_dir:
            self.output_dir_edit.setText(selected_dir)

    def on_format_changed(self, format_name):
        """Handle image format change"""
        is_jpeg = format_name.upper() == "JPEG"
        self.jpeg_quality_spin.setEnabled(is_jpeg)

    def load_current_settings(self):
        """Load current settings into the UI"""
        # Camera settings
        camera_id = self.current_settings.get("camera_id", 0)
        resolution = self.current_settings.get("resolution", "640x480")
        fps = self.current_settings.get("fps", 30)
        frame_skip = self.current_settings.get("frame_skip", 1)
        auto_resize = self.current_settings.get("auto_resize", True)

        # Refresh camera list first
        self.refresh_camera_list()

        # Set camera
        for i in range(self.camera_combo.count()):
            if self.camera_combo.itemData(i) == camera_id:
                self.camera_combo.setCurrentIndex(i)
                break

        # Set other camera settings
        resolution_index = self.resolution_combo.findText(resolution)
        if resolution_index >= 0:
            self.resolution_combo.setCurrentIndex(resolution_index)

        self.fps_spin.setValue(fps)
        self.frame_skip_spin.setValue(frame_skip)
        self.auto_resize_check.setChecked(auto_resize)

        # Processing settings
        device = self.current_settings.get("device", "Auto-detect")
        model_size = self.current_settings.get("model_input_size", "256x256")
        batch_size = self.current_settings.get("batch_size", 1)
        memory_fraction = self.current_settings.get("memory_fraction", 80)
        clear_cache = self.current_settings.get("clear_cache", True)

        device_index = self.device_combo.findText(device)
        if device_index >= 0:
            self.device_combo.setCurrentIndex(device_index)

        size_index = self.model_input_size_combo.findText(model_size)
        if size_index >= 0:
            self.model_input_size_combo.setCurrentIndex(size_index)

        self.batch_size_spin.setValue(batch_size)
        self.memory_fraction_slider.setValue(memory_fraction)
        self.clear_cache_check.setChecked(clear_cache)

        # Enhancement settings
        brightness = self.current_settings.get("brightness", 1.0)
        contrast = self.current_settings.get("contrast", 1.0)
        saturation = self.current_settings.get("saturation", 1.0)

        self.brightness_slider["slider"].setValue(int(brightness * 100))
        self.contrast_slider["slider"].setValue(int(contrast * 100))
        self.saturation_slider["slider"].setValue(int(saturation * 100))

        # Output settings
        output_dir = self.current_settings.get("output_directory", "")
        image_format = self.current_settings.get("image_format", "PNG")
        jpeg_quality = self.current_settings.get("jpeg_quality", 95)
        filename_prefix = self.current_settings.get("filename_prefix", "colorized")
        add_timestamp = self.current_settings.get("add_timestamp", True)
        preserve_original = self.current_settings.get("preserve_original", False)
        autosave_realtime = self.current_settings.get("autosave_realtime", False)
        autosave_interval = self.current_settings.get("autosave_interval", 5)

        self.output_dir_edit.setText(output_dir)

        format_index = self.image_format_combo.findText(image_format)
        if format_index >= 0:
            self.image_format_combo.setCurrentIndex(format_index)

        self.jpeg_quality_spin.setValue(jpeg_quality)
        self.filename_prefix_edit.setText(filename_prefix)
        self.add_timestamp_check.setChecked(add_timestamp)
        self.preserve_original_check.setChecked(preserve_original)
        self.autosave_realtime_check.setChecked(autosave_realtime)
        self.autosave_interval_spin.setValue(autosave_interval)

        # Interface settings
        theme = self.current_settings.get("theme", "Dark")
        show_fps = self.current_settings.get("show_fps", True)
        show_processing_time = self.current_settings.get("show_processing_time", True)
        show_memory_usage = self.current_settings.get("show_memory_usage", False)
        remember_size = self.current_settings.get("remember_window_size", True)
        start_maximized = self.current_settings.get("start_maximized", False)
        confirm_exit = self.current_settings.get("confirm_exit", True)

        theme_index = self.theme_combo.findText(theme)
        if theme_index >= 0:
            self.theme_combo.setCurrentIndex(theme_index)

        self.show_fps_check.setChecked(show_fps)
        self.show_processing_time_check.setChecked(show_processing_time)
        self.show_memory_usage_check.setChecked(show_memory_usage)
        self.remember_size_check.setChecked(remember_size)
        self.start_maximized_check.setChecked(start_maximized)
        self.confirm_exit_check.setChecked(confirm_exit)

    def get_settings(self) -> Dict[str, Any]:
        """Get current settings from UI"""
        settings = {}

        # Camera settings
        camera_data = self.camera_combo.currentData()
        settings["camera_id"] = camera_data if camera_data is not None else 0
        settings["resolution"] = self.resolution_combo.currentText()
        settings["fps"] = self.fps_spin.value()
        settings["frame_skip"] = self.frame_skip_spin.value()
        settings["auto_resize"] = self.auto_resize_check.isChecked()

        # Processing settings
        settings["device"] = self.device_combo.currentText()
        settings["model_input_size"] = self.model_input_size_combo.currentText()
        settings["batch_size"] = self.batch_size_spin.value()
        settings["memory_fraction"] = self.memory_fraction_slider.value()
        settings["clear_cache"] = self.clear_cache_check.isChecked()

        # Enhancement settings
        settings["brightness"] = self.brightness_slider["slider"].value() / 100.0
        settings["contrast"] = self.contrast_slider["slider"].value() / 100.0
        settings["saturation"] = self.saturation_slider["slider"].value() / 100.0

        # Output settings
        settings["output_directory"] = self.output_dir_edit.text()
        settings["image_format"] = self.image_format_combo.currentText()
        settings["jpeg_quality"] = self.jpeg_quality_spin.value()
        settings["filename_prefix"] = self.filename_prefix_edit.text()
        settings["add_timestamp"] = self.add_timestamp_check.isChecked()
        settings["preserve_original"] = self.preserve_original_check.isChecked()
        settings["autosave_realtime"] = self.autosave_realtime_check.isChecked()
        settings["autosave_interval"] = self.autosave_interval_spin.value()

        # Interface settings
        settings["theme"] = self.theme_combo.currentText()
        settings["show_fps"] = self.show_fps_check.isChecked()
        settings["show_processing_time"] = self.show_processing_time_check.isChecked()
        settings["show_memory_usage"] = self.show_memory_usage_check.isChecked()
        settings["remember_window_size"] = self.remember_size_check.isChecked()
        settings["start_maximized"] = self.start_maximized_check.isChecked()
        settings["confirm_exit"] = self.confirm_exit_check.isChecked()

        return settings

    def apply_settings(self):
        """Apply current settings"""
        self.temp_settings = self.get_settings()
        self.settings_changed.emit(self.temp_settings)

    def accept_settings(self):
        """Accept and apply settings"""
        self.apply_settings()
        self.accept()

    def apply_dark_theme(self):
        """Apply dark theme to the dialog"""
        self.setStyleSheet("""
            QDialog {
                background-color: #2b2b2b;
                color: #ffffff;
            }
            QTabWidget::pane {
                border: 1px solid #555;
                background-color: #2b2b2b;
            }
            QTabWidget::tab-bar {
                alignment: center;
            }
            QTabBar::tab {
                background-color: #3c3c3c;
                color: #ffffff;
                padding: 8px 16px;
                margin-right: 2px;
                border: 1px solid #555;
                border-bottom: none;
            }
            QTabBar::tab:selected {
                background-color: #0078d4;
                border-color: #0078d4;
            }
            QTabBar::tab:hover {
                background-color: #4c4c4c;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #555;
                border-radius: 5px;
                margin-top: 1ex;
                padding-top: 15px;
                color: #ffffff;
                background-color: #3c3c3c;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: #ffffff;
            }
            QLabel {
                color: #ffffff;
                background-color: transparent;
            }
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
                background-color: #3c3c3c;
                border: 1px solid #555;
                border-radius: 3px;
                padding: 5px;
                color: #ffffff;
                min-height: 20px;
            }
            QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
                border-color: #0078d4;
            }
            QPushButton {
                background-color: #0078d4;
                border: 1px solid #0078d4;
                border-radius: 3px;
                padding: 8px 16px;
                color: #ffffff;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #106ebe;
            }
            QPushButton:pressed {
                background-color: #005a9e;
            }
            QCheckBox {
                color: #ffffff;
                spacing: 5px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
            }
            QCheckBox::indicator:unchecked {
                border: 2px solid #555;
                border-radius: 3px;
                background-color: #2b2b2b;
            }
            QCheckBox::indicator:checked {
                border: 2px solid #0078d4;
                border-radius: 3px;
                background-color: #0078d4;
            }
            QSlider::groove:horizontal {
                border: 1px solid #555;
                height: 8px;
                background: #2b2b2b;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #0078d4;
                border: 1px solid #0078d4;
                width: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
            QSlider::handle:horizontal:hover {
                background: #106ebe;
            }
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 20px;
                border-left: 1px solid #555;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 5px solid #ffffff;
            }
            QComboBox QAbstractItemView {
                background-color: #3c3c3c;
                border: 1px solid #555;
                selection-background-color: #0078d4;
                color: #ffffff;
            }
        """)
