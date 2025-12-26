"""
Main Application Window for AI Colorization Studio
Provides the complete UI with camera capture, image processing, and colorization features
"""

import json
import logging

# Add necessary paths to sys.path
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PyQt5.QtCore import QSettings, QSize, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QFont, QIcon, QKeySequence, QPixmap
from PyQt5.QtWidgets import (
    QAction,
    QApplication,
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenuBar,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QStatusBar,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, current_dir)  # ui directory
sys.path.insert(0, parent_dir)  # colorization_app directory

# Import our custom modules
from comparison_view import ComparisonView
from settings_dialog import SettingsDialog

# Change directory to parent to import other modules
os.chdir(parent_dir)
sys.path.insert(0, os.path.join(parent_dir, "model"))
sys.path.insert(0, os.path.join(parent_dir, "utils"))

from camera import CAMERA_MODE_IR, CAMERA_MODE_SYSTEM, CameraThread, FrameProcessor
from image_utils import BatchProcessor, ImageProcessor
from model_loader import InferenceEngine, ModelLoader

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    """Main application window"""

    def __init__(self):
        super().__init__()

        # Initialize core components
        self.model_loader = ModelLoader()
        self.image_processor = ImageProcessor()
        self.inference_engine = None
        self.camera_thread = None
        self.frame_processor = None

        # UI state
        self.current_image = None
        self.current_colorized = None
        self.is_camera_active = False
        self.settings = QSettings("AIColorizationStudio", "Settings")

        # Performance monitoring
        self.fps_counter = 0
        self.fps_timer = QTimer()
        self.processing_times = []

        # Load default settings
        self.load_default_settings()

        # Initialize UI
        self.init_ui()
        self.create_menus()
        self.create_toolbar()
        self.create_status_bar()
        self.setup_connections()
        self.apply_theme()

        # Restore window state
        self.restore_window_state()

        # Auto-load demo model if available
        self.auto_load_demo_model()

        logger.info("Main window initialized successfully")

    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("AI Colorization Studio")
        self.setMinimumSize(1000, 700)

        # Create central widget with splitter
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)

        # Create main splitter
        main_splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(main_splitter)

        # Left panel - Controls
        self.create_control_panel()
        main_splitter.addWidget(self.control_panel)

        # Right panel - Comparison view
        self.comparison_view = ComparisonView()
        main_splitter.addWidget(self.comparison_view)

        # Set splitter proportions (20% controls, 80% display for better comparison view)
        main_splitter.setSizes([200, 800])
        main_splitter.setCollapsible(0, False)
        main_splitter.setCollapsible(1, False)

    def create_control_panel(self):
        """Create the left control panel"""
        self.control_panel = QFrame()
        self.control_panel.setFrameStyle(QFrame.Box)
        self.control_panel.setMaximumWidth(250)
        self.control_panel.setMinimumWidth(220)

        layout = QVBoxLayout(self.control_panel)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(12)

        # Model section
        model_frame = self.create_section_frame("Model")
        model_layout = QVBoxLayout(model_frame)
        model_layout.setContentsMargins(8, 8, 8, 8)
        model_layout.setSpacing(6)

        # Add section title
        model_title = QLabel("Model")
        model_title.setStyleSheet(
            "font-weight: bold; font-size: 11px; color: #fff; padding: 2px;"
        )
        model_layout.addWidget(model_title)

        self.model_status_label = QLabel("No model loaded")
        self.model_status_label.setWordWrap(True)
        self.model_status_label.setStyleSheet(
            "color: #ff6b6b; font-weight: bold; font-size: 10px; padding: 2px;"
        )

        self.load_model_btn = QPushButton("Load Model")
        self.load_model_btn.setMinimumHeight(30)

        self.model_info_label = QLabel("Load a model to begin colorization")
        self.model_info_label.setWordWrap(True)
        self.model_info_label.setStyleSheet(
            "font-size: 9px; color: #aaa; padding: 2px;"
        )

        model_layout.addWidget(self.model_status_label)
        model_layout.addWidget(self.load_model_btn)
        model_layout.addWidget(self.model_info_label)
        layout.addWidget(model_frame)

        # Camera section
        camera_frame = self.create_section_frame("Camera")
        camera_layout = QVBoxLayout(camera_frame)
        camera_layout.setContentsMargins(8, 8, 8, 8)
        camera_layout.setSpacing(6)

        # Add section title
        camera_title = QLabel("Camera")
        camera_title.setStyleSheet(
            "font-weight: bold; font-size: 11px; color: #fff; padding: 2px;"
        )
        camera_layout.addWidget(camera_title)

        # Camera source selection dropdown
        source_layout = QHBoxLayout()
        source_label = QLabel("Source:")
        source_label.setStyleSheet("font-size: 10px; color: #ccc;")
        self.camera_source_combo = QComboBox()
        self.camera_source_combo.addItems(["System Camera", "IR Camera"])
        self.camera_source_combo.setCurrentIndex(0)  # Default to System Camera
        self.camera_source_combo.setMinimumHeight(24)
        source_layout.addWidget(source_label)
        source_layout.addWidget(self.camera_source_combo)
        camera_layout.addLayout(source_layout)

        camera_controls = QHBoxLayout()
        camera_controls.setSpacing(4)
        self.start_camera_btn = QPushButton("Start")
        self.start_camera_btn.setMinimumHeight(28)
        self.stop_camera_btn = QPushButton("Stop")
        self.stop_camera_btn.setMinimumHeight(28)
        self.stop_camera_btn.setEnabled(False)

        camera_controls.addWidget(self.start_camera_btn)
        camera_controls.addWidget(self.stop_camera_btn)

        self.snapshot_btn = QPushButton("Snapshot")
        self.snapshot_btn.setMinimumHeight(28)
        self.snapshot_btn.setEnabled(False)

        camera_layout.addLayout(camera_controls)
        camera_layout.addWidget(self.snapshot_btn)
        layout.addWidget(camera_frame)

        # Image section
        image_frame = self.create_section_frame("Image Processing")
        image_layout = QVBoxLayout(image_frame)
        image_layout.setContentsMargins(8, 8, 8, 8)
        image_layout.setSpacing(6)

        # Add section title
        image_title = QLabel("Image Processing")
        image_title.setStyleSheet(
            "font-weight: bold; font-size: 11px; color: #fff; padding: 2px;"
        )
        image_layout.addWidget(image_title)

        self.load_image_btn = QPushButton("Load Image")
        self.load_image_btn.setMinimumHeight(30)

        self.process_image_btn = QPushButton("Colorize")
        self.process_image_btn.setMinimumHeight(30)
        self.process_image_btn.setEnabled(False)

        self.save_result_btn = QPushButton("Save Result")
        self.save_result_btn.setMinimumHeight(30)
        self.save_result_btn.setEnabled(False)

        image_layout.addWidget(self.load_image_btn)
        image_layout.addWidget(self.process_image_btn)
        image_layout.addWidget(self.save_result_btn)
        layout.addWidget(image_frame)

        # Batch processing section
        batch_frame = self.create_section_frame("Batch Processing")
        batch_layout = QVBoxLayout(batch_frame)
        batch_layout.setContentsMargins(8, 8, 8, 8)
        batch_layout.setSpacing(6)

        # Add section title
        batch_title = QLabel("Batch Processing")
        batch_title.setStyleSheet(
            "font-weight: bold; font-size: 11px; color: #fff; padding: 2px;"
        )
        batch_layout.addWidget(batch_title)

        self.batch_process_btn = QPushButton("Process Folder")
        self.batch_process_btn.setMinimumHeight(30)
        self.batch_process_btn.setEnabled(False)

        self.batch_progress = QProgressBar()
        self.batch_progress.setVisible(False)
        self.batch_progress.setMinimumHeight(20)

        batch_layout.addWidget(self.batch_process_btn)
        batch_layout.addWidget(self.batch_progress)
        layout.addWidget(batch_frame)

        # Performance info section
        perf_frame = self.create_section_frame("Performance")
        perf_layout = QVBoxLayout(perf_frame)
        perf_layout.setContentsMargins(8, 8, 8, 8)
        perf_layout.setSpacing(4)

        # Add section title
        perf_title = QLabel("Performance")
        perf_title.setStyleSheet(
            "font-weight: bold; font-size: 11px; color: #fff; padding: 2px;"
        )
        perf_layout.addWidget(perf_title)

        self.fps_label = QLabel("FPS: --")
        self.fps_label.setStyleSheet("font-size: 10px; color: #ccc; padding: 1px;")
        self.processing_time_label = QLabel("Processing: -- ms")
        self.processing_time_label.setStyleSheet(
            "font-size: 10px; color: #ccc; padding: 1px;"
        )
        self.device_label = QLabel("Device: --")
        self.device_label.setStyleSheet("font-size: 10px; color: #ccc; padding: 1px;")

        perf_layout.addWidget(self.fps_label)
        perf_layout.addWidget(self.processing_time_label)
        perf_layout.addWidget(self.device_label)
        layout.addWidget(perf_frame)

        # Settings button
        self.settings_btn = QPushButton("Settings")
        self.settings_btn.setMinimumHeight(32)
        layout.addWidget(self.settings_btn)

        # Add stretch at the bottom
        layout.addStretch()

    def create_section_frame(self, title: str) -> QFrame:
        """Create a section frame with title"""
        frame = QFrame()
        frame.setFrameStyle(QFrame.Box)

        # Don't create layout here - let the caller create it
        # Just add the title label if needed
        return frame

    def create_menus(self):
        """Create application menus"""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("File")

        # Open image action
        open_action = QAction("Open Image...", self)
        open_action.setShortcut(QKeySequence.Open)
        open_action.setIcon(self.style().standardIcon(self.style().SP_DialogOpenButton))
        open_action.triggered.connect(self.load_image)
        file_menu.addAction(open_action)

        # Save result action
        save_action = QAction("Save Result...", self)
        save_action.setShortcut(QKeySequence.Save)
        save_action.setIcon(self.style().standardIcon(self.style().SP_DialogSaveButton))
        save_action.triggered.connect(self.save_result)
        file_menu.addAction(save_action)

        file_menu.addSeparator()

        # Exit action
        exit_action = QAction("Exit", self)
        exit_action.setShortcut(QKeySequence("Ctrl+Q"))
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Model menu
        model_menu = menubar.addMenu("Model")

        # Load model action
        load_model_action = QAction("Load Model...", self)
        load_model_action.setShortcut(QKeySequence("Ctrl+M"))
        load_model_action.triggered.connect(self.load_model)
        model_menu.addAction(load_model_action)

        # View menu
        view_menu = menubar.addMenu("View")

        # Toggle fullscreen
        fullscreen_action = QAction("Toggle Fullscreen", self)
        fullscreen_action.setShortcut(QKeySequence("F11"))
        fullscreen_action.triggered.connect(self.toggle_fullscreen)
        view_menu.addAction(fullscreen_action)

        # Help menu
        help_menu = menubar.addMenu("Help")

        # About action
        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    def create_toolbar(self):
        """Create application toolbar"""
        toolbar = QToolBar("Main Toolbar")
        self.addToolBar(toolbar)

        # Load model
        toolbar.addAction(
            self.style().standardIcon(self.style().SP_DialogOpenButton),
            "Load Model",
            self.load_model,
        )

        toolbar.addSeparator()

        # Load image
        toolbar.addAction(
            self.style().standardIcon(self.style().SP_FileDialogDetailedView),
            "Load Image",
            self.load_image,
        )

        # Process image
        toolbar.addAction(
            self.style().standardIcon(self.style().SP_ComputerIcon),
            "Colorize",
            self.process_current_image,
        )

        # Save result
        toolbar.addAction(
            self.style().standardIcon(self.style().SP_DialogSaveButton),
            "Save Result",
            self.save_result,
        )

        toolbar.addSeparator()

        # Camera controls
        toolbar.addAction(
            self.style().standardIcon(self.style().SP_MediaPlay),
            "Start Camera",
            self.start_camera,
        )

        toolbar.addAction(
            self.style().standardIcon(self.style().SP_MediaStop),
            "Stop Camera",
            self.stop_camera,
        )

    def create_status_bar(self):
        """Create status bar"""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        # Ready message
        self.status_bar.showMessage("Ready", 3000)

    def setup_connections(self):
        """Setup signal connections"""
        # Control panel buttons
        self.load_model_btn.clicked.connect(self.load_model)
        self.start_camera_btn.clicked.connect(self.start_camera)
        self.stop_camera_btn.clicked.connect(self.stop_camera)
        self.snapshot_btn.clicked.connect(self.take_snapshot)
        self.load_image_btn.clicked.connect(self.load_image)
        self.process_image_btn.clicked.connect(self.process_current_image)
        self.save_result_btn.clicked.connect(self.save_result)
        self.batch_process_btn.clicked.connect(self.batch_process)
        self.settings_btn.clicked.connect(self.open_settings)

        # FPS timer
        self.fps_timer.timeout.connect(self.update_fps_display)
        self.fps_timer.start(1000)  # Update every second

    def load_default_settings(self):
        """Load default application settings"""
        self.app_settings = {
            "camera_id": 0,
            "resolution": "640x480",
            "fps": 30,
            "frame_skip": 1,
            "auto_resize": True,
            "device": "Auto-detect",
            "model_input_size": "256x256",
            "batch_size": 1,
            "memory_fraction": 80,
            "clear_cache": True,
            "brightness": 1.0,
            "contrast": 1.0,
            "saturation": 1.0,
            "output_directory": str(Path.home() / "AIColorization"),
            "image_format": "PNG",
            "jpeg_quality": 95,
            "filename_prefix": "colorized",
            "add_timestamp": True,
            "preserve_original": False,
            "autosave_realtime": False,
            "autosave_interval": 5,
            "theme": "Dark",
            "show_fps": True,
            "show_processing_time": True,
            "show_memory_usage": False,
            "remember_window_size": True,
            "start_maximized": False,
            "confirm_exit": True,
        }

    def auto_load_demo_model(self):
        """Auto-load demo model if available"""
        demo_model_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "demo_model.pth"
        )
        demo_model_path = os.path.normpath(demo_model_path)

        if os.path.exists(demo_model_path):
            logger.info(f"Found demo model at: {demo_model_path}")
            self.load_model_from_path(demo_model_path)
        else:
            logger.info("No demo model found - user will need to load a model manually")

    def load_model(self):
        """Load a trained model"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Model", "", "PyTorch Models (*.pth *.pt);;All Files (*)"
        )

        if file_path:
            self.load_model_from_path(file_path)

    def load_model_from_path(self, file_path):
        """Load model from specific path"""
        self.status_bar.showMessage("Loading model...")

        # Validate checkpoint first
        is_valid, message = self.model_loader.validate_checkpoint(file_path)

        if not is_valid:
            # Check if it's a weights_only issue that can be fixed
            if (
                "weights_only" in message
                or "WeightsUnpickler" in message
                or "GLOBAL" in message
            ):
                reply = QMessageBox.question(
                    self,
                    "Model Security Warning",
                    f"The model file contains unsafe operations:\n\n{message}\n\n"
                    "This is common with older PyTorch models. Would you like to:\n"
                    "• Fix the model automatically (recommended)\n"
                    "• Load anyway (not recommended)\n"
                    "• Cancel loading\n\n"
                    "Click 'Yes' to fix the model, 'No' to load anyway, 'Cancel' to abort.",
                    QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
                    QMessageBox.Yes,
                )

                if reply == QMessageBox.Yes:
                    # Try to fix the model
                    if self.fix_model_file(file_path):
                        return  # Model was fixed and loaded
                    else:
                        self.status_bar.showMessage("Model fixing failed", 3000)
                        return
                elif reply == QMessageBox.No:
                    # User wants to load anyway - continue with unsafe loading
                    pass
                else:
                    # User cancelled
                    self.status_bar.showMessage("Model loading cancelled", 3000)
                    return
            else:
                QMessageBox.warning(
                    self, "Invalid Model", f"Cannot load model:\n{message}"
                )
                self.status_bar.showMessage("Model loading failed", 3000)
                return

        # Load the model
        success = self.model_loader.load_checkpoint(file_path)

        if success:
            # Create inference engine
            self.inference_engine = InferenceEngine(self.model_loader)

            # Update UI
            is_demo = "demo_model" in os.path.basename(file_path).lower()
            if is_demo:
                self.model_status_label.setText("Demo model loaded")
                self.model_status_label.setStyleSheet(
                    "color: #ffd43b; font-weight: bold;"
                )
            else:
                self.model_status_label.setText("Model loaded successfully")
                self.model_status_label.setStyleSheet(
                    "color: #51cf66; font-weight: bold;"
                )

            # Update model info
            model_info = self.model_loader.get_model_info()
            info_text = f"Epoch: {model_info.get('epoch', 'N/A')}\n"
            info_text += f"Parameters: {model_info.get('total_params', 0):,}\n"
            info_text += f"Device: {model_info.get('device', 'N/A')}"
            if is_demo:
                info_text += "\n(Demo Model - Load your own for better results)"
            self.model_info_label.setText(info_text)

            # Update device label
            self.device_label.setText(f"Device: {model_info.get('device', 'N/A')}")

            # Enable processing buttons
            self.process_image_btn.setEnabled(True)
            self.batch_process_btn.setEnabled(True)

            self.status_bar.showMessage("Model loaded successfully", 3000)
            logger.info(f"Model loaded: {file_path}")
        else:
            QMessageBox.critical(self, "Error", "Failed to load model")
            self.status_bar.showMessage("Model loading failed", 3000)

    def start_camera(self):
        """Start camera capture"""
        if self.is_camera_active:
            return

        try:
            # Parse resolution
            resolution = self.app_settings.get("resolution", "640x480")
            width, height = map(int, resolution.split("x"))

            # Determine camera mode from dropdown selection
            camera_source = self.camera_source_combo.currentText()
            if camera_source == "IR Camera":
                camera_mode = CAMERA_MODE_IR
            else:
                camera_mode = CAMERA_MODE_SYSTEM

            # Create camera thread with selected mode
            camera_id = self.app_settings.get("camera_id", 0)
            self.camera_thread = CameraThread(camera_id, camera_mode)

            # Create frame processor if model is loaded
            if self.inference_engine:
                self.frame_processor = FrameProcessor(
                    self.inference_engine, self.image_processor
                )
                self.camera_thread.set_processing_function(self.process_camera_frame)

            # Set frame skip
            frame_skip = self.app_settings.get("frame_skip", 1)
            self.camera_thread.set_frame_skip(frame_skip)

            # Connect signals
            self.camera_thread.frame_captured.connect(self.on_frame_captured)
            self.camera_thread.frame_processed.connect(self.on_frame_processed)
            self.camera_thread.fps_updated.connect(self.on_fps_updated)
            self.camera_thread.error_occurred.connect(self.on_camera_error)

            # Start camera
            fps = self.app_settings.get("fps", 30)
            success = self.camera_thread.start_capture(width, height, fps)

            if success:
                self.is_camera_active = True
                self.start_camera_btn.setEnabled(False)
                self.stop_camera_btn.setEnabled(True)
                self.snapshot_btn.setEnabled(True)
                self.camera_source_combo.setEnabled(
                    False
                )  # Disable switching while active

                # Set comparison view to side-by-side mode
                self.comparison_view.set_mode(ComparisonView.SIDE_BY_SIDE)

                self.status_bar.showMessage(f"{camera_source} started", 3000)
                logger.info(f"{camera_source} started successfully")
            else:
                QMessageBox.critical(self, "Error", "Failed to start camera")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Camera error: {str(e)}")
            logger.error(f"Camera start error: {e}")

    def stop_camera(self):
        """Stop camera capture"""
        if not self.is_camera_active or not self.camera_thread:
            return

        try:
            self.camera_thread.stop_capture()
            self.camera_thread = None
            self.frame_processor = None

            self.is_camera_active = False
            self.start_camera_btn.setEnabled(True)
            self.stop_camera_btn.setEnabled(False)
            self.snapshot_btn.setEnabled(False)
            self.camera_source_combo.setEnabled(True)  # Re-enable camera selection

            # Clear displays
            self.comparison_view.clear_images()

            self.status_bar.showMessage("Camera stopped", 3000)
            logger.info("Camera stopped")

        except Exception as e:
            logger.error(f"Error stopping camera: {e}")

    def process_camera_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Process camera frame for colorization"""
        if self.frame_processor:
            return self.frame_processor.process_frame(frame)
        return None

    def on_frame_captured(self, frame: np.ndarray):
        """Handle captured frame"""
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Convert to grayscale for display
            gray_frame = self.image_processor.rgb_to_grayscale(rgb_frame)

            # Update before image
            self.comparison_view.set_before_image(gray_frame)

            # Store current frame for processing
            self.current_image = rgb_frame

        except Exception as e:
            logger.error(f"Error handling captured frame: {e}")

    def on_frame_processed(self, processed_frame: np.ndarray):
        """Handle processed frame"""
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)

            # Update after image
            self.comparison_view.set_after_image(rgb_frame)

            # Update processing time
            if self.frame_processor:
                processing_time = self.frame_processor.get_processing_time()
                self.processing_times.append(processing_time)

                # Keep only last 10 measurements
                if len(self.processing_times) > 10:
                    self.processing_times.pop(0)

                avg_time = sum(self.processing_times) / len(self.processing_times)
                self.processing_time_label.setText(f"Processing: {avg_time:.1f} ms")

        except Exception as e:
            logger.error(f"Error handling processed frame: {e}")

    def on_fps_updated(self, fps: float):
        """Handle FPS update"""
        self.fps_label.setText(f"FPS: {fps:.1f}")

    def on_camera_error(self, error_message: str):
        """Handle camera error"""
        QMessageBox.critical(self, "Camera Error", error_message)
        self.stop_camera()

    def take_snapshot(self):
        """Take a snapshot of current camera feed"""
        if not self.is_camera_active or not self.current_colorized:
            return

        try:
            # Get output directory
            output_dir = self.app_settings.get(
                "output_directory", str(Path.home() / "AIColorization")
            )
            os.makedirs(output_dir, exist_ok=True)

            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"snapshot_{timestamp}.png"
            output_path = os.path.join(output_dir, filename)

            # Save current colorized image or current camera frame
            image_to_save = None
            if self.current_colorized is not None:
                image_to_save = self.current_colorized
            elif hasattr(self, "current_image") and self.current_image is not None:
                # If no colorized version, save the original with simple colorization
                image_to_save = self.simple_colorize_image(self.current_image)

            if image_to_save is not None and self.image_processor.save_image(
                image_to_save, output_path
            ):
                self.status_bar.showMessage(f"Snapshot saved: {filename}", 3000)
                logger.info(f"Snapshot saved: {output_path}")
            else:
                QMessageBox.warning(self, "Error", "Failed to save snapshot")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Snapshot error: {str(e)}")
            logger.error(f"Snapshot error: {e}")

    def load_image(self):
        """Load an image for processing"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tiff);;All Files (*)",
        )

        if file_path:
            try:
                # Load image
                image = self.image_processor.load_image(file_path)

                if image is not None:
                    self.current_image = image

                    # Convert to grayscale for display
                    gray_image = self.image_processor.rgb_to_grayscale(image)

                    # Update display - set before image (grayscale)
                    self.comparison_view.set_before_image(gray_image)
                    self.comparison_view.set_after_image(None)  # Clear only after image

                    # Reset save button
                    self.save_result_btn.setEnabled(False)

                    # Clear current colorized image
                    self.current_colorized = None

                    # Set to side-by-side mode
                    self.comparison_view.set_mode(ComparisonView.SIDE_BY_SIDE)

                    self.status_bar.showMessage(
                        f"Image loaded: {os.path.basename(file_path)}", 3000
                    )
                    logger.info(f"Image loaded: {file_path}")
                else:
                    QMessageBox.warning(self, "Error", "Failed to load image")

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Image loading error: {str(e)}")
                logger.error(f"Image loading error: {e}")

    def process_current_image(self):
        """Process the current image"""
        if self.current_image is None:
            QMessageBox.warning(self, "Warning", "Please load an image first")
            return

        if self.inference_engine is None:
            QMessageBox.warning(self, "Warning", "Please load a model first")
            return

        try:
            self.status_bar.showMessage("Processing image...")
            QApplication.processEvents()

            # Preprocess image
            input_tensor = self.image_processor.preprocess_for_model(self.current_image)

            # Colorize
            start_time = time.time()
            output_tensor = self.inference_engine.colorize(input_tensor)
            processing_time = (time.time() - start_time) * 1000

            if output_tensor is not None:
                # Postprocess
                colorized = self.image_processor.postprocess_from_model(output_tensor)

                # Store result
                self.current_colorized = colorized

                # Update display - show both original (grayscale) and colorized
                gray_image = self.image_processor.rgb_to_grayscale(self.current_image)
                self.comparison_view.set_before_image(gray_image)
                self.comparison_view.set_after_image(colorized)

                # Enable save button
                self.save_result_btn.setEnabled(True)

                # Update processing time
                self.processing_time_label.setText(
                    f"Processing: {processing_time:.1f} ms"
                )

                self.status_bar.showMessage("Image processed successfully", 3000)
                logger.info(f"Image processed in {processing_time:.1f} ms")
            else:
                QMessageBox.critical(self, "Error", "Image processing failed")
                self.status_bar.showMessage("Processing failed", 3000)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Processing error: {str(e)}")
            logger.error(f"Processing error: {e}")
            self.status_bar.showMessage("Processing failed", 3000)

    def save_result(self):
        """Save the colorized result"""
        if self.current_colorized is None:
            QMessageBox.warning(self, "Warning", "No result to save")
            return

        try:
            # Get default filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            default_filename = f"colorized_{timestamp}.png"

            # Open save dialog
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save Result",
                default_filename,
                "PNG (*.png);;JPEG (*.jpg);;BMP (*.bmp);;TIFF (*.tiff);;All Files (*)",
            )

            if file_path:
                # Get quality setting
                quality = self.app_settings.get("jpeg_quality", 95)

                # Save image
                if self.image_processor.save_image(
                    self.current_colorized, file_path, quality
                ):
                    self.status_bar.showMessage(
                        f"Image saved: {os.path.basename(file_path)}", 3000
                    )
                    logger.info(f"Image saved: {file_path}")
                else:
                    QMessageBox.warning(self, "Error", "Failed to save image")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Save error: {str(e)}")
            logger.error(f"Save error: {e}")

    def batch_process(self):
        """Process a folder of images"""
        if self.inference_engine is None:
            QMessageBox.warning(self, "Warning", "Please load a model first")
            return

        # Select input folder
        input_folder = QFileDialog.getExistingDirectory(self, "Select Input Folder")

        if not input_folder:
            return

        # Select output folder
        output_folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")

        if not output_folder:
            return

        try:
            # Create batch processor
            batch_processor = BatchProcessor(self.image_processor)

            # Show progress bar
            self.batch_progress.setVisible(True)
            self.batch_progress.setValue(0)

            # Process folder
            def update_progress(progress, message):
                self.batch_progress.setValue(int(progress))
                self.status_bar.showMessage(message)
                QApplication.processEvents()

            results = batch_processor.process_folder(
                input_folder, output_folder, self.inference_engine, update_progress
            )

            # Hide progress bar
            self.batch_progress.setVisible(False)

            # Show results
            if results["success"]:
                message = f"Batch processing completed!\n"
                message += f"Processed: {results['processed']} images\n"
                message += f"Failed: {results['failed']} images"

                QMessageBox.information(self, "Batch Processing", message)
                self.status_bar.showMessage("Batch processing completed", 3000)
            else:
                QMessageBox.critical(
                    self,
                    "Error",
                    f"Batch processing failed: {results.get('error', 'Unknown error')}",
                )

        except Exception as e:
            self.batch_progress.setVisible(False)
            QMessageBox.critical(self, "Error", f"Batch processing error: {str(e)}")
            logger.error(f"Batch processing error: {e}")

    def open_settings(self):
        """Open settings dialog"""
        dialog = SettingsDialog(self, self.app_settings)

        if dialog.exec_() == QDialog.Accepted:
            # Update settings
            new_settings = dialog.get_settings()
            self.app_settings.update(new_settings)

            # Apply new settings
            self.apply_settings()

            logger.info("Settings updated")

    def apply_settings(self):
        """Apply updated settings"""
        try:
            # Update device label if changed
            device_setting = self.app_settings.get("device", "Auto-detect")
            if device_setting != "Auto-detect":
                if self.model_loader.is_model_loaded():
                    if device_setting == "CPU":
                        new_device = torch.device("cpu")
                    else:  # GPU
                        new_device = torch.device(
                            "cuda" if torch.cuda.is_available() else "cpu"
                        )

                    self.model_loader.set_device(new_device)
                    self.device_label.setText(f"Device: {new_device}")

            # Update output directory
            output_dir = self.app_settings.get("output_directory")
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

            # Apply theme
            theme = self.app_settings.get("theme", "Dark")
            if theme == "Dark":
                self.apply_theme()

        except Exception as e:
            logger.error(f"Error applying settings: {e}")

    def update_fps_display(self):
        """Update FPS display"""
        if not self.app_settings.get("show_fps", True):
            self.fps_label.hide()
        else:
            self.fps_label.show()

        if not self.app_settings.get("show_processing_time", True):
            self.processing_time_label.hide()
        else:
            self.processing_time_label.show()

    def toggle_fullscreen(self):
        """Toggle fullscreen mode"""
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def show_about(self):
        """Show about dialog"""
        about_text = """
        <h2>AI Colorization Studio</h2>
        <p>A professional desktop application for real-time grayscale to RGB image colorization.</p>
        <p><b>Features:</b></p>
        <ul>
        <li>Real-time camera colorization</li>
        <li>Image file processing</li>
        <li>Batch processing</li>
        <li>Enhanced Pix2Pix model support</li>
        <li>Multiple comparison modes</li>
        </ul>
        <p><b>Powered by:</b> PyTorch, PyQt5, OpenCV</p>
        <p><b>Version:</b> 1.0.0</p>
        """

        QMessageBox.about(self, "About AI Colorization Studio", about_text)

    def apply_theme(self):
        """Apply dark theme to the application"""
        dark_stylesheet = """
        QMainWindow {
            background-color: #2b2b2b;
            color: #ffffff;
        }
        QWidget {
            background-color: #2b2b2b;
            color: #ffffff;
            font-family: 'Segoe UI', Arial, sans-serif;
            font-size: 10pt;
        }
        QFrame {
            background-color: #3c3c3c;
            border: 1px solid #555;
            border-radius: 6px;
            margin: 2px;
        }
        QPushButton {
            background-color: #0078d4;
            border: 1px solid #0078d4;
            border-radius: 4px;
            padding: 6px 12px;
            color: #ffffff;
            font-weight: bold;
            font-size: 10pt;
            min-height: 24px;
        }
        QPushButton:hover {
            background-color: #106ebe;
        }
        QPushButton:pressed {
            background-color: #005a9e;
        }
        QPushButton:disabled {
            background-color: #555;
            color: #888;
            border-color: #555;
        }
        QLabel {
            color: #ffffff;
            background-color: transparent;
        }
        QMenuBar {
            background-color: #3c3c3c;
            color: #ffffff;
            border-bottom: 1px solid #555;
        }
        QMenuBar::item {
            background-color: transparent;
            padding: 8px 12px;
        }
        QMenuBar::item:selected {
            background-color: #0078d4;
        }
        QMenu {
            background-color: #3c3c3c;
            border: 1px solid #555;
            color: #ffffff;
        }
        QMenu::item {
            padding: 8px 25px;
        }
        QMenu::item:selected {
            background-color: #0078d4;
        }
        QToolBar {
            background-color: #3c3c3c;
            border: 1px solid #555;
            spacing: 5px;
            color: #ffffff;
        }
        QToolBar QToolButton {
            background-color: transparent;
            border: 1px solid transparent;
            border-radius: 3px;
            padding: 5px;
            color: #ffffff;
        }
        QToolBar QToolButton:hover {
            background-color: #0078d4;
            border-color: #0078d4;
        }
        QStatusBar {
            background-color: #3c3c3c;
            color: #ffffff;
            border-top: 1px solid #555;
        }
        QProgressBar {
            border: 1px solid #555;
            border-radius: 3px;
            text-align: center;
            color: #ffffff;
            background-color: #2b2b2b;
        }
        QProgressBar::chunk {
            background-color: #0078d4;
            border-radius: 2px;
        }
        QSplitter::handle {
            background-color: #555;
        }
        QSplitter::handle:horizontal {
            width: 3px;
        }
        QSplitter::handle:vertical {
            height: 3px;
        }
        """

        self.setStyleSheet(dark_stylesheet)

    def save_window_state(self):
        """Save window state"""
        if self.app_settings.get("remember_window_size", True):
            self.settings.setValue("geometry", self.saveGeometry())
            self.settings.setValue("windowState", self.saveState())

    def restore_window_state(self):
        """Restore window state"""
        if self.app_settings.get("remember_window_size", True):
            geometry = self.settings.value("geometry")
            if geometry:
                self.restoreGeometry(geometry)

            window_state = self.settings.value("windowState")
            if window_state:
                self.restoreState(window_state)

        if self.app_settings.get("start_maximized", False):
            self.showMaximized()

    def closeEvent(self, event):
        """Handle close event"""
        if self.app_settings.get("confirm_exit", True):
            reply = QMessageBox.question(
                self,
                "Confirm Exit",
                "Are you sure you want to exit?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )

            if reply != QMessageBox.Yes:
                event.ignore()
                return

        # Stop camera if active
        if self.is_camera_active:
            self.stop_camera()

        # Unload model
        if self.model_loader.is_model_loaded():
            self.model_loader.unload_model()

        # Save window state
        self.save_window_state()

        # Accept the close event
        event.accept()
        logger.info("Application closed")

    def simple_colorize_image(self, rgb_image):
        """Simple colorization fallback for when no model is loaded or working"""
        try:
            if rgb_image is None:
                return None

            # Convert to grayscale first
            if len(rgb_image.shape) == 3:
                gray = self.image_processor.rgb_to_grayscale(rgb_image)
            else:
                gray = rgb_image

            # Create colorized version
            h, w = gray.shape
            colorized = np.zeros((h, w, 3), dtype=np.uint8)

            # Apply simple color mapping
            for y in range(h):
                for x in range(w):
                    intensity = gray[y, x]

                    if intensity > 180:  # Bright areas -> warm tones
                        colorized[y, x] = [
                            min(255, intensity + 10),
                            intensity,
                            max(0, intensity - 15),
                        ]
                    elif intensity < 60:  # Dark areas -> cool tones
                        colorized[y, x] = [
                            max(0, intensity - 10),
                            max(0, intensity - 5),
                            min(255, intensity + 20),
                        ]
                    else:  # Mid tones -> neutral with slight warmth
                        colorized[y, x] = [
                            intensity,
                            min(255, intensity + 5),
                            intensity,
                        ]

            return colorized

        except Exception as e:
            logger.error(f"Error in simple colorization: {e}")
            return rgb_image if rgb_image is not None else None

    def fix_model_file(self, file_path):
        """Fix a model file with unsafe operations"""
        try:
            import tempfile
            from pathlib import Path

            self.status_bar.showMessage("Fixing model file...")
            QApplication.processEvents()

            # Create fixed model path
            original_path = Path(file_path)
            fixed_path = (
                original_path.parent
                / f"{original_path.stem}_fixed{original_path.suffix}"
            )

            # Try to fix the model
            logger.info(f"Attempting to fix model: {file_path}")

            # Load with weights_only=False to read everything
            checkpoint = torch.load(
                file_path, map_location=self.model_loader.device, weights_only=False
            )

            # Create a clean checkpoint
            fixed_checkpoint = {}

            # Extract state dict
            if "generator_state_dict" in checkpoint:
                state_dict = checkpoint["generator_state_dict"]
            elif "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint

            # Clean the state dict
            clean_state_dict = {}
            for key, value in state_dict.items():
                if torch.is_tensor(value):
                    clean_state_dict[key] = value.cpu().contiguous()

            fixed_checkpoint["generator_state_dict"] = clean_state_dict

            # Add safe config
            if "config" in checkpoint:
                config = checkpoint["config"]
                safe_config = {}
                for k, v in config.items():
                    if isinstance(v, (int, float, str, bool)):
                        safe_config[k] = v
                    else:
                        safe_config[k] = str(v)
                fixed_checkpoint["config"] = safe_config
            else:
                fixed_checkpoint["config"] = {
                    "IN_CHANNELS": 1,
                    "OUT_CHANNELS": 3,
                    "USE_ATTENTION": True,
                    "USE_SE_BLOCKS": True,
                }

            # Add epoch info
            fixed_checkpoint["epoch"] = checkpoint.get("epoch", 1)

            # Save the fixed checkpoint
            torch.save(fixed_checkpoint, fixed_path)

            # Verify it can be loaded safely
            test_checkpoint = torch.load(
                str(fixed_path), map_location="cpu", weights_only=True
            )

            # Now load the fixed model
            self.load_model_from_path(str(fixed_path))

            QMessageBox.information(
                self,
                "Model Fixed",
                f"The model has been fixed and saved as:\n{fixed_path.name}\n\n"
                "The fixed model is now loaded and ready to use.",
            )

            return True

        except Exception as e:
            logger.error(f"Error fixing model: {e}")
            QMessageBox.critical(
                self,
                "Model Fix Failed",
                f"Could not fix the model file:\n{str(e)}\n\n"
                "Please try with a different model file or contact support.",
            )
            return False


# Import cv2 here to avoid import issues
try:
    import cv2
except ImportError:
    print("OpenCV not found. Please install with: pip install opencv-python")
