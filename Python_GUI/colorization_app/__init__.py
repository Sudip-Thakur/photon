"""
AI Colorization Studio - Real-time Grayscale to RGB Colorization App

A professional desktop application using PyQt5 that performs real-time grayscale to RGB
image colorization using trained Enhanced Pix2Pix models.

Features:
- Real-time camera feed colorization
- Image file colorization
- Batch processing of multiple images
- Model management and loading
- Before/After comparison views
- Professional UI with dark theme
- Export results in various formats

Author: AI Colorization Studio Team
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "AI Colorization Studio Team"
__email__ = "contact@aicolorization.studio"
__license__ = "MIT"

# Import main components
try:
    import os
    import sys

    # Add current directory to path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, current_dir)
    sys.path.insert(0, os.path.join(current_dir, "model"))
    sys.path.insert(0, os.path.join(current_dir, "ui"))
    sys.path.insert(0, os.path.join(current_dir, "utils"))

    from camera import CameraCapture, CameraThread
    from image_utils import ImageProcessor
    from main_window import MainWindow
    from model_loader import InferenceEngine, ModelLoader

    __all__ = [
        "MainWindow",
        "ModelLoader",
        "InferenceEngine",
        "ImageProcessor",
        "CameraCapture",
        "CameraThread",
        "__version__",
        "__author__",
        "__email__",
        "__license__",
    ]

except ImportError as e:
    # Handle import errors gracefully during development
    import logging

    logger = logging.getLogger(__name__)
    logger.warning(f"Some modules could not be imported: {e}")

    __all__ = ["__version__", "__author__", "__email__", "__license__"]


def get_version():
    """Get the current version of the application"""
    return __version__


def get_system_info():
    """Get system information for debugging"""
    import platform
    import sys

    try:
        import torch

        torch_version = torch.__version__
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            cuda_version = torch.version.cuda
        else:
            cuda_version = "Not available"
    except ImportError:
        torch_version = "Not installed"
        cuda_available = False
        cuda_version = "Not available"

    try:
        import cv2

        opencv_version = cv2.__version__
    except ImportError:
        opencv_version = "Not installed"

    try:
        from PyQt5.Qt import PYQT_VERSION_STR
        from PyQt5.QtCore import QT_VERSION_STR

        qt_version = QT_VERSION_STR
        pyqt_version = PYQT_VERSION_STR
    except ImportError:
        qt_version = "Not installed"
        pyqt_version = "Not installed"

    return {
        "app_version": __version__,
        "python_version": sys.version,
        "platform": platform.platform(),
        "architecture": platform.architecture()[0],
        "torch_version": torch_version,
        "cuda_available": cuda_available,
        "cuda_version": cuda_version,
        "opencv_version": opencv_version,
        "qt_version": qt_version,
        "pyqt_version": pyqt_version,
    }


# Application metadata
APP_METADATA = {
    "name": "AI Colorization Studio",
    "version": __version__,
    "description": "Professional desktop application for real-time grayscale to RGB image colorization",
    "author": __author__,
    "email": __email__,
    "license": __license__,
    "url": "https://github.com/aicolorization/studio",
    "keywords": [
        "colorization",
        "image-processing",
        "deep-learning",
        "pix2pix",
        "pytorch",
        "pyqt5",
        "computer-vision",
        "gan",
        "real-time",
    ],
    "requirements": [
        "PyQt5>=5.15.0",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "opencv-python>=4.8.0",
        "Pillow>=10.0.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "tqdm>=4.65.0",
        "PyYAML>=6.0",
        "psutil>=5.9.0",
    ],
}
