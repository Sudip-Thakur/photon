#!/usr/bin/env python3
"""
AI Colorization Studio - Main Entry Point
A professional desktop application for real-time grayscale to RGB image colorization
using Enhanced Pix2Pix models with PyQt5.

Author: AI Colorization Studio Team
Version: 1.0.0

Usage:
    python main.py                                    # Use default IR camera settings
    python main.py --ip 192.168.137.1 --port 9000    # Custom IP and port
    python main.py --ip 192.168.1.100                # Custom IP with default port
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add the project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, "ui"))
sys.path.insert(0, os.path.join(current_dir, "model"))
sys.path.insert(0, os.path.join(current_dir, "utils"))


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="AI Colorization Studio - Real-time grayscale to RGB colorization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py                                    # Use default IR camera (192.168.137.1:9000)
    python main.py --ip 192.168.1.100                # Custom IP with default port
    python main.py --ip 10.0.0.50 --port 8000        # Custom IP and port
        """,
    )

    parser.add_argument(
        "--ip",
        type=str,
        default="192.168.137.1",
        help="IR camera IP address (default: 192.168.137.1)",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=9000,
        help="IR camera UDP port (default: 9000)",
    )

    return parser.parse_args()


def validate_ip(ip: str) -> bool:
    """Validate IP address format"""
    try:
        parts = ip.split(".")
        return len(parts) == 4 and all(0 <= int(part) <= 255 for part in parts)
    except ValueError:
        return False


def check_dependencies():
    """Check if required dependencies are installed"""
    missing = []

    try:
        from PyQt5.QtWidgets import QApplication
    except ImportError:
        missing.append("PyQt5")

    try:
        import torch
    except ImportError:
        missing.append("torch")

    try:
        import cv2
    except ImportError:
        missing.append("opencv-python")

    try:
        import av
    except ImportError:
        missing.append("av")

    if missing:
        print("❌ Missing required dependencies:")
        for dep in missing:
            print(f"   • {dep}")
        print("\n📦 Install missing packages:")
        print(f"   pip install {' '.join(missing)}")
        return False

    return True


def main():
    """Main application entry point"""
    # Parse command line arguments
    args = parse_arguments()

    print("🚀 AI Colorization Studio")
    print("=" * 40)
    print(f"📷 IR Camera IP: {args.ip}")
    print(f"📡 IR Camera Port: {args.port}")
    print("=" * 40)

    # Validate IP address
    if not validate_ip(args.ip):
        print(f"❌ Invalid IP address: {args.ip}")
        return 1

    # Validate port
    if not (1 <= args.port <= 65535):
        print(f"❌ Invalid port: {args.port}")
        return 1

    # Check dependencies
    if not check_dependencies():
        input("\nPress Enter to exit...")
        return 1

    # Configure IR camera settings BEFORE importing camera module
    # This ensures the settings are applied when camera module is loaded
    from camera import set_ir_camera_settings

    set_ir_camera_settings(args.ip, args.port)
    print(f"✅ IR Camera configured: {args.ip}:{args.port}")

    # Import after dependency check and camera configuration
    from main_window import MainWindow
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QApplication

    # Setup basic logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    try:
        # Create QApplication
        app = QApplication(sys.argv)
        app.setApplicationName("AI Colorization Studio")
        app.setApplicationVersion("1.0.0")

        # Enable High DPI support
        if hasattr(Qt, "AA_EnableHighDpiScaling"):
            app.setAttribute(Qt.AA_EnableHighDpiScaling, True)
        if hasattr(Qt, "AA_UseHighDpiPixmaps"):
            app.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

        # Create and show main window
        print("✅ Launching application...")
        main_window = MainWindow()
        main_window.show()

        logger.info("AI Colorization Studio started successfully")

        # Run the application
        return app.exec_()

    except Exception as e:
        print(f"❌ Error starting application: {e}")
        logger.error(f"Application startup failed: {e}")
        return 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n👋 Application interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"💥 Fatal error: {e}")
        sys.exit(1)
