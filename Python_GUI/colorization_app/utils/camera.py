"""
Camera Capture Utilities for Real-time Colorization
Supports both System Camera (OpenCV) and IR Camera (PyAV UDP stream)
"""

import logging
import threading
import time
from typing import Callable, Optional, Tuple

import av
import cv2
import numpy as np
import torch
from PyQt5.QtCore import QObject, QThread, pyqtSignal

# Set up logging
logger = logging.getLogger(__name__)

# ============================================================================
# IR Camera Configuration - CHANGE THESE VALUES TO MATCH YOUR IR CAMERA
# ============================================================================
IR_CAMERA_IP = "192.168.137.1"  # Change this to your IR camera's IP address
IR_CAMERA_PORT = 9000  # Change this to your IR camera's port
IR_CAMERA_TIMEOUT = 5  # Connection timeout in seconds
# ============================================================================

# Camera modes
CAMERA_MODE_SYSTEM = "system"
CAMERA_MODE_IR = "ir"


class CameraCapture(QObject):
    """Handles camera capture operations - supports both System and IR cameras"""

    def __init__(self, camera_id: int = 0, mode: str = CAMERA_MODE_SYSTEM):
        """
        Initialize camera capture

        Args:
            camera_id: Camera device ID (for system camera)
            mode: Camera mode - "system" or "ir"
        """
        super().__init__()
        self.camera_id = camera_id
        self.mode = mode
        self.is_capturing = False
        self.frame_width = 640
        self.frame_height = 480
        self.fps = 30

        # System camera (OpenCV)
        self.cap = None

        # IR Camera (PyAV)
        self.container = None
        self.video_stream = None
        self.ip_address = IR_CAMERA_IP
        self.port = IR_CAMERA_PORT
        self.timeout = IR_CAMERA_TIMEOUT
        self._url = self._build_url()

        # Thread-safe frame storage for IR camera
        self._latest_frame = None
        self._frame_lock = threading.Lock()
        self._capture_thread = None
        self._stop_event = threading.Event()

    def _build_url(self) -> str:
        """Build the UDP streaming URL for IR camera"""
        return (
            f"udp://{self.ip_address}:{self.port}?fifo_size=100000&overrun_nonfatal=1"
        )

    def set_mode(self, mode: str):
        """
        Set camera mode

        Args:
            mode: "system" or "ir"
        """
        if self.is_capturing:
            self.release_camera()
        self.mode = mode
        logger.info(f"Camera mode set to: {mode}")

    def set_ir_camera_config(self, ip_address: str, port: int):
        """
        Set IR camera IP configuration

        Args:
            ip_address: IP address of the IR camera
            port: UDP port for streaming
        """
        self.ip_address = ip_address
        self.port = port
        self._url = self._build_url()
        logger.info(f"IR Camera config updated: {self._url}")

    def initialize_camera(
        self, width: int = 640, height: int = 480, fps: int = 30
    ) -> bool:
        """
        Initialize camera with specified parameters

        Args:
            width: Frame width
            height: Frame height
            fps: Frames per second

        Returns:
            True if successful, False otherwise
        """
        if self.mode == CAMERA_MODE_SYSTEM:
            return self._initialize_system_camera(width, height, fps)
        else:
            return self._initialize_ir_camera(width, height, fps)

    def _initialize_system_camera(
        self, width: int = 640, height: int = 480, fps: int = 30
    ) -> bool:
        """Initialize system camera using OpenCV"""
        try:
            self.cap = cv2.VideoCapture(self.camera_id)

            if not self.cap.isOpened():
                logger.error(f"Cannot open system camera {self.camera_id}")
                return False

            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_FPS, fps)

            # Store actual values (camera might not support requested values)
            self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))

            self.is_capturing = True
            logger.info(
                f"System Camera initialized: {self.frame_width}x{self.frame_height} @ {self.fps}fps"
            )
            return True

        except Exception as e:
            logger.error(f"Error initializing system camera: {e}")
            return False

    def _initialize_ir_camera(
        self, width: int = 640, height: int = 480, fps: int = 30
    ) -> bool:
        """Initialize IR camera using PyAV"""
        try:
            # Release any existing connection
            self.release_camera()

            logger.info(f"Connecting to IR camera at {self._url}")

            # Use a thread with timeout for connection
            connection_result = {"success": False, "error": None}

            def connect_with_timeout():
                try:
                    self.container = av.open(
                        self._url,
                        options={
                            "fflags": "nobuffer",
                            "flags": "low_delay",
                            "analyzeduration": "500000",
                            "probesize": "32768",
                        },
                        timeout=self.timeout,
                    )
                    connection_result["success"] = True
                except Exception as e:
                    connection_result["error"] = str(e)
                    connection_result["success"] = False

            # Start connection in a thread
            connect_thread = threading.Thread(target=connect_with_timeout)
            connect_thread.daemon = True
            connect_thread.start()
            connect_thread.join(timeout=self.timeout + 2)

            if connect_thread.is_alive():
                logger.error("IR camera connection timed out")
                self.release_camera()
                return False

            if not connection_result["success"]:
                logger.error(
                    f"IR camera connection failed: {connection_result['error']}"
                )
                self.release_camera()
                return False

            if self.container is None:
                logger.error("Container is None after connection")
                return False

            # Find video stream
            self.video_stream = next(
                (s for s in self.container.streams if s.type == "video"), None
            )

            if self.video_stream is None:
                logger.error("No video stream found in IR camera feed")
                self.release_camera()
                return False

            # Get stream properties
            if hasattr(self.video_stream, "width") and self.video_stream.width:
                self.frame_width = self.video_stream.width
            else:
                self.frame_width = width

            if hasattr(self.video_stream, "height") and self.video_stream.height:
                self.frame_height = self.video_stream.height
            else:
                self.frame_height = height

            if (
                hasattr(self.video_stream, "average_rate")
                and self.video_stream.average_rate
            ):
                self.fps = float(self.video_stream.average_rate)
            else:
                self.fps = fps

            self.is_capturing = True
            self._stop_event.clear()

            # Start background capture thread
            self._capture_thread = threading.Thread(
                target=self._ir_capture_loop, daemon=True
            )
            self._capture_thread.start()

            logger.info(
                f"IR Camera initialized: {self.frame_width}x{self.frame_height} @ {self.fps}fps"
            )
            return True

        except Exception as e:
            logger.error(f"Error initializing IR camera: {e}")
            self.release_camera()
            return False

    def _ir_capture_loop(self):
        """Background thread for continuous IR camera frame capture"""
        try:
            for packet in self.container.demux(self.video_stream):
                if self._stop_event.is_set():
                    break

                try:
                    for frame in packet.decode():
                        if self._stop_event.is_set():
                            break

                        img = frame.to_ndarray(format="bgr24")

                        with self._frame_lock:
                            self._latest_frame = img

                except Exception as decode_error:
                    logger.warning(f"Frame decode error: {decode_error}")
                    continue

        except Exception as e:
            if not self._stop_event.is_set():
                logger.error(f"Error in IR capture loop: {e}")
        finally:
            logger.info("IR capture loop ended")

    def capture_frame(self) -> Optional[np.ndarray]:
        """
        Capture a single frame from camera

        Returns:
            Frame as numpy array (BGR) or None if failed
        """
        if self.mode == CAMERA_MODE_SYSTEM:
            return self._capture_system_frame()
        else:
            return self._capture_ir_frame()

    def _capture_system_frame(self) -> Optional[np.ndarray]:
        """Capture frame from system camera"""
        if self.cap is None or not self.cap.isOpened():
            return None

        try:
            ret, frame = self.cap.read()
            if ret:
                return frame
            else:
                logger.warning("Failed to capture frame from system camera")
                return None

        except Exception as e:
            logger.error(f"Error capturing frame: {e}")
            return None

    def _capture_ir_frame(self) -> Optional[np.ndarray]:
        """Capture frame from IR camera (non-blocking)"""
        if not self.is_capturing:
            return None

        with self._frame_lock:
            if self._latest_frame is not None:
                frame = self._latest_frame.copy()
                return frame
            return None

    def release_camera(self):
        """Release camera resources"""
        self.is_capturing = False

        # Release system camera
        if self.cap is not None:
            self.cap.release()
            self.cap = None

        # Release IR camera
        self._stop_event.set()

        if self._capture_thread is not None and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=2)
            self._capture_thread = None

        with self._frame_lock:
            self._latest_frame = None

        if self.container is not None:
            try:
                self.container.close()
            except Exception as e:
                logger.warning(f"Error closing container: {e}")
            finally:
                self.container = None

        self.video_stream = None
        logger.info(f"{self.mode.upper()} Camera released")

    def is_camera_available(self) -> bool:
        """Check if camera is available and working"""
        if self.mode == CAMERA_MODE_SYSTEM:
            return self.cap is not None and self.cap.isOpened()
        else:
            return self.is_capturing and self.container is not None

    def get_camera_info(self) -> dict:
        """Get camera information"""
        info = {
            "width": self.frame_width,
            "height": self.frame_height,
            "fps": self.fps,
            "camera_id": self.camera_id,
            "mode": self.mode,
        }

        if self.mode == CAMERA_MODE_IR:
            info.update(
                {
                    "ip_address": self.ip_address,
                    "port": self.port,
                    "url": self._url,
                }
            )

        return info

    @staticmethod
    def get_available_cameras() -> list:
        """
        Get list of available system cameras

        Returns:
            List of available camera IDs
        """
        available_cameras = []

        for i in range(6):
            try:
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    available_cameras.append(i)
                    cap.release()
            except:
                pass

        return available_cameras


class CameraThread(QThread):
    """Thread for continuous camera capture and processing"""

    # Signals
    frame_captured = pyqtSignal(np.ndarray)  # Original frame
    frame_processed = pyqtSignal(np.ndarray)  # Processed frame
    fps_updated = pyqtSignal(float)  # Current FPS
    error_occurred = pyqtSignal(str)  # Error message

    def __init__(self, camera_id: int = 0, mode: str = CAMERA_MODE_SYSTEM):
        """
        Initialize camera thread

        Args:
            camera_id: Camera device ID
            mode: Camera mode - "system" or "ir"
        """
        super().__init__()
        self.camera = CameraCapture(camera_id, mode)
        self.processing_function = None
        self.is_running = False
        self.frame_skip = 1  # Process every nth frame
        self.frame_count = 0

        # FPS calculation
        self.fps_start_time = time.time()
        self.fps_frame_count = 0
        self.current_fps = 0.0

    def set_mode(self, mode: str):
        """Set camera mode"""
        self.camera.set_mode(mode)

    def set_ir_camera_config(self, ip_address: str, port: int):
        """
        Set IR camera IP configuration

        Args:
            ip_address: IP address of the IR camera
            port: UDP port for streaming
        """
        self.camera.set_ir_camera_config(ip_address, port)

    def set_processing_function(
        self, func: Optional[Callable[[np.ndarray], np.ndarray]]
    ):
        """
        Set function to process frames

        Args:
            func: Function that takes frame and returns processed frame
        """
        self.processing_function = func

    def set_frame_skip(self, skip: int):
        """
        Set frame skip rate for processing

        Args:
            skip: Process every nth frame (1 = process all frames)
        """
        self.frame_skip = max(1, skip)

    def start_capture(self, width: int = 640, height: int = 480, fps: int = 30) -> bool:
        """
        Start camera capture

        Args:
            width: Frame width
            height: Frame height
            fps: Target FPS

        Returns:
            True if successful
        """
        if not self.camera.initialize_camera(width, height, fps):
            mode_name = "IR" if self.camera.mode == CAMERA_MODE_IR else "System"
            self.error_occurred.emit(
                f"Failed to initialize {mode_name} camera. Check settings."
            )
            return False

        self.is_running = True
        self.start()
        return True

    def stop_capture(self):
        """Stop camera capture"""
        self.is_running = False
        self.wait()  # Wait for thread to finish
        self.camera.release_camera()

    def run(self):
        """Main thread execution"""
        mode_name = "IR" if self.camera.mode == CAMERA_MODE_IR else "System"
        logger.info(f"{mode_name} Camera thread started")

        # Track frames for detecting stale data (IR camera)
        last_frame = None
        stale_count = 0
        max_stale = 30

        while self.is_running:
            try:
                # Capture frame
                frame = self.camera.capture_frame()

                if frame is None:
                    stale_count += 1
                    if stale_count > max_stale and self.camera.mode == CAMERA_MODE_IR:
                        logger.warning(
                            "No new frames received, stream may be disconnected"
                        )
                        stale_count = 0
                    self.msleep(33 if self.camera.mode == CAMERA_MODE_IR else 10)
                    continue

                # Check if frame is new (for IR camera)
                if self.camera.mode == CAMERA_MODE_IR:
                    if last_frame is not None and np.array_equal(frame, last_frame):
                        stale_count += 1
                        self.msleep(10)
                        continue

                stale_count = 0
                last_frame = (
                    frame.copy() if self.camera.mode == CAMERA_MODE_IR else None
                )
                self.frame_count += 1

                # Emit original frame
                self.frame_captured.emit(frame.copy())

                # Process frame if function is set and it's time to process
                if self.processing_function and (
                    self.frame_count % self.frame_skip == 0
                ):
                    try:
                        processed_frame = self.processing_function(frame)
                        if processed_frame is not None:
                            self.frame_processed.emit(processed_frame)
                    except Exception as e:
                        logger.error(f"Error in processing function: {e}")

                # Update FPS
                self.fps_frame_count += 1
                current_time = time.time()
                if current_time - self.fps_start_time >= 1.0:  # Update every second
                    self.current_fps = self.fps_frame_count / (
                        current_time - self.fps_start_time
                    )
                    self.fps_updated.emit(self.current_fps)

                    # Reset for next calculation
                    self.fps_start_time = current_time
                    self.fps_frame_count = 0

                # Small delay to prevent excessive CPU usage
                self.msleep(1)

            except Exception as e:
                logger.error(f"Error in camera thread: {e}")
                self.error_occurred.emit(f"Camera error: {str(e)}")
                break

        logger.info(f"{mode_name} Camera thread stopped")


class VideoRecorder:
    """Records video with optional processing"""

    def __init__(self):
        """Initialize video recorder"""
        self.writer = None
        self.is_recording = False
        self.output_path = ""
        self.frame_width = 0
        self.frame_height = 0
        self.fps = 30

    def start_recording(
        self, output_path: str, width: int, height: int, fps: int = 30
    ) -> bool:
        """
        Start video recording

        Args:
            output_path: Output video file path
            width: Frame width
            height: Frame height
            fps: Frames per second

        Returns:
            True if successful
        """
        try:
            # Define codec (MP4V for .mp4 files, XVID for .avi files)
            if output_path.lower().endswith(".mp4"):
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            else:
                fourcc = cv2.VideoWriter_fourcc(*"XVID")

            self.writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            if not self.writer.isOpened():
                logger.error("Failed to open video writer")
                return False

            self.is_recording = True
            self.output_path = output_path
            self.frame_width = width
            self.frame_height = height
            self.fps = fps

            logger.info(f"Started recording: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Error starting recording: {e}")
            return False

    def write_frame(self, frame: np.ndarray):
        """
        Write frame to video

        Args:
            frame: Frame to write (BGR format)
        """
        if not self.is_recording or self.writer is None:
            return

        try:
            # Ensure frame has correct size
            if frame.shape[:2] != (self.frame_height, self.frame_width):
                frame = cv2.resize(frame, (self.frame_width, self.frame_height))

            # Ensure frame is BGR (OpenCV format)
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                self.writer.write(frame)

        except Exception as e:
            logger.error(f"Error writing frame: {e}")

    def stop_recording(self):
        """Stop video recording"""
        if self.writer is not None:
            self.writer.release()
            self.writer = None

        self.is_recording = False
        logger.info(f"Recording stopped: {self.output_path}")

    def is_recording_active(self) -> bool:
        """Check if recording is active"""
        return self.is_recording


class FrameProcessor:
    """Processes frames for real-time colorization"""

    def __init__(self, inference_engine, image_processor):
        """
        Initialize frame processor

        Args:
            inference_engine: Model inference engine
            image_processor: Image processing utilities
        """
        self.inference_engine = inference_engine
        self.image_processor = image_processor
        self.processing_time = 0.0

    def process_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Process a single frame for colorization

        Args:
            frame: Input frame (BGR format from OpenCV)

        Returns:
            Colorized frame or None if processing failed
        """
        start_time = time.time()

        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Preprocess for model
            input_tensor = self.image_processor.preprocess_for_model(rgb_frame)

            # Colorize
            output_tensor = self.inference_engine.colorize(input_tensor)

            if output_tensor is None:
                return None

            # Postprocess
            colorized_rgb = self.image_processor.postprocess_from_model(output_tensor)

            # Convert back to BGR for OpenCV
            colorized_bgr = cv2.cvtColor(colorized_rgb, cv2.COLOR_RGB2BGR)

            # Record processing time
            self.processing_time = (time.time() - start_time) * 1000  # Convert to ms

            return colorized_bgr

        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return None

    def get_processing_time(self) -> float:
        """Get last processing time in milliseconds"""
        return self.processing_time

    def create_side_by_side_frame(
        self, original: np.ndarray, processed: np.ndarray
    ) -> np.ndarray:
        """
        Create side-by-side comparison frame

        Args:
            original: Original frame
            processed: Processed frame

        Returns:
            Combined frame
        """
        try:
            # Ensure same height
            h1, w1 = original.shape[:2]
            h2, w2 = processed.shape[:2]

            if h1 != h2:
                # Resize processed to match original height
                scale = h1 / h2
                new_w2 = int(w2 * scale)
                processed = cv2.resize(processed, (new_w2, h1))
                w2 = new_w2

            # Convert original to grayscale for comparison
            if len(original.shape) == 3:
                gray_original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
                gray_original = cv2.cvtColor(gray_original, cv2.COLOR_GRAY2BGR)
            else:
                gray_original = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)

            # Create combined frame
            gap = 5
            combined = np.ones((h1, w1 + w2 + gap, 3), dtype=np.uint8) * 255

            # Add frames
            combined[:, :w1] = gray_original
            combined[:, w1 + gap :] = processed

            return combined

        except Exception as e:
            logger.error(f"Error creating side-by-side frame: {e}")
            return original


def convert_frame_format(frame: np.ndarray, target_format: str = "RGB") -> np.ndarray:
    """
    Convert frame between different color formats

    Args:
        frame: Input frame
        target_format: Target format ("RGB", "BGR", "GRAY")

    Returns:
        Converted frame
    """
    if target_format == "RGB" and len(frame.shape) == 3:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    elif target_format == "BGR" and len(frame.shape) == 3:
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    elif target_format == "GRAY":
        if len(frame.shape) == 3:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame
    else:
        return frame


# ============================================================================
# Helper function to easily change IR camera settings from other modules
# ============================================================================
def set_ir_camera_settings(ip_address: str, port: int):
    """
    Update the global IR camera settings

    Args:
        ip_address: IP address of the IR camera
        port: UDP port for streaming

    Usage:
        from camera import set_ir_camera_settings
        set_ir_camera_settings("192.168.1.100", 9000)
    """
    global IR_CAMERA_IP, IR_CAMERA_PORT
    IR_CAMERA_IP = ip_address
    IR_CAMERA_PORT = port
    logger.info(f"Global IR camera settings updated: {ip_address}:{port}")
