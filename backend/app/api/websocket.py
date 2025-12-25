import asyncio
import av
import cv2
import numpy as np
import json
import time
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from loguru import logger
import threading

from app.model.preprocessor import FrameProcessor
from app.model.loader import model_loader

router = APIRouter()


class StreamProcessor:
    """Handles UDP/RTSP video stream capture and processing"""
    
    def __init__(self):
        self.container = None
        self.video_stream = None
        self.processing = False
        self.stream_url = None
        self.fps = 30
        self.frame_width = 640
        self.frame_height = 480
    
    def start_stream(self, stream_url: str):
        """Start UDP/RTSP stream capture using PyAV"""
        try:
            logger.info(f"Opening stream: {stream_url}")
            
            # Open stream with PyAV
            self.container = av.open(
                stream_url,
                options={
                    "fflags": "nobuffer",
                    "flags": "low_delay",
                    "analyzeduration": "0",
                    "probesize": "32",
                }
            )
            
            # Get video stream
            self.video_stream = next(s for s in self.container.streams if s.type == "video")
            
            # Get stream properties
            if self.video_stream.width:
                self.frame_width = self.video_stream.width
            if self.video_stream.height:
                self.frame_height = self.video_stream.height
            if self.video_stream.average_rate:
                self.fps = float(self.video_stream.average_rate)
            
            self.stream_url = stream_url
            self.processing = True
            
            logger.info(f"Stream started: {self.frame_width}x{self.frame_height} @ {self.fps}fps")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start stream: {e}")
            return False
    
    def stop_stream(self):
        """Stop stream capture"""
        self.processing = False
        if self.container:
            try:
                self.container.close()
            except:
                pass
            self.container = None
        self.video_stream = None
        logger.info("Stream stopped")
    
    async def process_and_stream(self, websocket: WebSocket):
        """
        Capture frames from stream, process with model, send via WebSocket
        """
        if not self.container or not self.video_stream:
            await websocket.send_json({"error": "Stream not started"})
            return
        
        frame_count = 0
        start_time = time.time()
        total_latency = 0
        
        try:
            for packet in self.container.demux(self.video_stream):
                if not self.processing:
                    break
                
                try:
                    for frame in packet.decode():
                        if not self.processing:
                            break
                        
                        # Convert PyAV frame to numpy array
                        img = frame.to_ndarray(format="bgr24")
                        
                        # Process frame with model
                        process_start = time.perf_counter()
                        processed_frame, model_latency = FrameProcessor.process_single_frame(img)
                        process_time = (time.perf_counter() - process_start) * 1000
                        
                        frame_count += 1
                        total_latency += process_time
                        
                        # Encode to JPEG
                        encode_start = time.perf_counter()
                        success, jpeg_buffer = cv2.imencode('.jpg', processed_frame, 
                                                           [cv2.IMWRITE_JPEG_QUALITY, 85])
                        encode_time = (time.perf_counter() - encode_start) * 1000
                        
                        if not success:
                            logger.warning("Failed to encode frame")
                            continue
                        
                        total_time = process_time + encode_time
                        
                        # Prepare metadata
                        metadata = {
                            "frame": frame_count,
                            "model_latency_ms": model_latency,
                            "processing_time_ms": process_time,
                            "encoding_time_ms": encode_time,
                            "total_time_ms": total_time,
                            "current_fps": 1000 / total_time if total_time > 0 else 0,
                            "avg_fps": frame_count / ((time.time() - start_time) + 0.001),
                            "timestamp": time.time(),
                            "frame_size": len(jpeg_buffer.tobytes()),
                            "resolution": f"{img.shape[1]}x{img.shape[0]}",
                            "stream_url": self.stream_url
                        }
                        
                        # Send metadata as JSON
                        await websocket.send_json(metadata)
                        
                        # Send frame as binary
                        await websocket.send_bytes(jpeg_buffer.tobytes())
                        
                        # Small delay
                        await asyncio.sleep(0.001)
                        
                except Exception as e:
                    logger.error(f"Frame processing error: {e}")
                    continue
                
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            raise
        
        finally:
            logger.info(f"Streaming stopped. Processed {frame_count} frames")


class CameraProcessor:
    """Handles real-time camera capture and processing (Original - for system camera)"""
    
    def __init__(self):
        self.cap = None
        self.processing = False
        self.fps = 30
        self.frame_width = 640
        self.frame_height = 480
    
    def start_camera(self, camera_id: int = 0):
        """Start camera capture"""
        try:
            self.cap = cv2.VideoCapture(camera_id)
            
            if not self.cap.isOpened():
                logger.error(f"Cannot open camera {camera_id}")
                return False
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            # Get actual properties
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            logger.info(f"Camera started: {actual_width}x{actual_height} @ {actual_fps}fps")
            self.processing = True
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start camera: {e}")
            return False
    
    def stop_camera(self):
        """Stop camera capture"""
        self.processing = False
        if self.cap:
            self.cap.release()
            self.cap = None
        logger.info("Camera stopped")
    
    async def process_and_stream(self, websocket: WebSocket):
        """
        Capture frames from camera, process with model, send via WebSocket
        """
        if not self.cap or not self.cap.isOpened():
            await websocket.send_json({"error": "Camera not started"})
            return
        
        frame_count = 0
        start_time = time.time()
        total_latency = 0
        
        try:
            while self.processing:
                # Capture frame
                ret, frame = self.cap.read()
                if not ret:
                    logger.warning("Failed to capture frame")
                    await asyncio.sleep(0.01)
                    continue
                
                # Process frame with model
                process_start = time.perf_counter()
                processed_frame, model_latency = FrameProcessor.process_single_frame(frame)
                process_time = (time.perf_counter() - process_start) * 1000
                
                frame_count += 1
                total_latency += process_time
                
                # Encode to JPEG (compressed for transmission)
                encode_start = time.perf_counter()
                success, jpeg_buffer = cv2.imencode('.jpg', processed_frame, 
                                                   [cv2.IMWRITE_JPEG_QUALITY, 85])
                encode_time = (time.perf_counter() - encode_start) * 1000
                
                if not success:
                    logger.warning("Failed to encode frame")
                    continue
                
                total_time = process_time + encode_time
                
                # Prepare metadata
                metadata = {
                    "frame": frame_count,
                    "model_latency_ms": model_latency,
                    "processing_time_ms": process_time,
                    "encoding_time_ms": encode_time,
                    "total_time_ms": total_time,
                    "current_fps": 1000 / total_time if total_time > 0 else 0,
                    "avg_fps": frame_count / ((time.time() - start_time) + 0.001),
                    "timestamp": time.time(),
                    "frame_size": len(jpeg_buffer.tobytes()),
                    "resolution": f"{frame.shape[1]}x{frame.shape[0]}"
                }
                
                # Send metadata as JSON
                await websocket.send_json(metadata)
                
                # Send frame as binary
                await websocket.send_bytes(jpeg_buffer.tobytes())
                
                # Small delay to control frame rate
                await asyncio.sleep(0.001)
                
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            raise
        
        finally:
            logger.info(f"Streaming stopped. Processed {frame_count} frames")


# Global instances
camera_processor = CameraProcessor()
stream_processor = StreamProcessor()


# =====================================================
# WebSocket Endpoints for UDP/RTSP Stream
# =====================================================

@router.websocket("/ws/stream/realtime")
async def stream_realtime_websocket(websocket: WebSocket):
    """
    Real-time UDP/RTSP stream processing WebSocket
    
    Flow:
    1. Client connects
    2. Client sends start command with stream URL
    3. Server starts stream, processes frames, sends JSON metadata + binary frames
    4. Client receives real-time colorized video
    
    Message format from client:
    - Start: {"action": "start", "stream_url": "udp://192.168.137.1:9000?fifo_size=100000&overrun_nonfatal=1"}
    - Stop: {"action": "stop"}
    """
    await websocket.accept()
    logger.info("Real-time stream WebSocket connected")
    
    try:
        # Wait for initial command
        message = await websocket.receive_text()
        config = json.loads(message)
        
        if config.get("action") == "start":
            stream_url = config.get("stream_url")
            
            if not stream_url:
                await websocket.send_json({"error": "stream_url is required"})
                await websocket.close()
                return
            
            # Start stream
            success = stream_processor.start_stream(stream_url)
            if not success:
                await websocket.send_json({"error": "Failed to start stream"})
                await websocket.close()
                return
            
            # Send confirmation
            await websocket.send_json({
                "status": "stream_started",
                "stream_url": stream_url,
                "resolution": f"{stream_processor.frame_width}x{stream_processor.frame_height}",
                "fps": stream_processor.fps,
                "model_device": str(model_loader.device)
            })
            
            # Start processing and streaming
            await stream_processor.process_and_stream(websocket)
            
        elif config.get("action") == "stop":
            stream_processor.stop_stream()
            await websocket.send_json({"status": "stream_stopped"})
            
        else:
            await websocket.send_json({"error": "Unknown action"})
    
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
        stream_processor.stop_stream()
        
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        stream_processor.stop_stream()
        await websocket.send_json({"error": str(e)})


@router.websocket("/ws/stream/simple")
async def stream_simple_websocket(websocket: WebSocket):
    """
    Simple UDP stream WebSocket - provide URL in connection
    Auto-starts on connection with default settings
    """
    await websocket.accept()
    logger.info("Simple stream WebSocket connected")
    
    try:
        # Wait for stream URL
        message = await websocket.receive_text()
        config = json.loads(message)
        stream_url = config.get("stream_url", "udp://192.168.137.1:9000?fifo_size=100000&overrun_nonfatal=1")
        
        # Start stream
        success = stream_processor.start_stream(stream_url)
        if not success:
            await websocket.send_json({"error": "Failed to start stream"})
            await websocket.close()
            return
        
        # Send initial info
        await websocket.send_json({
            "status": "streaming_started",
            "stream_url": stream_url,
            "message": "Sending JSON metadata + binary frames"
        })
        
        # Start streaming
        await stream_processor.process_and_stream(websocket)
        
    except WebSocketDisconnect:
        logger.info("Simple stream WebSocket disconnected")
        
    except Exception as e:
        logger.error(f"Simple stream error: {e}")
        
    finally:
        stream_processor.stop_stream()


# =====================================================
# Original Camera WebSocket Endpoints (Keep these)
# =====================================================

@router.websocket("/ws/camera/realtime")
async def camera_realtime_websocket(websocket: WebSocket):
    """Original camera endpoint - uses system camera"""
    await websocket.accept()
    logger.info("Real-time camera WebSocket connected")
    
    try:
        message = await websocket.receive_text()
        config = json.loads(message)
        
        if config.get("action") == "start":
            camera_id = config.get("camera_id", 0)
            
            if "width" in config:
                camera_processor.frame_width = config["width"]
            if "height" in config:
                camera_processor.frame_height = config["height"]
            if "fps" in config:
                camera_processor.fps = config["fps"]
            
            success = camera_processor.start_camera(camera_id)
            if not success:
                await websocket.send_json({"error": "Failed to start camera"})
                await websocket.close()
                return
            
            await websocket.send_json({
                "status": "camera_started",
                "camera_id": camera_id,
                "resolution": f"{camera_processor.frame_width}x{camera_processor.frame_height}",
                "fps": camera_processor.fps,
                "model_device": str(model_loader.device)
            })
            
            await camera_processor.process_and_stream(websocket)
            
        elif config.get("action") == "stop":
            camera_processor.stop_camera()
            await websocket.send_json({"status": "camera_stopped"})
            
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
        camera_processor.stop_camera()
        
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        camera_processor.stop_camera()


@router.websocket("/ws/camera/simple")
async def camera_simple_websocket(websocket: WebSocket):
    """Simple camera WebSocket - starts immediately on connection"""
    await websocket.accept()
    logger.info("Simple camera WebSocket connected")
    
    success = camera_processor.start_camera(0)
    if not success:
        await websocket.send_json({"error": "Failed to start camera"})
        await websocket.close()
        return
    
    try:
        await websocket.send_json({
            "status": "streaming_started",
            "camera_id": 0,
            "message": "Sending JSON metadata + binary frames"
        })
        
        await camera_processor.process_and_stream(websocket)
        
    except WebSocketDisconnect:
        logger.info("Simple camera WebSocket disconnected")
        
    except Exception as e:
        logger.error(f"Simple camera error: {e}")
        
    finally:
        camera_processor.stop_camera()