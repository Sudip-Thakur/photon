# """
# Video file processing and management
# """
# import cv2
# import numpy as np
# import os
# import time
# import uuid
# from typing import Optional, Tuple, Dict, Any, Generator
# from pathlib import Path
# import json
# from loguru import logger
# import asyncio

# from app.utils.config import settings
# from app.model.preprocessor import FrameProcessor

# class VideoHandler:
#     """Handle video file processing"""
    
#     def __init__(self):
#         self.temp_dir = Path(settings.TEMP_DIR)
#         self.temp_dir.mkdir(exist_ok=True)
        
#     def validate_video(self, video_path: str) -> Tuple[bool, str, Dict[str, Any]]:
#         """Validate video file"""
#         if not os.path.exists(video_path):
#             return False, "File not found", {}
        
#         # Check file size
#         file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
#         if file_size_mb > settings.MAX_VIDEO_SIZE_MB:
#             return False, f"File too large ({file_size_mb:.1f}MB > {settings.MAX_VIDEO_SIZE_MB}MB)", {}
        
#         # Check extension
#         ext = Path(video_path).suffix.lower()
#         if ext not in settings.ALLOWED_EXTENSIONS:
#             return False, f"Unsupported file extension: {ext}", {}
        
#         # Try to open video
#         cap = cv2.VideoCapture(video_path)
#         if not cap.isOpened():
#             return False, "Cannot open video file", {}
        
#         # Get video info
#         fps = cap.get(cv2.CAP_PROP_FPS)
#         width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#         height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#         total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#         duration = total_frames / fps if fps > 0 else 0
        
#         cap.release()
        
#         info = {
#             "fps": fps,
#             "width": width,
#             "height": height,
#             "total_frames": total_frames,
#             "duration": duration,
#             "file_size_mb": file_size_mb,
#             "format": ext
#         }
        
#         return True, "Valid video", info
    
#     def process_video(self, 
#                      input_path: str, 
#                      output_path: Optional[str] = None,
#                      batch_size: int = 4,
#                      max_frames: Optional[int] = None) -> Dict[str, Any]:
#         """
#         Process a video file and save result
        
#         Returns:
#             Dictionary with processing results and statistics
#         """
#         logger.info(f"Processing video: {input_path}")
        
#         # Validate video
#         is_valid, message, video_info = self.validate_video(input_path)
#         if not is_valid:
#             raise ValueError(f"Invalid video: {message}")
        
#         # Generate output path if not provided
#         if output_path is None:
#             output_name = f"processed_{Path(input_path).stem}_{uuid.uuid4().hex[:8]}.mp4"
#             output_path = str(self.temp_dir / output_name)
        
#         # Open input video
#         cap = cv2.VideoCapture(input_path)
        
#         # Get video properties
#         fps = video_info["fps"]
#         width = video_info["width"]
#         height = video_info["height"]
#         total_frames = video_info["total_frames"]
        
#         # Use model output size for output video
#         output_size = (settings.MODEL_IMAGE_SIZE, settings.MODEL_IMAGE_SIZE)
        
#         # Create output video writer
#         fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#         out = cv2.VideoWriter(output_path, fourcc, fps, output_size)
        
#         # Statistics
#         stats = {
#             "input_path": input_path,
#             "output_path": output_path,
#             "total_frames": total_frames,
#             "processed_frames": 0,
#             "latencies": [],
#             "start_time": time.time(),
#             "errors": 0
#         }
        
#         frame_count = 0
#         batch_frames = []
        
#         if max_frames is None:
#             max_frames = total_frames
        
#         logger.info(f"Processing {min(max_frames, total_frames)} frames "
#                    f"({width}x{height} @ {fps}fps)")
        
#         try:
#             while frame_count < max_frames:
#                 ret, frame = cap.read()
#                 if not ret:
#                     break
                
#                 batch_frames.append(frame)
                
#                 # Process batch when full
#                 if len(batch_frames) >= batch_size:
#                     try:
#                         processed_batch, batch_time = FrameProcessor.process_batch(batch_frames)
                        
#                         # Write processed frames
#                         for processed_frame in processed_batch:
#                             out.write(processed_frame)
#                             stats["processed_frames"] += 1
#                             stats["latencies"].append(batch_time)
                        
#                         frame_count += len(batch_frames)
#                         batch_frames = []
                        
#                         # Log progress
#                         if frame_count % 50 == 0:
#                             elapsed = time.time() - stats["start_time"]
#                             fps_current = frame_count / elapsed if elapsed > 0 else 0
#                             logger.info(f"  Processed {frame_count}/{max_frames} frames "
#                                        f"({fps_current:.1f} FPS)")
                    
#                     except Exception as e:
#                         logger.error(f"Error processing batch: {e}")
#                         stats["errors"] += 1
#                         # Skip this batch
#                         batch_frames = []
            
#             # Process remaining frames
#             if batch_frames:
#                 try:
#                     processed_batch, batch_time = FrameProcessor.process_batch(batch_frames)
                    
#                     for processed_frame in processed_batch:
#                         out.write(processed_frame)
#                         stats["processed_frames"] += 1
#                         stats["latencies"].append(batch_time)
                    
#                     frame_count += len(batch_frames)
                
#                 except Exception as e:
#                     logger.error(f"Error processing final batch: {e}")
#                     stats["errors"] += 1
        
#         finally:
#             # Cleanup
#             cap.release()
#             out.release()
            
#             # Calculate statistics
#             stats["end_time"] = time.time()
#             stats["total_time"] = stats["end_time"] - stats["start_time"]
            
#             if stats["latencies"]:
#                 latencies = stats["latencies"]
#                 stats["avg_latency_ms"] = np.mean(latencies)
#                 stats["min_latency_ms"] = np.min(latencies)
#                 stats["max_latency_ms"] = np.max(latencies)
#                 stats["std_latency_ms"] = np.std(latencies)
#                 stats["avg_fps"] = 1000 / stats["avg_latency_ms"] if stats["avg_latency_ms"] > 0 else 0
#                 stats["processing_fps"] = stats["processed_frames"] / stats["total_time"]
            
#             # Success rate
#             stats["success_rate"] = (stats["processed_frames"] / frame_count * 100 
#                                      if frame_count > 0 else 0)
            
#             logger.info(f"Video processing complete: {stats['processed_frames']} frames "
#                        f"in {stats['total_time']:.2f}s "
#                        f"({stats.get('processing_fps', 0):.1f} FPS)")
            
#             return stats
    
#     async def process_video_async(self,
#                                 input_path: str,
#                                 output_path: Optional[str] = None,
#                                 progress_callback=None) -> Dict[str, Any]:
#         """Process video asynchronously with progress updates"""
#         # Run in thread pool to avoid blocking
#         loop = asyncio.get_event_loop()
#         result = await loop.run_in_executor(
#             None,
#             lambda: self.process_video(input_path, output_path)
#         )
        
#         return result
    
#     def get_frame_generator(self, video_path: str, max_frames: Optional[int] = None):
#         """Generator that yields frames from video"""
#         cap = cv2.VideoCapture(video_path)
#         frame_count = 0
        
#         try:
#             while True:
#                 if max_frames is not None and frame_count >= max_frames:
#                     break
                
#                 ret, frame = cap.read()
#                 if not ret:
#                     break
                
#                 yield frame
#                 frame_count += 1
        
#         finally:
#             cap.release()
    
#     def create_video_from_frames(self, frames: list, output_path: str, fps: int = 30):
#         """Create video from list of frames"""
#         if not frames:
#             raise ValueError("No frames provided")
        
#         # Get frame dimensions
#         height, width = frames[0].shape[:2]
        
#         # Create video writer
#         fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#         out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
#         for frame in frames:
#             out.write(frame)
        
#         out.release()
        
#         return output_path



"""
Video file processing and management - FIXED VERSION
"""
import cv2
import numpy as np
import os
import time
import uuid
from typing import Optional, Tuple, Dict, Any, Generator
from pathlib import Path
import json
from loguru import logger
import asyncio
import platform

from app.utils.config import settings
from app.model.preprocessor import FrameProcessor

class VideoHandler:
    """Handle video file processing"""
    
    def __init__(self):
        self.temp_dir = Path(settings.TEMP_DIR)
        self.temp_dir.mkdir(exist_ok=True)
    
    def _get_video_writer(self, output_path: str, fps: float, width: int, height: int):
        """
        Create a reliable video writer that works on Windows
        
        Returns: (video_writer, final_output_path)
        """
        # Try different codecs based on platform
        if platform.system() == 'Windows':
            # Windows often has issues with 'mp4v', try alternatives
            codec_options = [
                ('XVID', '.avi'),      # Most reliable on Windows
                ('MJPG', '.avi'),      # Good alternative
                ('mp4v', '.mp4'),      # Try MP4 as last resort
            ]
        else:
            # Linux/Mac usually works with mp4v
            codec_options = [
                ('mp4v', '.mp4'),
                ('XVID', '.avi'),
                ('avc1', '.mp4'),
            ]
        
        # Extract base name and extension
        base_name = Path(output_path).stem
        original_ext = Path(output_path).suffix.lower()
        
        for codec, preferred_ext in codec_options:
            try:
                # Use preferred extension for this codec
                if not output_path.endswith(preferred_ext):
                    output_path = str(Path(output_path).with_suffix(preferred_ext))
                
                fourcc = cv2.VideoWriter_fourcc(*codec)
                
                # Create video writer with isColor=True (important!)
                out = cv2.VideoWriter(
                    output_path, 
                    fourcc, 
                    fps, 
                    (width, height),
                    isColor=True  # Explicitly say it's color video
                )
                
                # Test if writer is opened
                if out.isOpened():
                    logger.info(f"✅ Using codec: {codec} for {output_path}")
                    
                    # Test write a dummy frame
                    dummy_frame = np.zeros((height, width, 3), dtype=np.uint8)
                    out.write(dummy_frame)
                    
                    return out, output_path
                else:
                    logger.warning(f"❌ Codec {codec} failed to open writer")
                    out.release()
                    
            except Exception as e:
                logger.warning(f"❌ Codec {codec} failed: {e}")
                continue
        
        raise RuntimeError(f"Failed to create video writer for {output_path}")
    
    def _ensure_frame_format(self, frame: np.ndarray) -> np.ndarray:
        """
        Ensure frame is in correct format for video writer
        - BGR format (3 channels)
        - uint8 dtype
        - Correct dimensions
        """
        if frame is None:
            raise ValueError("Frame is None")
        
        # Ensure uint8
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)
        
        # Ensure 3 channels (BGR)
        if len(frame.shape) == 2:  # Grayscale
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] == 1:  # Single channel
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] == 4:  # RGBA
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        elif frame.shape[2] == 3:
            # Already 3 channels, ensure it's BGR
            # If it came from model output (RGB), convert to BGR
            pass  # Assume it's already BGR
        
        return frame
    
    def validate_video(self, video_path: str) -> Tuple[bool, str, Dict[str, Any]]:
        """Validate video file"""
        if not os.path.exists(video_path):
            return False, "File not found", {}
        
        # Check file size
        file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
        if file_size_mb > settings.MAX_VIDEO_SIZE_MB:
            return False, f"File too large ({file_size_mb:.1f}MB > {settings.MAX_VIDEO_SIZE_MB}MB)", {}
        
        # Check extension
        ext = Path(video_path).suffix.lower()
        if ext not in settings.ALLOWED_EXTENSIONS:
            return False, f"Unsupported file extension: {ext}", {}
        
        # Try to open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False, "Cannot open video file", {}
        
        # Get video info
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        cap.release()
        
        info = {
            "fps": fps,
            "width": width,
            "height": height,
            "total_frames": total_frames,
            "duration": duration,
            "file_size_mb": file_size_mb,
            "format": ext
        }
        
        return True, "Valid video", info
    
    def process_video(self, 
                     input_path: str, 
                     output_path: Optional[str] = None,
                     batch_size: int = 4,
                     max_frames: Optional[int] = None) -> Dict[str, Any]:
        """
        Process a video file and save result
        
        Returns:
            Dictionary with processing results and statistics
        """
        logger.info(f"Processing video: {input_path}")
        
        # Validate video
        is_valid, message, video_info = self.validate_video(input_path)
        if not is_valid:
            raise ValueError(f"Invalid video: {message}")
        
        # Generate output path if not provided
        if output_path is None:
            output_name = f"processed_{Path(input_path).stem}_{uuid.uuid4().hex[:8]}"
            output_path = str(self.temp_dir / output_name)
        
        # Open input video
        cap = cv2.VideoCapture(input_path)
        
        # Get video properties
        fps = video_info["fps"]
        if fps <= 0:
            fps = 30.0  # Default if fps is invalid
        
        # Use model output size for output video
        output_width = settings.MODEL_IMAGE_SIZE
        output_height = settings.MODEL_IMAGE_SIZE
        
        # Create output video writer with reliable codec
        out, final_output_path = self._get_video_writer(
            output_path, fps, output_width, output_height
        )
        
        # Statistics
        stats = {
            "input_path": input_path,
            "output_path": final_output_path,
            "total_frames": video_info["total_frames"],
            "processed_frames": 0,
            "latencies": [],
            "start_time": time.time(),
            "errors": 0,
            "codec_used": Path(final_output_path).suffix
        }
        
        frame_count = 0
        batch_frames = []
        
        if max_frames is None:
            max_frames = video_info["total_frames"]
        
        logger.info(f"Processing {min(max_frames, video_info['total_frames'])} frames "
                   f"({video_info['width']}x{video_info['height']} @ {fps:.1f}fps) -> "
                   f"{output_width}x{output_height}")
        
        try:
            while frame_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                batch_frames.append(frame)
                
                # Process batch when full
                if len(batch_frames) >= batch_size:
                    try:
                        processed_batch, batch_time = FrameProcessor.process_batch(batch_frames)
                        
                        # Write processed frames
                        for processed_frame in processed_batch:
                            # Ensure frame format
                            processed_frame = self._ensure_frame_format(processed_frame)
                            
                            # Resize to output dimensions if needed
                            if (processed_frame.shape[1], processed_frame.shape[0]) != (output_width, output_height):
                                processed_frame = cv2.resize(
                                    processed_frame, 
                                    (output_width, output_height),
                                    interpolation=cv2.INTER_LINEAR
                                )
                            
                            # Write frame
                            out.write(processed_frame)
                            stats["processed_frames"] += 1
                            stats["latencies"].append(batch_time)
                        
                        frame_count += len(batch_frames)
                        batch_frames = []
                        
                        # Log progress
                        if frame_count % 50 == 0:
                            elapsed = time.time() - stats["start_time"]
                            fps_current = frame_count / elapsed if elapsed > 0 else 0
                            logger.info(f"  Processed {frame_count}/{max_frames} frames "
                                       f"({fps_current:.1f} FPS)")
                    
                    except Exception as e:
                        logger.error(f"Error processing batch: {e}")
                        stats["errors"] += 1
                        # Skip this batch
                        batch_frames = []
            
            # Process remaining frames
            if batch_frames:
                try:
                    processed_batch, batch_time = FrameProcessor.process_batch(batch_frames)
                    
                    for processed_frame in processed_batch:
                        processed_frame = self._ensure_frame_format(processed_frame)
                        
                        if (processed_frame.shape[1], processed_frame.shape[0]) != (output_width, output_height):
                            processed_frame = cv2.resize(
                                processed_frame, 
                                (output_width, output_height),
                                interpolation=cv2.INTER_LINEAR
                            )
                        
                        out.write(processed_frame)
                        stats["processed_frames"] += 1
                        stats["latencies"].append(batch_time)
                    
                    frame_count += len(batch_frames)
                
                except Exception as e:
                    logger.error(f"Error processing final batch: {e}")
                    stats["errors"] += 1
        
        finally:
            # Cleanup
            cap.release()
            if out is not None:
                out.release()
                logger.info(f"Video writer released for {final_output_path}")
            
            # Calculate statistics
            stats["end_time"] = time.time()
            stats["total_time"] = stats["end_time"] - stats["start_time"]
            
            if stats["latencies"]:
                latencies = stats["latencies"]
                stats["avg_latency_ms"] = np.mean(latencies)
                stats["min_latency_ms"] = np.min(latencies)
                stats["max_latency_ms"] = np.max(latencies)
                stats["std_latency_ms"] = np.std(latencies)
                stats["avg_fps"] = 1000 / stats["avg_latency_ms"] if stats["avg_latency_ms"] > 0 else 0
                stats["processing_fps"] = stats["processed_frames"] / stats["total_time"] if stats["total_time"] > 0 else 0
            
            # Success rate
            stats["success_rate"] = (stats["processed_frames"] / frame_count * 100 
                                     if frame_count > 0 else 0)
            
            # File size info
            if os.path.exists(final_output_path):
                stats["output_size_mb"] = os.path.getsize(final_output_path) / (1024 * 1024)
            
            logger.info(f"Video processing complete: {stats['processed_frames']} frames "
                       f"in {stats['total_time']:.2f}s "
                       f"({stats.get('processing_fps', 0):.1f} FPS)")
            
            return stats
    
    async def process_video_async(self,
                                input_path: str,
                                output_path: Optional[str] = None,
                                progress_callback=None) -> Dict[str, Any]:
        """Process video asynchronously with progress updates"""
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self.process_video(input_path, output_path)
        )
        
        return result
    
    def get_frame_generator(self, video_path: str, max_frames: Optional[int] = None):
        """Generator that yields frames from video"""
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        
        try:
            while True:
                if max_frames is not None and frame_count >= max_frames:
                    break
                
                ret, frame = cap.read()
                if not ret:
                    break
                
                yield frame
                frame_count += 1
        
        finally:
            cap.release()
    
    def create_video_from_frames(self, frames: list, output_path: str, fps: int = 30):
        """Create video from list of frames"""
        if not frames:
            raise ValueError("No frames provided")
        
        # Get frame dimensions from first frame
        first_frame = self._ensure_frame_format(frames[0])
        height, width = first_frame.shape[:2]
        
        # Create video writer
        out, final_output_path = self._get_video_writer(output_path, fps, width, height)
        
        frames_written = 0
        for frame in frames:
            frame = self._ensure_frame_format(frame)
            
            # Resize if dimensions don't match
            if (frame.shape[1], frame.shape[0]) != (width, height):
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
            
            out.write(frame)
            frames_written += 1
        
        out.release()
        
        logger.info(f"Created video: {final_output_path} ({frames_written} frames)")
        return final_output_path
    
    def save_as_image_sequence(self, frames: list, output_dir: str, prefix: str = "frame"):
        """
        Save frames as PNG image sequence (alternative to video)
        Returns path to directory containing images
        """
        os.makedirs(output_dir, exist_ok=True)
        
        for i, frame in enumerate(frames):
            frame = self._ensure_frame_format(frame)
            
            # Convert BGR to RGB for PNG
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Save as PNG
            output_path = os.path.join(output_dir, f"{prefix}_{i:04d}.png")
            cv2.imwrite(output_path, rgb_frame)
        
        logger.info(f"Saved {len(frames)} frames as PNG sequence in {output_dir}")
        
        # Create a README file with instructions
        readme_path = os.path.join(output_dir, "README.txt")
        with open(readme_path, 'w') as f:
            f.write(f"Image sequence: {len(frames)} frames\n")
            f.write("To create video with FFmpeg:\n")
            f.write(f'ffmpeg -framerate 30 -i "{prefix}_%04d.png" -c:v libx264 -pix_fmt yuv420p output.mp4\n')
        
        return output_dir