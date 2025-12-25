import cv2
import numpy as np
import torch
import time
from typing import Tuple, Optional
import base64
from PIL import Image
import io
from loguru import logger

from app.model.loader import model_loader
from app.utils.config import settings

class FrameProcessor:
    """Process individual frames"""
    
    @staticmethod
    def preprocess_frame(frame: np.ndarray) -> torch.Tensor:
        """Preprocess a frame for the model"""
        # Convert BGR to RGB
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Apply transformations
        transform = model_loader.get_transform()
        tensor = transform(frame)
        tensor = tensor.unsqueeze(0)  # Add batch dimension
        return tensor
    
    @staticmethod
    def postprocess_output(output_tensor: torch.Tensor, 
                          original_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """Convert model output back to image"""
        output_tensor = output_tensor.squeeze(0).detach().cpu()
        output_tensor = (output_tensor * 0.5 + 0.5).clamp(0, 1)
        
        # Convert to PIL Image
        from torchvision import transforms
        to_pil = transforms.ToPILImage()
        output_pil = to_pil(output_tensor)
        
        # Resize to original size if needed
        if original_size:
            output_pil = output_pil.resize(original_size[::-1], Image.Resampling.LANCZOS)
        
        # Convert to numpy array (RGB)
        output_np = np.array(output_pil)
        
        # Convert RGB to BGR for OpenCV
        output_np = cv2.cvtColor(output_np, cv2.COLOR_RGB2BGR)
        
        return output_np
    
    @staticmethod
    def process_single_frame(frame: np.ndarray) -> Tuple[np.ndarray, float]:
        """Process a single frame and return result with latency"""
        device = model_loader.device
        model = model_loader.get_model()
        
        # Preprocess
        input_tensor = FrameProcessor.preprocess_frame(frame)
        input_tensor = input_tensor.to(device)
        
        # Get original size for resizing back
        original_size = frame.shape[:2]
        
        # Inference
        start_time = time.perf_counter()
        
        with torch.no_grad():
            output_tensor = model(input_tensor)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        inference_time = (time.perf_counter() - start_time) * 1000  # ms
        
        # Postprocess
        output_frame = FrameProcessor.postprocess_output(output_tensor, original_size)
        
        return output_frame, inference_time
    
    @staticmethod
    def process_batch(frames: list) -> Tuple[list, float]:
        """Process a batch of frames"""
        if not frames:
            return [], 0.0
        
        device = model_loader.device
        model = model_loader.get_model()
        
        # Preprocess all frames
        batch_tensors = []
        original_sizes = []
        
        for frame in frames:
            original_sizes.append(frame.shape[:2])
            tensor = FrameProcessor.preprocess_frame(frame)
            batch_tensors.append(tensor)
        
        # Create batch
        batch = torch.cat(batch_tensors, dim=0).to(device)
        
        # Inference
        start_time = time.perf_counter()
        
        with torch.no_grad():
            outputs = model(batch)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        inference_time = (time.perf_counter() - start_time) * 1000  # ms
        
        # Postprocess each frame
        processed_frames = []
        for i in range(len(frames)):
            output_frame = FrameProcessor.postprocess_output(
                outputs[i:i+1],  # Get single frame
                original_sizes[i]
            )
            processed_frames.append(output_frame)
        
        # Calculate per-frame latency
        per_frame_time = inference_time / len(frames)
        
        return processed_frames, per_frame_time
    
    @staticmethod
    def image_to_base64(image: np.ndarray) -> str:
        """Convert numpy image to base64 string"""
        # Encode image as JPEG
        success, encoded_image = cv2.imencode('.jpg', image)
        if not success:
            raise ValueError("Failed to encode image")
        
        # Convert to base64
        base64_str = base64.b64encode(encoded_image).decode('utf-8')
        return base64_str
    
    @staticmethod
    def base64_to_image(base64_str: str) -> np.ndarray:
        """Convert base64 string to numpy image"""
        # Decode base64
        image_data = base64.b64decode(base64_str)
        
        # Convert to numpy array
        nparr = np.frombuffer(image_data, np.uint8)
        
        # Decode image
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Failed to decode image from base64")
        
        return image