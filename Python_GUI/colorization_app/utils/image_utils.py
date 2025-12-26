"""
Image Processing Utilities for Colorization App
Handles image loading, preprocessing, postprocessing, and conversions
"""

import logging
from typing import Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image, ImageEnhance

# Set up logging
logger = logging.getLogger(__name__)


class ImageProcessor:
    """Handles all image processing operations for colorization"""

    def __init__(self, target_size: int = 256):
        """
        Initialize image processor

        Args:
            target_size: Target size for model input (square images)
        """
        self.target_size = target_size

        # Define transforms
        self.to_tensor = transforms.ToTensor()
        self.to_pil = transforms.ToPILImage()

        # Normalization for model input
        self.normalize = transforms.Normalize(mean=[0.5], std=[0.5])  # [-1, 1]
        self.denormalize = lambda x: (x + 1.0) / 2.0  # Back to [0, 1]

    def load_image(self, image_path: str) -> Optional[np.ndarray]:
        """
        Load image from file path

        Args:
            image_path: Path to image file

        Returns:
            Image as numpy array (RGB) or None if failed
        """
        try:
            # Use PIL for better format support
            image = Image.open(image_path)

            # Convert to RGB if needed
            if image.mode != "RGB":
                image = image.convert("RGB")

            # Convert to numpy array
            image_array = np.array(image)

            logger.info(f"Loaded image: {image_path}, shape: {image_array.shape}")
            return image_array

        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            return None

    def save_image(self, image: np.ndarray, save_path: str, quality: int = 95) -> bool:
        """
        Save image to file

        Args:
            image: Image array (RGB, 0-255)
            save_path: Output file path
            quality: JPEG quality (1-95)

        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure image is in correct format
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)

            # Convert to PIL Image
            pil_image = Image.fromarray(image)

            # Save with appropriate format
            if save_path.lower().endswith((".jpg", ".jpeg")):
                pil_image.save(save_path, "JPEG", quality=quality)
            elif save_path.lower().endswith(".png"):
                pil_image.save(save_path, "PNG")
            else:
                pil_image.save(save_path)

            logger.info(f"Image saved: {save_path}")
            return True

        except Exception as e:
            logger.error(f"Error saving image {save_path}: {e}")
            return False

    def rgb_to_grayscale(self, rgb_image: np.ndarray) -> np.ndarray:
        """
        Convert RGB image to grayscale

        Args:
            rgb_image: RGB image array

        Returns:
            Grayscale image array
        """
        if len(rgb_image.shape) == 3 and rgb_image.shape[2] == 3:
            # Use standard luminance weights
            gray = np.dot(rgb_image[..., :3], [0.299, 0.587, 0.114])
            return gray.astype(np.uint8)
        else:
            return rgb_image

    def numpy_to_tensor(
        self, image: np.ndarray, normalize: bool = True
    ) -> torch.Tensor:
        """
        Convert numpy array to PyTorch tensor

        Args:
            image: Input image array
            normalize: Whether to normalize to [-1, 1]

        Returns:
            PyTorch tensor
        """
        # Ensure correct format and data type
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = np.clip(image, 0, 255).astype(np.uint8)

        # Ensure contiguous array
        if not image.flags["C_CONTIGUOUS"]:
            image = np.ascontiguousarray(image)

        if len(image.shape) == 2:  # Grayscale
            image = np.expand_dims(image, axis=2)

        # Convert to PIL then tensor for proper channel ordering
        try:
            pil_image = Image.fromarray(
                image.squeeze() if image.shape[2] == 1 else image
            )
            tensor = self.to_tensor(pil_image)

            if normalize and tensor.max() <= 1.0:
                # Normalize to [-1, 1] for model input
                tensor = self.normalize(tensor)

            return tensor
        except Exception as e:
            logger.error(f"Error converting numpy to tensor: {e}")
            # Fallback: create tensor directly
            if len(image.shape) == 3:
                tensor = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            else:
                tensor = torch.from_numpy(image).unsqueeze(0).float() / 255.0

            if normalize:
                tensor = (tensor - 0.5) * 2.0  # Convert to [-1, 1]

            return tensor

    def tensor_to_numpy(
        self, tensor: torch.Tensor, denormalize: bool = True
    ) -> np.ndarray:
        """
        Convert PyTorch tensor to numpy array

        Args:
            tensor: Input tensor
            denormalize: Whether to denormalize from [-1, 1] to [0, 1]

        Returns:
            Numpy array (0-255, uint8)
        """
        try:
            # Ensure tensor is on CPU
            if tensor.is_cuda:
                tensor = tensor.cpu()

            # Remove batch dimension if present
            if tensor.dim() == 4:
                tensor = tensor.squeeze(0)

            # Denormalize if needed
            if denormalize and (tensor.min() < 0 or tensor.max() > 1):
                tensor = self.denormalize(tensor)

            # Clamp values
            tensor = torch.clamp(tensor, 0, 1)

            # Convert to numpy
            array = tensor.detach().numpy()

            # Transpose if needed (C, H, W) -> (H, W, C)
            if len(array.shape) == 3 and array.shape[0] in [1, 3]:
                array = np.transpose(array, (1, 2, 0))

            # Remove single channel dimension for grayscale
            if len(array.shape) == 3 and array.shape[2] == 1:
                array = array.squeeze(2)

            # Convert to 0-255 and ensure uint8
            array = np.clip(array * 255, 0, 255).astype(np.uint8)

            # Ensure contiguous array
            if not array.flags["C_CONTIGUOUS"]:
                array = np.ascontiguousarray(array)

            return array

        except Exception as e:
            logger.error(f"Error converting tensor to numpy: {e}")
            # Return a placeholder array if conversion fails
            return np.zeros((256, 256, 3), dtype=np.uint8)

    def resize_image(
        self,
        image: np.ndarray,
        target_size: Optional[int] = None,
        maintain_aspect: bool = True,
    ) -> np.ndarray:
        """
        Resize image to target size

        Args:
            image: Input image array
            target_size: Target size (uses self.target_size if None)
            maintain_aspect: Whether to maintain aspect ratio

        Returns:
            Resized image
        """
        if target_size is None:
            target_size = self.target_size

        h, w = image.shape[:2]

        if maintain_aspect:
            # Calculate scaling factor
            scale = target_size / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)

            # Resize
            resized = cv2.resize(
                image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4
            )

            # Pad to square
            if len(image.shape) == 3:
                padded = np.zeros(
                    (target_size, target_size, image.shape[2]), dtype=image.dtype
                )
            else:
                padded = np.zeros((target_size, target_size), dtype=image.dtype)

            # Center the image
            y_offset = (target_size - new_h) // 2
            x_offset = (target_size - new_w) // 2

            if len(image.shape) == 3:
                padded[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = (
                    resized
                )
            else:
                padded[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = (
                    resized
                )

            return padded
        else:
            # Direct resize
            return cv2.resize(
                image, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4
            )

    def center_crop(self, image: np.ndarray, crop_size: int) -> np.ndarray:
        """
        Center crop image to specified size

        Args:
            image: Input image
            crop_size: Size of the crop (square)

        Returns:
            Cropped image
        """
        h, w = image.shape[:2]

        if h < crop_size or w < crop_size:
            # Pad if image is smaller than crop size
            if len(image.shape) == 3:
                padded = np.zeros(
                    (max(crop_size, h), max(crop_size, w), image.shape[2]),
                    dtype=image.dtype,
                )
                pad_y = (padded.shape[0] - h) // 2
                pad_x = (padded.shape[1] - w) // 2
                padded[pad_y : pad_y + h, pad_x : pad_x + w] = image
            else:
                padded = np.zeros(
                    (max(crop_size, h), max(crop_size, w)), dtype=image.dtype
                )
                pad_y = (padded.shape[0] - h) // 2
                pad_x = (padded.shape[1] - w) // 2
                padded[pad_y : pad_y + h, pad_x : pad_x + w] = image

            image = padded
            h, w = image.shape[:2]

        # Calculate crop coordinates
        start_y = (h - crop_size) // 2
        start_x = (w - crop_size) // 2

        return image[start_y : start_y + crop_size, start_x : start_x + crop_size]
    
    def center_crop_to_square(self, image: np.ndarray) -> np.ndarray:
    
        h, w = image.shape[:2]
        
        if h == w:
            return image
        
        # Crop to smaller dimension (match training!)
        crop_size = min(h, w)
        
        # Calculate center crop coordinates
        start_y = (h - crop_size) // 2
        start_x = (w - crop_size) // 2
        
        # Crop
        if len(image.shape) == 3:
            return image[start_y:start_y+crop_size, start_x:start_x+crop_size, :]
        else:
            return image[start_y:start_y+crop_size, start_x:start_x+crop_size]

    # def preprocess_for_model(self, image: np.ndarray) -> torch.Tensor:
    #     """
    #     Complete preprocessing pipeline for model input

    #     Args:
    #         image: Input image (RGB or grayscale)

    #     Returns:
    #         Preprocessed tensor ready for model
    #     """
    #     try:
    #         # Ensure image is in correct format
    #         if image.dtype != np.uint8:
    #             if image.max() <= 1.0:
    #                 image = (image * 255).astype(np.uint8)
    #             else:
    #                 image = np.clip(image, 0, 255).astype(np.uint8)

    #         # Convert to grayscale if RGB
    #         if len(image.shape) == 3 and image.shape[2] == 3:
    #             image = self.rgb_to_grayscale(image)

    #         # Resize/crop to target size
    #         image = self.resize_image(image, maintain_aspect=True)

    #         # Convert to tensor and normalize
    #         tensor = self.numpy_to_tensor(image, normalize=True)

    #         # Add batch dimension
    #         tensor = tensor.unsqueeze(0)

    #         return tensor

    #     except Exception as e:
    #         logger.error(f"Error in preprocessing: {e}")
    #         # Return a default tensor if preprocessing fails
    #         return torch.zeros(1, 1, self.target_size, self.target_size)
    def preprocess_for_model(self, image: np.ndarray) -> torch.Tensor:
        """
        Complete preprocessing pipeline for model input.
        MUST MATCH TRAINING PREPROCESSING EXACTLY!
        
        Args:
            image: Input image (RGB or grayscale)
        
        Returns:
            Preprocessed tensor ready for model [1, 1, 256, 256]
        """
        try:
            # Ensure image is in correct format
            if image.dtype != np.uint8:
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                else:
                    image = np.clip(image, 0, 255).astype(np.uint8)
            
            # Convert to grayscale if RGB
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = self.rgb_to_grayscale(image)
            
            # STEP 1: CENTER CROP TO SQUARE (CRITICAL!)
            image = self.center_crop_to_square(image)
            
            # STEP 2: Resize to 256x256 (no padding!)
            image = cv2.resize(image, (self.target_size, self.target_size), 
                            interpolation=cv2.INTER_LINEAR)  # Use LINEAR like PIL
            
            # STEP 3: Convert to PIL Image then tensor (matches training)
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            
            pil_image = Image.fromarray(image)
            tensor = self.to_tensor(pil_image)  # [1, H, W], range [0, 1]
            
            # STEP 4: Normalize to [-1, 1] exactly like training
            tensor = self.normalize(tensor)  # (x - 0.5) / 0.5
            
            # STEP 5: Add batch dimension
            tensor = tensor.unsqueeze(0)  # [1, 1, 256, 256]
            
            return tensor
        
        except Exception as e:
            logger.error(f"Error in preprocessing: {e}")
            import traceback
            traceback.print_exc()
            # Return a default tensor if preprocessing fails
            return torch.zeros(1, 1, self.target_size, self.target_size)

    def postprocess_from_model(self, tensor: torch.Tensor) -> np.ndarray:
        """
        Complete postprocessing pipeline from model output

        Args:
            tensor: Model output tensor

        Returns:
            Processed image array (RGB, 0-255)
        """
        return self.tensor_to_numpy(tensor, denormalize=True)

    def enhance_image(
        self,
        image: np.ndarray,
        brightness: float = 1.0,
        contrast: float = 1.0,
        saturation: float = 1.0,
    ) -> np.ndarray:
        """
        Enhance image with brightness, contrast, and saturation adjustments

        Args:
            image: Input image array
            brightness: Brightness factor (1.0 = no change)
            contrast: Contrast factor (1.0 = no change)
            saturation: Saturation factor (1.0 = no change)

        Returns:
            Enhanced image array
        """
        try:
            # Convert to PIL
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)

            pil_image = Image.fromarray(image)

            # Apply enhancements
            if brightness != 1.0:
                enhancer = ImageEnhance.Brightness(pil_image)
                pil_image = enhancer.enhance(brightness)

            if contrast != 1.0:
                enhancer = ImageEnhance.Contrast(pil_image)
                pil_image = enhancer.enhance(contrast)

            if saturation != 1.0 and pil_image.mode == "RGB":
                enhancer = ImageEnhance.Color(pil_image)
                pil_image = enhancer.enhance(saturation)

            return np.array(pil_image)

        except Exception as e:
            logger.error(f"Error enhancing image: {e}")
            return image

    def create_side_by_side(
        self, left_image: np.ndarray, right_image: np.ndarray, gap: int = 10
    ) -> np.ndarray:
        """
        Create side-by-side comparison image

        Args:
            left_image: Left image
            right_image: Right image
            gap: Gap between images in pixels

        Returns:
            Combined image
        """
        # Ensure images have same height
        h1, w1 = left_image.shape[:2]
        h2, w2 = right_image.shape[:2]

        target_height = max(h1, h2)

        # Resize if needed
        if h1 != target_height:
            scale = target_height / h1
            new_w1 = int(w1 * scale)
            left_image = cv2.resize(left_image, (new_w1, target_height))
            w1 = new_w1

        if h2 != target_height:
            scale = target_height / h2
            new_w2 = int(w2 * scale)
            right_image = cv2.resize(right_image, (new_w2, target_height))
            w2 = new_w2

        # Create combined image
        total_width = w1 + w2 + gap

        if len(left_image.shape) == 3:
            combined = (
                np.ones(
                    (target_height, total_width, left_image.shape[2]),
                    dtype=left_image.dtype,
                )
                * 255
            )
            combined[:, :w1] = left_image
            combined[:, w1 + gap :] = right_image
        else:
            combined = (
                np.ones((target_height, total_width), dtype=left_image.dtype) * 255
            )
            combined[:, :w1] = left_image
            combined[:, w1 + gap :] = right_image

        return combined

    @staticmethod
    def get_supported_formats() -> list:
        """Get list of supported image formats"""
        return [".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"]

    @staticmethod
    def is_valid_image_file(file_path: str) -> bool:
        """Check if file has valid image extension"""
        return any(
            file_path.lower().endswith(ext)
            for ext in ImageProcessor.get_supported_formats()
        )


class BatchProcessor:
    """Handles batch processing of multiple images"""

    def __init__(self, image_processor: ImageProcessor):
        """
        Initialize batch processor

        Args:
            image_processor: Image processor instance
        """
        self.image_processor = image_processor

    def process_folder(
        self,
        input_folder: str,
        output_folder: str,
        inference_engine,
        progress_callback=None,
    ) -> dict:
        """
        Process all images in a folder

        Args:
            input_folder: Input directory path
            output_folder: Output directory path
            inference_engine: Inference engine for colorization
            progress_callback: Callback function for progress updates

        Returns:
            Dictionary with processing results
        """
        import os
        from pathlib import Path

        # Get all image files
        input_path = Path(input_folder)
        image_files = []

        for ext in ImageProcessor.get_supported_formats():
            image_files.extend(input_path.glob(f"*{ext}"))
            image_files.extend(input_path.glob(f"*{ext.upper()}"))

        if not image_files:
            return {"success": False, "error": "No image files found in input folder"}

        # Create output directory
        os.makedirs(output_folder, exist_ok=True)

        results = {
            "success": True,
            "total_images": len(image_files),
            "processed": 0,
            "failed": 0,
            "errors": [],
        }

        for i, image_file in enumerate(image_files):
            try:
                # Load image
                image = self.image_processor.load_image(str(image_file))
                if image is None:
                    raise ValueError("Failed to load image")

                # Preprocess
                input_tensor = self.image_processor.preprocess_for_model(image)

                # Colorize
                output_tensor = inference_engine.colorize(input_tensor)
                if output_tensor is None:
                    raise ValueError("Colorization failed")

                # Postprocess
                colorized = self.image_processor.postprocess_from_model(output_tensor)

                # Save result
                output_path = os.path.join(
                    output_folder, f"colorized_{image_file.name}"
                )
                if not self.image_processor.save_image(colorized, output_path):
                    raise ValueError("Failed to save image")

                results["processed"] += 1

                # Update progress
                if progress_callback:
                    progress = (i + 1) / len(image_files) * 100
                    progress_callback(progress, f"Processed {image_file.name}")

            except Exception as e:
                results["failed"] += 1
                results["errors"].append(f"{image_file.name}: {str(e)}")
                logger.error(f"Error processing {image_file}: {e}")

        return results
