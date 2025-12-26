"""
Model Loader for Enhanced Pix2Pix Colorization
Handles loading of trained model checkpoints and provides inference capabilities
"""

import logging
import os
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from enhanced_pix2pix import EnhancedGeneratorUNet, create_enhanced_generator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelLoader:
    """Handles loading and managing trained Pix2Pix models"""

    def __init__(self, device: Optional[torch.device] = None):
        """
        Initialize model loader

        Args:
            device: PyTorch device to load models on. If None, auto-detects GPU/CPU
        """
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.generator = None
        self.model_info = {}
        self.is_loaded = False

        logger.info(f"ModelLoader initialized with device: {self.device}")

    def load_checkpoint(self, checkpoint_path: str, strict: bool = True) -> bool:
        """
        Load a trained model checkpoint

        Args:
            checkpoint_path: Path to the .pth checkpoint file
            strict: Whether to strictly enforce state dict loading

        Returns:
            bool: True if loading successful, False otherwise
        """
        try:
            if not os.path.exists(checkpoint_path):
                logger.error(f"Checkpoint file not found: {checkpoint_path}")
                return False

            logger.info(f"Loading checkpoint from: {checkpoint_path}")

            # Load checkpoint with proper handling for security
            try:
                # First try with weights_only=True for security
                checkpoint = torch.load(
                    checkpoint_path, map_location=self.device, weights_only=True
                )
            except Exception as e:
                logger.warning(f"Failed to load with weights_only=True: {e}")
                try:
                    # Fall back to unsafe loading with pickle
                    import pickle

                    import numpy

                    # Add safe globals for numpy operations
                    safe_globals = {
                        "__builtins__": {},
                        "torch": torch,
                        "collections": __import__("collections"),
                        "numpy": numpy,
                        "np": numpy,
                    }
                    # Load with unsafe mode but controlled globals
                    with open(checkpoint_path, "rb") as f:
                        checkpoint = torch.load(
                            f, map_location=self.device, pickle_module=pickle
                        )
                except Exception as e2:
                    logger.error(f"Failed to load checkpoint: {e2}")
                    raise e2

            # Extract model configuration if available
            config = checkpoint.get("config", {})

            # Get model parameters
            in_channels = config.get("IN_CHANNELS", 1)
            out_channels = config.get("OUT_CHANNELS", 3)
            use_attention = config.get("USE_ATTENTION", True)
            use_se = config.get("USE_SE_BLOCKS", True)

            # Create generator model
            self.generator = create_enhanced_generator(
                in_channels=in_channels,
                out_channels=out_channels,
                use_attention=use_attention,
                use_se=use_se,
            )

            # Load state dict
            if "generator_state_dict" in checkpoint:
                state_dict = checkpoint["generator_state_dict"]
            elif "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                # Assume the entire checkpoint is the state dict
                state_dict = checkpoint

            # Handle potential key mismatches
            try:
                self.generator.load_state_dict(state_dict, strict=strict)
            except RuntimeError as e:
                if "weights_only" in str(e) or "WeightsUnpickler" in str(e):
                    logger.warning(
                        "Model contains custom objects, loading with weights_only=False"
                    )
                    # Reload without weights_only restriction
                    try:
                        checkpoint = torch.load(
                            checkpoint_path,
                            map_location=self.device,
                            weights_only=False,
                        )
                        if "generator_state_dict" in checkpoint:
                            state_dict = checkpoint["generator_state_dict"]
                        elif "model_state_dict" in checkpoint:
                            state_dict = checkpoint["model_state_dict"]
                        else:
                            state_dict = checkpoint
                        self.generator.load_state_dict(state_dict, strict=False)
                    except Exception as e2:
                        logger.error(f"Failed to reload model: {e2}")
                        raise e2
                elif not strict:
                    logger.warning(f"Loading with strict=False due to: {e}")
                    self.generator.load_state_dict(state_dict, strict=False)
                else:
                    raise e

            # Move model to device and set to eval mode
            self.generator.to(self.device)
            self.generator.eval()

            # Store model information
            self.model_info = {
                "checkpoint_path": checkpoint_path,
                "epoch": checkpoint.get("epoch", "unknown"),
                "in_channels": in_channels,
                "out_channels": out_channels,
                "use_attention": use_attention,
                "use_se": use_se,
                "device": str(self.device),
                "total_params": sum(p.numel() for p in self.generator.parameters()),
                "trainable_params": sum(
                    p.numel() for p in self.generator.parameters() if p.requires_grad
                ),
            }

            self.is_loaded = True
            logger.info(f"Model loaded successfully! Epoch: {self.model_info['epoch']}")
            logger.info(f"Total parameters: {self.model_info['total_params']:,}")

            return True

        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            self.generator = None
            self.model_info = {}
            self.is_loaded = False
            return False

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        return self.model_info.copy()

    def is_model_loaded(self) -> bool:
        """Check if a model is currently loaded"""
        return self.is_loaded and self.generator is not None

    def get_generator(self) -> Optional[EnhancedGeneratorUNet]:
        """Get the loaded generator model"""
        return self.generator if self.is_loaded else None

    def unload_model(self):
        """Unload the current model and free memory"""
        if self.generator is not None:
            del self.generator
            self.generator = None

        self.model_info = {}
        self.is_loaded = False

        # Clear GPU cache if using CUDA
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        logger.info("Model unloaded and memory cleared")

    def get_device(self) -> torch.device:
        """Get the current device"""
        return self.device

    def set_device(self, device: torch.device):
        """
        Change the device and move model if loaded

        Args:
            device: New device to use
        """
        old_device = self.device
        self.device = device

        if self.is_loaded and self.generator is not None:
            logger.info(f"Moving model from {old_device} to {device}")
            self.generator.to(device)
            self.model_info["device"] = str(device)

        # Clear old device cache
        if old_device.type == "cuda":
            torch.cuda.empty_cache()

    def validate_checkpoint(self, checkpoint_path: str) -> Tuple[bool, str]:
        """
        Validate a checkpoint file without loading it

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            if not os.path.exists(checkpoint_path):
                return False, f"File not found: {checkpoint_path}"

            # Try to load just the checkpoint structure
            try:
                checkpoint = torch.load(
                    checkpoint_path, map_location="cpu", weights_only=True
                )
            except Exception:
                # Fall back to unsafe loading for validation
                try:
                    checkpoint = torch.load(
                        checkpoint_path, map_location="cpu", weights_only=False
                    )
                except Exception as e:
                    return False, f"Cannot load checkpoint: {str(e)}"

            # Check for required keys
            required_keys = ["generator_state_dict", "model_state_dict"]
            has_required = any(key in checkpoint for key in required_keys)

            if not has_required and not isinstance(checkpoint, dict):
                return (
                    False,
                    "Invalid checkpoint format - no recognizable state dict found",
                )

            # Check if it looks like a state dict
            if not has_required:
                # Assume entire checkpoint is state dict
                first_key = next(iter(checkpoint.keys()))
                if not (
                    isinstance(first_key, str)
                    and (
                        "." in first_key or "weight" in first_key or "bias" in first_key
                    )
                ):
                    return False, "Checkpoint doesn't appear to contain model weights"

            return True, "Valid checkpoint"

        except Exception as e:
            return False, f"Error validating checkpoint: {str(e)}"


class InferenceEngine:
    """Handles model inference for colorization"""

    def __init__(self, model_loader: ModelLoader):
        """
        Initialize inference engine

        Args:
            model_loader: Loaded model loader instance
        """
        self.model_loader = model_loader
        self.device = model_loader.get_device()

    def preprocess_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        Preprocess input image for model inference

        Args:
            image: Input tensor of shape (B, C, H, W) or (C, H, W)

        Returns:
            Preprocessed tensor ready for model
        """
        # Add batch dimension if needed
        if image.dim() == 3:
            image = image.unsqueeze(0)

        # Ensure image is on correct device
        image = image.to(self.device)

        # Normalize to [-1, 1] if not already
        if image.max() > 1.0:
            image = image / 255.0

        # Convert to [-1, 1] range
        image = (image - 0.5) * 2.0

        return image

    def postprocess_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        Postprocess model output

        Args:
            image: Model output tensor

        Returns:
            Processed tensor in [0, 1] range
        """
        # Convert from [-1, 1] to [0, 1]
        image = (image + 1.0) / 2.0

        # Clamp to valid range
        image = torch.clamp(image, 0.0, 1.0)

        return image

    @torch.no_grad()
    def colorize(self, grayscale_image: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Colorize a grayscale image

        Args:
            grayscale_image: Input grayscale tensor

        Returns:
            Colorized RGB tensor or None if model not loaded
        """
        if not self.model_loader.is_model_loaded():
            logger.error("No model loaded for inference")
            return None

        try:
            # Preprocess input
            input_tensor = self.preprocess_image(grayscale_image)

            # Run inference
            generator = self.model_loader.get_generator()
            generator.eval()

            with torch.no_grad():
                colorized = generator(input_tensor)

            # Postprocess output
            colorized = self.postprocess_image(colorized)

            return colorized

        except Exception as e:
            logger.error(f"Error during inference: {e}")
            return None

    def get_memory_usage(self) -> Dict[str, float]:
        """Get current GPU/CPU memory usage"""
        usage = {}

        if self.device.type == "cuda":
            usage["gpu_allocated"] = torch.cuda.memory_allocated() / 1024**3  # GB
            usage["gpu_cached"] = torch.cuda.memory_reserved() / 1024**3  # GB

        import psutil

        usage["cpu_percent"] = psutil.cpu_percent()
        usage["ram_used"] = psutil.virtual_memory().used / 1024**3  # GB

        return usage
