import torch
import torch.nn as nn
from typing import Optional
from loguru import logger
import time

from app.utils.config import settings
from app.model.pix2pix import GeneratorUNet

class ModelLoader:
    """Singleton model loader"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelLoader, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.device = torch.device(settings.MODEL_DEVICE)
        self.model: Optional[GeneratorUNet] = None
        self.transform = None
        self.is_loaded_flag = False
        self._initialized = True
        
        logger.info(f"ModelLoader initialized with device: {self.device}")
    
    async def load_model(self):
        """Load the model"""
        if self.is_loaded_flag:
            logger.info("Model already loaded")
            return True
        
        try:
            start_time = time.time()
            
            # Create model
            self.model = GeneratorUNet(
                in_channels=1,  # Grayscale input
                out_channels=3  # RGB output
            )
            
            # Load checkpoint
            checkpoint_path = settings.MODEL_CHECKPOINT_PATH
            if checkpoint_path and torch.cuda.is_available():
                checkpoint = torch.load(checkpoint_path, map_location=self.device,weights_only=True)
                if 'generator_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['generator_state_dict'])
                elif 'state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)
            
            # Move to device
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # Count parameters
            total_params = sum(p.numel() for p in self.model.parameters())
            
            # Create transform
            from torchvision import transforms
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((settings.MODEL_IMAGE_SIZE, settings.MODEL_IMAGE_SIZE)),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
            
            load_time = time.time() - start_time
            logger.info(f"   Model loaded in {load_time:.2f}s")
            logger.info(f"   Parameters: {total_params:,}")
            logger.info(f"   Device: {self.device}")
            
            self.is_loaded_flag = True
            return True
            
        except Exception as e:
            logger.error(f"  Failed to load model: {e}")
            self.is_loaded_flag = False
            raise
    
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.is_loaded_flag
    
    def get_model(self) -> GeneratorUNet:
        """Get the loaded model"""
        if not self.is_loaded_flag:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        return self.model
    
    def get_transform(self):
        """Get the image transform"""
        if not self.is_loaded_flag:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        return self.transform

# Global model loader instance
model_loader = ModelLoader()