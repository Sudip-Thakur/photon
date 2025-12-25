import os
from pathlib import Path
from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import validator, Field
import yaml

class Settings(BaseSettings):
    """Application settings - FLAT structure"""
    
    # App
    APP_NAME: str = "Pix2Pix Video API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # Model
    MODEL_CHECKPOINT_PATH: str = "./checkpoints/best.pth"
    MODEL_IMAGE_SIZE: int = 256
    MODEL_DEVICE: str = "cuda"  # or "cpu"
    MODEL_BATCH_SIZE: int = 4
    
    # Video
    TEMP_DIR: str = "./temp_videos"
    MAX_VIDEO_SIZE_MB: int = 100
    ALLOWED_EXTENSIONS: str  = ".mp4,.avi,.mov,.mkv,.jpg,.jpeg,.png,.bmp"  

    DEFAULT_FPS: int = 30
    
    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    MAX_UPLOAD_SIZE: int = 104857600  # 100MB
    ENABLE_CORS: bool = True
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "./logs/api.log"
    
    @validator("MODEL_DEVICE")
    def validate_device(cls, v):
        import torch
        if v == "cuda" and not torch.cuda.is_available():
            print("⚠ CUDA not available, falling back to CPU")
            return "cpu"
        return v
    
    @property
    def allowed_extensions_list(self) -> List[str]:
        """Convert comma-separated string to list"""
        return [ext.strip() for ext in self.ALLOWED_EXTENSIONS.split(",")]
    
    @property
    def max_upload_size_bytes(self) -> int:
        return self.MAX_UPLOAD_SIZE
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        # Allow reading from YAML
        extra = "ignore"  # This allows extra fields from YAML without errors

def load_yaml_config() -> dict:
    """Load configuration from YAML file - returns flat dict"""
    config_path = Path("config.yaml")
    if config_path.exists():
        with open(config_path, 'r') as f:
            yaml_data = yaml.safe_load(f)
            
            # Flatten the YAML structure if it's nested
            flat_config = {}
            if isinstance(yaml_data, dict):
                for key, value in yaml_data.items():
                    if isinstance(value, dict):
                        # If YAML has nested structure, flatten it
                        for sub_key, sub_value in value.items():
                            flat_key = f"{key}_{sub_key}".upper()
                            flat_config[flat_key] = sub_value
                    else:
                        flat_config[key.upper()] = value
            
            return flat_config
    return {}

# Load YAML config
yaml_config = load_yaml_config()

# Create settings instance, YAML will override defaults
settings = Settings(**yaml_config)

# Debug print
print(f"   Config loaded:")
print(f"   App: {settings.APP_NAME} v{settings.APP_VERSION}")
print(f"   Model: {settings.MODEL_CHECKPOINT_PATH}")
print(f"   Device: {settings.MODEL_DEVICE}")