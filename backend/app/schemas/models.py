# from pydantic import BaseModel, Field
# from typing import Optional, List, Dict, Any
# from enum import Enum

# class ProcessMode(str, Enum):
#     FRAME = "frame"
#     VIDEO = "video"
#     REALTIME = "realtime"

# class FrameRequest(BaseModel):
#     """Request for single frame processing"""
#     image_data: str  # Base64 encoded image
#     return_format: str = "base64"  # or "url"
    
# class FrameResponse(BaseModel):
#     """Response for frame processing"""
#     success: bool
#     processing_time_ms: float
#     output_format: str
#     output_data: Optional[str] = None  # Base64 or URL
#     error: Optional[str] = None

# class VideoRequest(BaseModel):
#     """Request for video processing"""
#     video_url: Optional[str] = None  # URL to video
#     process_mode: ProcessMode = ProcessMode.VIDEO
#     output_format: str = "mp4"
#     quality: int = Field(90, ge=1, le=100)
    
# class VideoResponse(BaseModel):
#     """Response for video processing"""
#     success: bool
#     processing_id: Optional[str] = None
#     estimated_time_seconds: Optional[float] = None
#     output_url: Optional[str] = None
#     status_url: Optional[str] = None
#     error: Optional[str] = None

# class ProcessingStatus(str, Enum):
#     PENDING = "pending"
#     PROCESSING = "processing"
#     COMPLETED = "completed"
#     FAILED = "failed"

# class StatusResponse(BaseModel):
#     """Processing status response"""
#     processing_id: str
#     status: ProcessingStatus
#     progress: float = Field(0.0, ge=0.0, le=1.0)
#     estimated_time_remaining: Optional[float] = None
#     result_url: Optional[str] = None
#     error: Optional[str] = None

# class HealthResponse(BaseModel):
#     """Health check response"""
#     status: str
#     model_loaded: bool
#     device: str
#     timestamp: str
#     version: str

# class ModelInfo(BaseModel):
#     """Model information"""
#     name: str
#     version: str
#     input_size: List[int]
#     output_size: List[int]
#     parameters: int
#     device: str


from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum

class ProcessMode(str, Enum):
    FRAME = "frame"
    VIDEO = "video"
    REALTIME = "realtime"

# Remove FrameRequest since we're using multipart now
class FrameResponse(BaseModel):
    """Response for frame processing"""
    success: bool
    processing_time_ms: float
    image_format: str = "jpeg"
    image_size_bytes: int
    latency_breakdown: Optional[Dict[str, float]] = None
    error: Optional[str] = None

class VideoRequest(BaseModel):
    """Request for video processing"""
    process_mode: ProcessMode = ProcessMode.VIDEO
    output_format: str = "mp4"
    quality: int = Field(90, ge=1, le=100)
    
class VideoResponse(BaseModel):
    """Response for video processing"""
    success: bool
    processing_id: Optional[str] = None
    estimated_time_seconds: Optional[float] = None
    output_url: Optional[str] = None
    status_url: Optional[str] = None
    error: Optional[str] = None

class ProcessingStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class StatusResponse(BaseModel):
    """Processing status response"""
    processing_id: str
    status: ProcessingStatus
    progress: float = Field(0.0, ge=0.0, le=1.0)
    estimated_time_remaining: Optional[float] = None
    result_url: Optional[str] = None
    error: Optional[str] = None

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    device: str
    timestamp: str
    version: str

class ModelInfo(BaseModel):
    """Model information"""
    name: str
    version: str
    input_size: List[int]
    output_size: List[int]
    parameters: int
    device: str