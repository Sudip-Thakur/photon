import os
import cv2
import uuid
import time
import numpy as np
from typing import Optional, Dict, Any
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks,Form, Response
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
import aiofiles
from loguru import logger
from app.utils.config import settings
from app.schemas.models import (
  FrameResponse, VideoRequest, VideoResponse,  
    StatusResponse, ProcessingStatus, HealthResponse
)
from app.model.preprocessor import FrameProcessor
from app.model.video_processor import VideoHandler
from app.model.loader import model_loader
from app.api.websocket import camera_processor
from pathlib import Path
router = APIRouter()
processing_tasks: Dict[str, Dict[str, Any]] = {}
@router.get("/health", response_model=HealthResponse)
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model_loader.is_loaded(),
        "device": str(model_loader.device),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "version": settings.APP_VERSION
    }
@router.post("/process/frame")
async def process_single_frame(
    file: UploadFile = File(..., description="Image file (JPEG, PNG, BMP)"),
    return_format: str = Form("jpeg", description="Output format: jpeg or png"),
    quality: int = Form(90, ge=1, le=100, description="JPEG quality (1-100)")
):
    try:
        if file.content_type not in ["image/jpeg", "image/png", "image/bmp"]:
            raise HTTPException(400, "Only JPEG, PNG, and BMP files are supported")
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        input_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if input_image is None:
            raise HTTPException(400, "Failed to decode image")
        output_image, processing_time = FrameProcessor.process_single_frame(input_image)
        if return_format.lower() in ["jpg", "jpeg"]:
            media_type = "image/jpeg"
            ext = ".jpg"
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        else:  
            media_type = "image/png"
            ext = ".png"
            encode_params = [cv2.IMWRITE_PNG_COMPRESSION, min(9, quality // 10)]
        success, encoded_image = cv2.imencode(ext, output_image, encode_params)
        if not success:
            raise HTTPException(500, "Failed to encode output image")
        output_bytes = encoded_image.tobytes()
        return Response(
            content=output_bytes,
            media_type=media_type,
            headers={
                "X-Processing-Time": f"{processing_time:.2f}ms",
                "Content-Disposition": f"inline; filename=colorized_{file.filename}"
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing frame: {e}")
        raise HTTPException(500, f"Processing error: {str(e)}")
@router.post("/process/video/upload")
async def process_video_upload(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    batch_size: int = 4,
    max_frames: Optional[int] = None
):
    if file.size > settings.max_upload_size_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max size: {settings.MAX_VIDEO_SIZE_MB}MB"
        )
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in settings.allowed_extensions_list:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format. Allowed: {settings.ALLOWED_EXTENSIONS}"
        )
    processing_id = str(uuid.uuid4())
    temp_dir = Path(settings.TEMP_DIR)
    temp_dir.mkdir(exist_ok=True)
    input_path = temp_dir / f"upload_{processing_id}{file_ext}"
    output_path = temp_dir / f"processed_{processing_id}.mp4"
    try:
        async with aiofiles.open(input_path, 'wb') as out_file:
            content = await file.read()
            await out_file.write(content)
        logger.info(f"Video uploaded: {file.filename} -> {input_path}")
        processing_tasks[processing_id] = {
            "status": ProcessingStatus.PENDING,
            "progress": 0.0,
            "input_path": str(input_path),
            "output_path": str(output_path),
            "start_time": time.time(),
            "error": None
        }
        background_tasks.add_task(
            process_video_background,
            processing_id,
            str(input_path),
            str(output_path),
            batch_size,
            max_frames
        )
        return JSONResponse(
            status_code=202,  
            content={
                "message": "Video processing started",
                "processing_id": processing_id,
                "status_url": f"/api/v1/process/status/{processing_id}",
                "estimated_time": "Check status endpoint for updates"
            }
        )
    except Exception as e:
        logger.error(f"Error uploading video: {e}")
        if os.path.exists(input_path):
            os.remove(input_path)
        raise HTTPException(status_code=500, detail=str(e))
@router.get("/process/status/{processing_id}", response_model=StatusResponse)
async def get_processing_status(processing_id: str):
    if processing_id not in processing_tasks:
        raise HTTPException(status_code=404, detail="Processing ID not found")
    task = processing_tasks[processing_id]
    if task["status"] == ProcessingStatus.COMPLETED:
        output_file = task.get("output_file") or task.get("output_path")
        if output_file and os.path.exists(output_file):
            result_url = f"/api/v1/process/download/{processing_id}"
            task["result_url"] = result_url
        else:
            logger.warning(f"Output file not found for {processing_id}")
            logger.warning(f"Looking for: {output_file}")
            import glob
            temp_dir = settings.TEMP_DIR
            possible_files = glob.glob(f"{temp_dir}/processed_*{processing_id}*")
            possible_files.extend(glob.glob(f"{temp_dir}/*{processing_id}*"))
            if possible_files:
                found_file = possible_files[0]
                task["output_file"] = found_file
                task["result_url"] = f"/api/v1/process/download/{processing_id}"
                logger.info(f"Found file: {found_file}")
            else:
                task["status"] = ProcessingStatus.FAILED
                task["error"] = "Output file not found after processing"
                logger.error(f"No output file found for {processing_id}")
    return StatusResponse(
        processing_id=processing_id,
        status=task["status"],
        progress=task["progress"],
        estimated_time_remaining=None,
        result_url=task.get("result_url"),
        error=task.get("error")
    )
@router.get("/process/download/{processing_id}")
async def download_processed_video(processing_id: str):
    if processing_id not in processing_tasks:
        raise HTTPException(status_code=404, detail="Processing ID not found")
    task = processing_tasks[processing_id]
    if task["status"] != ProcessingStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Video processing not complete")
    possible_paths = []
    if "output_path" in task:
        possible_paths.append(task["output_path"])
    if "output_file" in task:
        possible_paths.append(task["output_file"])
    if "stats" in task and "output_path" in task["stats"]:
        possible_paths.append(task["stats"]["output_path"])
    import glob
    temp_dir = settings.TEMP_DIR
    pattern = f"{temp_dir}/*{processing_id}*"
    possible_paths.extend(glob.glob(pattern))
    pattern2 = f"{temp_dir}/processed_*"
    possible_paths.extend(glob.glob(pattern2))
    possible_paths = list(set(possible_paths))
    existing_paths = [p for p in possible_paths if os.path.exists(p)]
    if not existing_paths:
        logger.error(f"No output file found for {processing_id}. Searched:")
        logger.error(f"  Task output_path: {task.get('output_path')}")
        logger.error(f"  Task output_file: {task.get('output_file')}")
        logger.error(f"  Stats output_path: {task.get('stats', {}).get('output_path')}")
        logger.error(f"  Pattern search: {glob.glob(f'{temp_dir}/*{processing_id}*')}")
        logger.error(f"  All temp files: {glob.glob(f'{temp_dir}/*')}")
        raise HTTPException(
            status_code=404, 
            detail="Processed video not found. The file may have been deleted or failed to save."
        )
    output_path = existing_paths[0]
    file_ext = os.path.splitext(output_path)[1].lower()
    media_types = {
        '.mp4': 'video/mp4',
        '.avi': 'video/x-msvideo',
        '.mov': 'video/quicktime',
        '.mkv': 'video/x-matroska',
        '.webm': 'video/webm'
    }
    media_type = media_types.get(file_ext, 'application/octet-stream')
    filename = f"colorized_{processing_id[:8]}{file_ext}"
    logger.info(f"Serving file: {output_path} ({os.path.getsize(output_path) / 1024 / 1024:.2f} MB)")
    return FileResponse(
        path=output_path,
        filename=filename,
        media_type=media_type,
        headers={
            "Content-Disposition": f"attachment; filename={filename}",
            "X-Video-Size": f"{os.path.getsize(output_path)}",
            "X-Video-Path": output_path
        }
    )
@router.post("/process/realtime")
async def start_realtime_processing():
    return {
        "message": "Real-time processing endpoint",
        "websocket_url": f"ws://{settings.API_HOST}:{settings.API_PORT}/api/v1/ws/realtime",
        "status": "Coming soon"
    }
@router.get("/model/info")
async def get_model_info():
    if not model_loader.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")
    model = model_loader.get_model()
    total_params = sum(p.numel() for p in model.parameters())
    return {
        "name": "Pix2Pix U-Net",
        "version": "1.0",
        "input_size": [settings.MODEL_IMAGE_SIZE, settings.MODEL_IMAGE_SIZE, 1],  
        "output_size": [settings.MODEL_IMAGE_SIZE, settings.MODEL_IMAGE_SIZE, 3],  
        "parameters": total_params,
        "device": str(model_loader.device),
        "checkpoint": settings.MODEL_CHECKPOINT_PATH
    }
@router.get("/system/stats")
async def get_system_stats():
    import psutil
    import torch
    process = psutil.Process()
    memory_info = process.memory_info()
    gpu_memory = None
    if torch.cuda.is_available():
        gpu_memory = {
            "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
            "cached_mb": torch.cuda.memory_reserved() / 1024**2,
            "max_allocated_mb": torch.cuda.max_memory_allocated() / 1024**2
        }
    return {
        "cpu_percent": psutil.cpu_percent(),
        "memory_mb": memory_info.rss / 1024**2,
        "gpu_memory": gpu_memory,
        "active_tasks": len([t for t in processing_tasks.values() 
                           if t["status"] in [ProcessingStatus.PENDING, ProcessingStatus.PROCESSING]]),
        "completed_tasks": len([t for t in processing_tasks.values() 
                              if t["status"] == ProcessingStatus.COMPLETED]),
        "uptime_seconds": time.time() - app_start_time
    }
async def process_video_background(processing_id: str, input_path: str, 
                                 output_path: str, batch_size: int, 
                                 max_frames: Optional[int] = None):
    try:
        processing_tasks[processing_id].update({
            "status": ProcessingStatus.PROCESSING,
            "progress": 0.1
        })
        video_handler = VideoHandler()
        is_valid, message, video_info = video_handler.validate_video(input_path)
        if not is_valid:
            raise ValueError(f"Invalid video: {message}")
        processing_tasks[processing_id]["progress"] = 0.2
        stats = video_handler.process_video(
            input_path=input_path,
            output_path=output_path,
            batch_size=batch_size,
            max_frames=max_frames
        )
        output_file = stats.get("output_path", output_path)
        if os.path.exists(output_file):
            file_size = os.path.getsize(output_file) / (1024 * 1024)  
            logger.info(f"✅ Output file created: {output_file} ({file_size:.2f} MB)")
        else:
            logger.error(f"❌ Output file NOT found: {output_file}")
            possible_files = [
                output_file,
                output_file.replace('.mp4', '.avi'),
                output_file.replace('.avi', '.mp4'),
                output_path,  
            ]
            for file in possible_files:
                if os.path.exists(file):
                    logger.info(f"✅ Found alternative file: {file}")
                    stats["output_path"] = file
                    output_file = file
                    break
        processing_tasks[processing_id].update({
            "status": ProcessingStatus.COMPLETED,
            "progress": 1.0,
            "stats": stats,
            "end_time": time.time(),
            "processing_time": stats["total_time"],
            "output_file": output_file  
        })
        logger.info(f"Background processing completed: {processing_id}")
    except Exception as e:
        logger.error(f"Background processing failed for {processing_id}: {e}")
        logger.error(traceback.format_exc())
        processing_tasks[processing_id].update({
            "status": ProcessingStatus.FAILED,
            "error": str(e),
            "end_time": time.time()
        })
        try:
            if os.path.exists(input_path):
                os.remove(input_path)
            if 'output_file' in locals() and os.path.exists(output_file):
                os.remove(output_file)
        except Exception as cleanup_error:
            logger.error(f"Cleanup error: {cleanup_error}")
app_start_time = time.time()
@router.get("/camera/status")
async def get_camera_status():
    return {
        "camera_running": camera_processor.processing,
        "camera_id": 0 if camera_processor.cap else None,
        "resolution": f"{camera_processor.frame_width}x{camera_processor.frame_height}",
        "fps": camera_processor.fps,
        "model_device": str(model_loader.device),
        "model_loaded": model_loader.is_loaded()
    }
@router.post("/camera/start")
async def start_camera_endpoint(camera_id: int = 0):
    success = camera_processor.start_camera(camera_id)
    return {
        "success": success,
        "camera_id": camera_id,
        "message": "Camera started" if success else "Failed to start camera"
    }
@router.post("/camera/stop")
async def stop_camera_endpoint():
    camera_processor.stop_camera()
    return {"success": True, "message": "Camera stopped"}
@router.post("/camera/config")
async def configure_camera(
    width: int = 640,
    height: int = 480,
    fps: int = 30
):
    camera_processor.frame_width = width
    camera_processor.frame_height = height
    camera_processor.fps = fps
    return {
        "success": True,
        "resolution": f"{width}x{height}",
        "fps": fps,
        "message": "Camera configuration updated"
    }