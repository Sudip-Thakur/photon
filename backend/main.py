import os
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
import traceback
from datetime import datetime

from app.api.endpoints import router as api_router
from app.utils.config import settings
from app.model.loader import ModelLoader
from app.api.websocket import router as ws_router



# Create app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Real-time Pix2Pix video colorization API",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
if settings.ENABLE_CORS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception: {exc}")
    logger.error(traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={
            "message": "Internal server error",
            "detail": str(exc),
            "path": request.url.path
        }
    )

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize resources on startup"""
    logger.info("🚀 Starting Pix2Pix Video API")
    
    # Create necessary directories
    os.makedirs(settings.TEMP_DIR, exist_ok=True)
    os.makedirs("./logs", exist_ok=True)
    
    # Load model
    try:
        from app.model.loader import model_loader
        await model_loader.load_model()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup resources on shutdown"""
    logger.info("👋 Shutting down Pix2Pix Video API")
    
    # Cleanup temp directory
    import shutil
    if os.path.exists(settings.TEMP_DIR):
        shutil.rmtree(settings.TEMP_DIR)
        logger.info("🧹 Cleaned up temp directory")

# Include routers
app.include_router(api_router, prefix="/api/v1")

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Pix2Pix Video Colorization API",
        "version": settings.APP_VERSION,
        "docs": "/docs",
        "endpoints": {
            "health": "/api/v1/health",
            "process_frame": "/api/v1/process/frame",
            "process_video": "/api/v1/process/video",
            "realtime": "/api/v1/process/realtime"
        }
    }
app.include_router(ws_router)


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
        log_level="info"
    )