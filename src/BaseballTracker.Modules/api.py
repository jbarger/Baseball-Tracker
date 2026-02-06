"""
Baseball Tracker - Python CV API
Provides computer vision services for ball and bat tracking
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import logging

# Import tracking modules
from ball_tracking.tracker import BallTracker
from bat_tracking.tracker import BatTracker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Baseball Tracker CV API",
    description="Computer vision services for baseball swing analysis",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Data Models
# ============================================================================

class Point3D(BaseModel):
    """3D point in space"""
    x: float
    y: float
    z: float

class BallTrackingResult(BaseModel):
    """Result from ball tracking analysis"""
    exit_velocity_mph: float
    launch_angle_degrees: float
    spray_angle_degrees: float
    contact_frame: int
    trajectory_points: List[Point3D]
    confidence: float

class BatTrackingResult(BaseModel):
    """Result from bat tracking analysis"""
    bat_speed_mph: float
    attack_angle_degrees: float
    contact_point: Point3D
    contact_frame: int
    swing_path_points: List[Point3D]
    confidence: float

class TrackingRequest(BaseModel):
    """Request for tracking analysis"""
    video_path: str
    camera_id: Optional[str] = "default"

# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Baseball Tracker CV API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "baseball-tracker-cv"
    }

@app.post("/track/ball", response_model=BallTrackingResult)
async def track_ball(request: TrackingRequest):
    """
    Track baseball in video and calculate launch metrics
    
    Args:
        request: TrackingRequest with video path and camera ID
        
    Returns:
        BallTrackingResult with exit velocity, launch angle, etc.
    """
    try:
        logger.info(f"Processing ball tracking for: {request.video_path}")
        
        # Validate video exists
        if not os.path.exists(request.video_path):
            raise HTTPException(
                status_code=404,
                detail=f"Video not found: {request.video_path}"
            )
        
        # Process video
        tracker = BallTracker()
        result = tracker.process_video(request.video_path)
        
        logger.info(f"Ball tracking complete: {result.exit_velocity_mph} mph, "
                   f"{result.launch_angle_degrees}°")
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ball tracking failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/track/bat", response_model=BatTrackingResult)
async def track_bat(request: TrackingRequest):
    """
    Track bat through swing and measure bat speed
    
    Args:
        request: TrackingRequest with video path and camera ID
        
    Returns:
        BatTrackingResult with bat speed, attack angle, etc.
    """
    try:
        logger.info(f"Processing bat tracking for: {request.video_path}")
        
        # Validate video exists
        if not os.path.exists(request.video_path):
            raise HTTPException(
                status_code=404,
                detail=f"Video not found: {request.video_path}"
            )
        
        # Process video
        tracker = BatTracker()
        result = tracker.process_video(request.video_path)
        
        logger.info(f"Bat tracking complete: {result.bat_speed_mph} mph")
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Bat tracking failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# Startup/Shutdown
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("Baseball Tracker CV API starting up...")
    logger.info("Ready to process swing analysis requests")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Baseball Tracker CV API shutting down...")
