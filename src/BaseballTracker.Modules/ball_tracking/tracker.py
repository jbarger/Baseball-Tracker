"""
Ball Tracking Module
Detects and tracks baseball through video to calculate launch metrics

NOTE: This is a STUB implementation for Sprint 1
Real CV implementation will be added in Sprint 2
"""
import time
import random
import logging

logger = logging.getLogger(__name__)

class BallTracker:
    """
    Ball tracking using computer vision
    
    Sprint 1: Returns mock data
    Sprint 2: Will implement real CV with YOLO + Kalman filter
    """
    
    def __init__(self):
        logger.info("BallTracker initialized (STUB mode)")
    
    def process_video(self, video_path: str):
        # Import inside method to avoid circular import
        from api import BallTrackingResult, Point3D
        
        logger.info(f"Processing video (STUB): {video_path}")
        
        # Simulate processing time
        time.sleep(1.0)
        
        # Generate realistic mock data
        exit_velocity = round(random.uniform(85, 100), 1)
        launch_angle = round(random.uniform(10, 30), 1)
        spray_angle = round(random.uniform(-20, 20), 1)
        contact_frame = random.randint(30, 50)
        
        # Mock trajectory (simplified parabola)
        trajectory = [
            Point3D(x=0, y=0, z=0),
            Point3D(x=10, y=5, z=1),
            Point3D(x=20, y=15, z=3),
            Point3D(x=30, y=20, z=5),
            Point3D(x=40, y=22, z=6),
        ]
        
        result = BallTrackingResult(
            exit_velocity_mph=exit_velocity,
            launch_angle_degrees=launch_angle,
            spray_angle_degrees=spray_angle,
            contact_frame=contact_frame,
            trajectory_points=trajectory,
            confidence=0.92
        )
        
        logger.info(f"Ball tracking complete (STUB): {exit_velocity} mph, {launch_angle}°")
        return result