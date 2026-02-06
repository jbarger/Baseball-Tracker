using BaseballTracker.Core.Models;

namespace BaseballTracker.Core.Interfaces;

/// <summary>
/// Bridge for calling Python CV services from .NET
/// </summary>
public interface IPythonBridge
{
    /// <summary>
    /// Track baseball in video and calculate launch metrics
    /// </summary>
    Task<BallTrackingResult> TrackBallAsync(string videoPath, string cameraId = "default");
    
    /// <summary>
    /// Track bat through swing and measure bat speed
    /// </summary>
    Task<BatTrackingResult> TrackBatAsync(string videoPath, string cameraId = "default");
    
    /// <summary>
    /// Check if Python service is healthy
    /// </summary>
    Task<bool> HealthCheckAsync();
}
