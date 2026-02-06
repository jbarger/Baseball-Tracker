namespace BaseballTracker.Core.Models;

/// <summary>
/// Represents a single baseball swing with all captured metrics
/// </summary>
public class SwingData
{
    public Guid Id { get; set; }
    public Guid SessionId { get; set; }
    public Guid PlayerId { get; set; }
    public DateTime Timestamp { get; set; }
    
    // Ball metrics
    public double? ExitVelocityMph { get; set; }
    public double? LaunchAngleDegrees { get; set; }
    public double? SprayAngleDegrees { get; set; }
    
    // Bat metrics
    public double? BatSpeedMph { get; set; }
    public double? AttackAngleDegrees { get; set; }
    
    // Video references
    public string? BallCameraVideoPath { get; set; }
    public string? SwingCameraVideoPath { get; set; }
    
    // Processing metadata
    public SwingProcessingStatus Status { get; set; }
    public double? BallTrackingConfidence { get; set; }
    public double? BatTrackingConfidence { get; set; }
    public string? ProcessingErrors { get; set; }
    
    // Navigation properties
    public Session Session { get; set; } = null!;
    public Player Player { get; set; } = null!;
}

/// <summary>
/// Status of swing processing pipeline
/// </summary>
public enum SwingProcessingStatus
{
    Captured,
    Processing,
    Completed,
    Failed
}
