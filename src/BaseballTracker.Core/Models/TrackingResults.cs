namespace BaseballTracker.Core.Models;

/// <summary>
/// Result from ball tracking analysis
/// </summary>
public record BallTrackingResult(
    double ExitVelocityMph,
    double LaunchAngleDegrees,
    double SprayAngleDegrees,
    int ContactFrame,
    List<Point3D> TrajectoryPoints,
    double Confidence
);

/// <summary>
/// Result from bat tracking analysis
/// </summary>
public record BatTrackingResult(
    double BatSpeedMph,
    double AttackAngleDegrees,
    Point3D ContactPoint,
    int ContactFrame,
    List<Point3D> SwingPathPoints,
    double Confidence
);

/// <summary>
/// 3D point in space (real-world coordinates)
/// </summary>
public record Point3D(double X, double Y, double Z);
