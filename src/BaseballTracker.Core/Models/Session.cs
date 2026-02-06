namespace BaseballTracker.Core.Models;

/// <summary>
/// Represents a practice or training session
/// </summary>
public class Session
{
    public Guid Id { get; set; }
    public DateTime StartTime { get; set; }
    public DateTime? EndTime { get; set; }
    public string? Notes { get; set; }
    
    // Collections
    public List<SwingData> Swings { get; set; } = new();
    
    // Computed properties
    public bool IsActive => EndTime == null;
    public TimeSpan? Duration => EndTime.HasValue ? EndTime.Value - StartTime : null;
}
