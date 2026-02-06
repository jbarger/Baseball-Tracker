namespace BaseballTracker.Core.Models;

/// <summary>
/// Represents a player/hitter in the system
/// </summary>
public class Player
{
    public Guid Id { get; set; }
    public string Name { get; set; } = string.Empty;
    public string? Email { get; set; }
    public DateTime CreatedAt { get; set; }
    
    // Player preferences
    public BattingSide BattingSide { get; set; }
    public string? Notes { get; set; }
    
    // Collections
    public List<SwingData> Swings { get; set; } = new();
}

/// <summary>
/// Which side of the plate the batter hits from
/// </summary>
public enum BattingSide
{
    Right,
    Left,
    Switch
}
