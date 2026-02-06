using BaseballTracker.Core.Models;

namespace BaseballTracker.Core.Interfaces;

/// <summary>
/// Repository for swing data persistence
/// </summary>
public interface ISwingRepository
{
    Task<Guid> CreateSwingAsync(SwingData swing);
    Task<SwingData?> GetSwingAsync(Guid swingId);
    Task<List<SwingData>> GetSwingsBySessionAsync(Guid sessionId);
    Task<List<SwingData>> GetSwingsByPlayerAsync(Guid playerId);
    Task UpdateSwingAsync(SwingData swing);
    Task DeleteSwingAsync(Guid swingId);
}

/// <summary>
/// Repository for session management
/// </summary>
public interface ISessionRepository
{
    Task<Guid> CreateSessionAsync(Session session);
    Task<Session?> GetSessionAsync(Guid sessionId);
    Task<List<Session>> GetRecentSessionsAsync(int count = 10);
    Task<Session?> GetActiveSessionAsync();
    Task UpdateSessionAsync(Session session);
}

/// <summary>
/// Repository for player management
/// </summary>
public interface IPlayerRepository
{
    Task<Guid> CreatePlayerAsync(Player player);
    Task<Player?> GetPlayerAsync(Guid playerId);
    Task<Player?> GetPlayerByNameAsync(string name);
    Task<List<Player>> GetAllPlayersAsync();
    Task UpdatePlayerAsync(Player player);
    Task DeletePlayerAsync(Guid playerId);
}
