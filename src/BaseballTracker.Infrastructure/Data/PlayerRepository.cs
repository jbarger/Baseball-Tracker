using Microsoft.EntityFrameworkCore;
using BaseballTracker.Core.Interfaces;
using BaseballTracker.Core.Models;

namespace BaseballTracker.Infrastructure.Data;

/// <summary>
/// Repository implementation for player management
/// </summary>
public class PlayerRepository : IPlayerRepository
{
    private readonly BaseballTrackerDbContext _context;

    public PlayerRepository(BaseballTrackerDbContext context)
    {
        _context = context;
    }

    public async Task<Guid> CreatePlayerAsync(Player player)
    {
        player.Id = Guid.NewGuid();
        player.CreatedAt = DateTime.UtcNow;
        
        _context.Players.Add(player);
        await _context.SaveChangesAsync();
        
        return player.Id;
    }

    public async Task<Player?> GetPlayerAsync(Guid playerId)
    {
        return await _context.Players
            .Include(p => p.Swings)
            .FirstOrDefaultAsync(p => p.Id == playerId);
    }

    public async Task<Player?> GetPlayerByNameAsync(string name)
    {
        return await _context.Players
            .FirstOrDefaultAsync(p => p.Name == name);
    }

    public async Task<List<Player>> GetAllPlayersAsync()
    {
        return await _context.Players
            .OrderBy(p => p.Name)
            .ToListAsync();
    }

    public async Task UpdatePlayerAsync(Player player)
    {
        _context.Players.Update(player);
        await _context.SaveChangesAsync();
    }

    public async Task DeletePlayerAsync(Guid playerId)
    {
        var player = await _context.Players.FindAsync(playerId);
        if (player != null)
        {
            _context.Players.Remove(player);
            await _context.SaveChangesAsync();
        }
    }
}
