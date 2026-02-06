using Microsoft.EntityFrameworkCore;
using BaseballTracker.Core.Interfaces;
using BaseballTracker.Core.Models;

namespace BaseballTracker.Infrastructure.Data;

/// <summary>
/// Repository implementation for swing data
/// </summary>
public class SwingRepository : ISwingRepository
{
    private readonly BaseballTrackerDbContext _context;

    public SwingRepository(BaseballTrackerDbContext context)
    {
        _context = context;
    }

    public async Task<Guid> CreateSwingAsync(SwingData swing)
    {
        swing.Id = Guid.NewGuid();
        swing.Timestamp = DateTime.UtcNow;
        
        _context.Swings.Add(swing);
        await _context.SaveChangesAsync();
        
        return swing.Id;
    }

    public async Task<SwingData?> GetSwingAsync(Guid swingId)
    {
        return await _context.Swings
            .Include(s => s.Session)
            .Include(s => s.Player)
            .FirstOrDefaultAsync(s => s.Id == swingId);
    }

    public async Task<List<SwingData>> GetSwingsBySessionAsync(Guid sessionId)
    {
        return await _context.Swings
            .Where(s => s.SessionId == sessionId)
            .Include(s => s.Player)
            .OrderBy(s => s.Timestamp)
            .ToListAsync();
    }

    public async Task<List<SwingData>> GetSwingsByPlayerAsync(Guid playerId)
    {
        return await _context.Swings
            .Where(s => s.PlayerId == playerId)
            .Include(s => s.Session)
            .OrderByDescending(s => s.Timestamp)
            .ToListAsync();
    }

    public async Task UpdateSwingAsync(SwingData swing)
    {
        _context.Swings.Update(swing);
        await _context.SaveChangesAsync();
    }

    public async Task DeleteSwingAsync(Guid swingId)
    {
        var swing = await _context.Swings.FindAsync(swingId);
        if (swing != null)
        {
            _context.Swings.Remove(swing);
            await _context.SaveChangesAsync();
        }
    }
}
