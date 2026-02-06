using Microsoft.EntityFrameworkCore;
using BaseballTracker.Core.Interfaces;
using BaseballTracker.Core.Models;

namespace BaseballTracker.Infrastructure.Data;

/// <summary>
/// Repository implementation for session management
/// </summary>
public class SessionRepository : ISessionRepository
{
    private readonly BaseballTrackerDbContext _context;

    public SessionRepository(BaseballTrackerDbContext context)
    {
        _context = context;
    }

    public async Task<Guid> CreateSessionAsync(Session session)
    {
        session.Id = Guid.NewGuid();
        session.StartTime = DateTime.UtcNow;
        
        _context.Sessions.Add(session);
        await _context.SaveChangesAsync();
        
        return session.Id;
    }

    public async Task<Session?> GetSessionAsync(Guid sessionId)
    {
        return await _context.Sessions
            .Include(s => s.Swings)
            .ThenInclude(sw => sw.Player)
            .FirstOrDefaultAsync(s => s.Id == sessionId);
    }

    public async Task<List<Session>> GetRecentSessionsAsync(int count = 10)
    {
        return await _context.Sessions
            .OrderByDescending(s => s.StartTime)
            .Take(count)
            .ToListAsync();
    }

    public async Task<Session?> GetActiveSessionAsync()
    {
        return await _context.Sessions
            .Where(s => s.EndTime == null)
            .OrderByDescending(s => s.StartTime)
            .FirstOrDefaultAsync();
    }

    public async Task UpdateSessionAsync(Session session)
    {
        _context.Sessions.Update(session);
        await _context.SaveChangesAsync();
    }
}
