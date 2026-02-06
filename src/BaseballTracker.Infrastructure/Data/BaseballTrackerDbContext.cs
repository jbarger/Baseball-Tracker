using Microsoft.EntityFrameworkCore;
using BaseballTracker.Core.Models;

namespace BaseballTracker.Infrastructure.Data;

/// <summary>
/// Entity Framework database context for baseball tracker
/// </summary>
public class BaseballTrackerDbContext : DbContext
{
    public BaseballTrackerDbContext(DbContextOptions<BaseballTrackerDbContext> options)
        : base(options)
    {
    }

    public DbSet<SwingData> Swings => Set<SwingData>();
    public DbSet<Session> Sessions => Set<Session>();
    public DbSet<Player> Players => Set<Player>();

    protected override void OnModelCreating(ModelBuilder modelBuilder)
    {
        base.OnModelCreating(modelBuilder);

        // Configure SwingData
        modelBuilder.Entity<SwingData>(entity =>
        {
            entity.HasKey(e => e.Id);
            entity.HasIndex(e => e.SessionId);
            entity.HasIndex(e => e.PlayerId);
            entity.HasIndex(e => e.Timestamp);
            entity.HasIndex(e => e.Status);
            
            entity.HasOne(e => e.Session)
                .WithMany(s => s.Swings)
                .HasForeignKey(e => e.SessionId)
                .OnDelete(DeleteBehavior.Cascade);
                
            entity.HasOne(e => e.Player)
                .WithMany(p => p.Swings)
                .HasForeignKey(e => e.PlayerId)
                .OnDelete(DeleteBehavior.Restrict);
        });

        // Configure Session
        modelBuilder.Entity<Session>(entity =>
        {
            entity.HasKey(e => e.Id);
            entity.HasIndex(e => e.StartTime);
        });

        // Configure Player
        modelBuilder.Entity<Player>(entity =>
        {
            entity.HasKey(e => e.Id);
            entity.HasIndex(e => e.Name);
            
            entity.Property(e => e.Name)
                .IsRequired()
                .HasMaxLength(200);
                
            entity.Property(e => e.Email)
                .HasMaxLength(200);
        });
    }
}
