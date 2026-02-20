using Microsoft.EntityFrameworkCore;
using BaseballTracker.Core.Interfaces;
using BaseballTracker.Infrastructure.Data;
using BaseballTracker.Infrastructure.Python;

var builder = WebApplication.CreateBuilder(args);

// Raise upload limits for video files (default Kestrel limit is 30 MB)
builder.Services.Configure<Microsoft.AspNetCore.Http.Features.FormOptions>(o =>
{
    o.MultipartBodyLengthLimit = 500_000_000; // 500 MB
});
builder.WebHost.ConfigureKestrel(k =>
{
    k.Limits.MaxRequestBodySize = 500_000_000; // 500 MB
});

// Add services
builder.Services.AddControllers();
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen();

// Database
builder.Services.AddDbContext<BaseballTrackerDbContext>(options =>
    options.UseSqlite(builder.Configuration.GetConnectionString("DefaultConnection")));

// Repositories
builder.Services.AddScoped<ISwingRepository, SwingRepository>();
builder.Services.AddScoped<ISessionRepository, SessionRepository>();
builder.Services.AddScoped<IPlayerRepository, PlayerRepository>();

// Python bridge
builder.Services.AddHttpClient<IPythonBridge, PythonBridge>(client =>
{
    var pythonEndpoint = builder.Configuration["PythonService:Endpoint"] 
        ?? "http://localhost:8000";
    client.BaseAddress = new Uri(pythonEndpoint);
    client.Timeout = TimeSpan.FromMinutes(5); // CV processing can take time
});

// CORS for development
builder.Services.AddCors(options =>
{
    options.AddDefaultPolicy(policy =>
    {
        policy.AllowAnyOrigin()
              .AllowAnyMethod()
              .AllowAnyHeader();
    });
});

var app = builder.Build();

// Configure pipeline
if (app.Environment.IsDevelopment())
{
    app.UseSwagger();
    app.UseSwaggerUI();
}

app.UseCors();
app.UseDefaultFiles();
app.UseStaticFiles();
app.UseAuthorization();
app.MapControllers();

// Health check endpoint
app.MapGet("/health", () => Results.Ok(new { status = "healthy", service = "baseball-tracker-api" }));

// Ensure database is created
using (var scope = app.Services.CreateScope())
{
    var db = scope.ServiceProvider.GetRequiredService<BaseballTrackerDbContext>();
    db.Database.EnsureCreated();
}

app.Run();
